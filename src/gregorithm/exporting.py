"""Export of published result tables to standalone CSV or Parquet files.

`export_table` streams one Parquet table of an immutable generation into a new file outside the
result store. The export contains only the selected numerical columns: no manifest, fingerprint
or Parquet schema metadata is embedded. The written file is re-read and compared value by value
(including nulls and exact float bits) before it appears at its destination.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import uuid
from collections.abc import Callable, Iterable
from contextlib import suppress
from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.csv as pacsv  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from .contracts import ExportFormat, ExportReceipt, ExportRequest, OperationId, ResourceEstimate
from .diagnostics import Diagnostic, DiagnosticCode, DiagnosticKind
from .progress import (
    CancelToken,
    OperationCancelled,
    ProgressEvent,
    ProgressSink,
    check_cancelled,
    report_progress,
)
from .storage import open_result

FaultHook = Callable[[str], None]


class ExportError(RuntimeError):
    """Structured export failure; `diagnostic` holds the reason code.

    Codes include `EXPORT_INVALID_SOURCE`, `EXPORT_DESTINATION_EXISTS`, `EXPORT_IO_ERROR` and
    `INSUFFICIENT_DISK_SPACE`.
    """

    def __init__(self, diagnostic: Diagnostic) -> None:
        """Store the diagnostic and use its message as text."""
        self.diagnostic = diagnostic
        super().__init__(diagnostic.message)


def _error(code: DiagnosticCode, message: str) -> ExportError:
    """Build an `ExportError` with an error diagnostic for the export operation."""
    return ExportError(
        Diagnostic(
            code,
            DiagnosticKind.ERROR,
            message,
            operation=OperationId.EXPORT_TABLE.value,
        )
    )


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(request: ExportRequest) -> tuple[Path, pq.ParquetFile, pa.Schema]:
    """Open and validate the requested table of a published result.

    Returns the table path, an open `ParquetFile` and the output schema without metadata. Only
    files declared by the manifest under a public directory may be exported.
    """
    info = open_result(request.result_path)
    candidate = info.ref.path / Path(request.table_path)
    public_roots = {"points", "indexes", "overview", "tables", "curves"}
    if (
        Path(request.table_path).parts[0] not in public_roots
        or candidate not in info.files
        or not candidate.is_file()
        or candidate.is_symlink()
    ):
        raise _error(
            DiagnosticCode.EXPORT_INVALID_SOURCE,
            "The selected table is not an immutable file in the published result.",
        )
    try:
        parquet = pq.ParquetFile(candidate)
        source_schema = parquet.schema_arrow
    except (OSError, pa.ArrowException) as error:
        raise _error(
            DiagnosticCode.EXPORT_INVALID_SOURCE,
            "The selected result table is not readable Parquet.",
        ) from error
    columns = source_schema.names if request.columns is None else list(request.columns)
    if len(columns) != len(set(columns)) or any(
        name not in source_schema.names for name in columns
    ):
        parquet.close()
        raise _error(DiagnosticCode.EXPORT_INVALID_SOURCE, "An export column is unavailable.")
    schema = pa.schema(
        [
            pa.field(
                source_schema.field(name).name,
                source_schema.field(name).type,
                nullable=source_schema.field(name).nullable,
            )
            for name in columns
        ],
        metadata=None,
    )
    return candidate, parquet, schema


def _arrays_exact(left: pa.Array, right: pa.Array) -> bool:
    """Return True if two Arrow arrays have equal type, nulls and exact values/bits."""
    if left.type != right.type or len(left) != len(right):
        return False
    if left.is_null().to_pylist() != right.is_null().to_pylist():
        return False
    if pa.types.is_floating(left.type):
        left_values = left.fill_null(0.0).to_numpy(zero_copy_only=False)
        right_values = right.fill_null(0.0).to_numpy(zero_copy_only=False)
        unsigned = np.dtype(f"u{left_values.dtype.itemsize}")
        return bool(np.array_equal(left_values.view(unsigned), right_values.view(unsigned)))
    return bool(left.equals(right))


def _validate_batches(
    source_batches: Iterable[pa.RecordBatch],
    exported_batches: Iterable[pa.RecordBatch],
    schema: pa.Schema,
    expected_rows: int,
) -> None:
    """Compare source and exported batches row by row; raise ValueError on change."""
    source_iterator = iter(source_batches)
    exported_iterator = iter(exported_batches)
    source: pa.RecordBatch | None = None
    exported: pa.RecordBatch | None = None
    source_offset = exported_offset = validated = 0
    while True:
        if source is None or source_offset == source.num_rows:
            source = next(source_iterator, None)
            source_offset = 0
        if exported is None or exported_offset == exported.num_rows:
            exported = next(exported_iterator, None)
            exported_offset = 0
        if source is None or exported is None:
            if source is not None or exported is not None:
                raise ValueError("Export row count changed")
            break
        if exported.schema.names != schema.names or any(
            exported.schema.field(index).type != field.type for index, field in enumerate(schema)
        ):
            raise ValueError("Export schema changed")
        count = min(source.num_rows - source_offset, exported.num_rows - exported_offset)
        left = source.slice(source_offset, count)
        right = exported.slice(exported_offset, count)
        if any(
            not _arrays_exact(left.column(index), right.column(index))
            for index in range(len(schema))
        ):
            raise ValueError("Exported values, nulls or row order changed")
        source_offset += count
        exported_offset += count
        validated += count
    if validated != expected_rows:
        raise ValueError("Export row count changed")


def _validate_export(
    path: Path, source_path: Path, request: ExportRequest, schema: pa.Schema, rows: int
) -> None:
    """Re-read the temporary export and verify schema, row count and values."""
    source = pq.ParquetFile(source_path)
    try:
        if request.format is ExportFormat.PARQUET:
            exported = pq.ParquetFile(path)
            try:
                if exported.schema_arrow.metadata is not None:
                    raise ValueError("Parquet schema metadata was not stripped")
                if exported.schema_arrow != schema or exported.metadata.num_rows != rows:
                    raise ValueError("Parquet export schema or row count changed")
                _validate_batches(
                    source.iter_batches(batch_size=request.batch_size, columns=schema.names),
                    exported.iter_batches(batch_size=request.batch_size),
                    schema,
                    rows,
                )
            finally:
                exported.close()
        else:
            exported_csv = pacsv.open_csv(
                path,
                read_options=pacsv.ReadOptions(block_size=max(4_096, request.batch_size * 128)),
                convert_options=pacsv.ConvertOptions(
                    column_types={field.name: field.type for field in schema},
                    strings_can_be_null=False,
                ),
            )
            _validate_batches(
                source.iter_batches(batch_size=request.batch_size, columns=schema.names),
                exported_csv,
                schema,
                rows,
            )
    finally:
        source.close()


def estimate_export(request: ExportRequest) -> ResourceEstimate:
    """Estimate the bytes an export will write.

    The estimate uses Parquet metadata of the selected columns; CSV output is assumed to be up to
    three times the uncompressed size. The published source is not counted again because it
    already occupies disk space.
    """
    source, parquet, schema = _source(request)
    try:
        selected = set(schema.names)
        uncompressed = 0
        for row_group in range(parquet.metadata.num_row_groups):
            metadata = parquet.metadata.row_group(row_group)
            for column in range(metadata.num_columns):
                item = metadata.column(column)
                if item.path_in_schema in selected:
                    uncompressed += item.total_uncompressed_size
        selected_fraction = len(selected) / max(1, len(parquet.schema_arrow.names))
        expected = max(1, math.ceil(source.stat().st_size * selected_fraction))
        if request.format is ExportFormat.CSV:
            expected = max(expected, uncompressed)
            pessimistic = max(expected, uncompressed * 3 + 65_536)
        else:
            pessimistic = max(expected, uncompressed + 65_536)
        return ResourceEstimate(expected, pessimistic, (pessimistic * 11 + 9) // 10)
    finally:
        parquet.close()


def export_table(
    request: ExportRequest,
    *,
    progress: ProgressSink | None = None,
    cancel: CancelToken | None = None,
    fault_hook: FaultHook | None = None,
) -> ExportReceipt:
    """Export one table of a published result to a new CSV or Parquet file.

    Args:
        request: `ExportRequest` naming the generation, table, destination and format.
        progress: Optional progress sink receiving row counts.
        cancel: Optional cancellation token.
        fault_hook: Testing hook called at named checkpoints; leave it None.

    Returns:
        An `ExportReceipt` with row count, size and SHA-256 of the written file.

    Existing destinations are never overwritten. Data are written to a hidden temporary file,
    validated and then linked into place, so a failed or cancelled export leaves no partial file.
    The destination filesystem must support hard links (all common local filesystems do).
    """
    check_cancelled(cancel, OperationId.EXPORT_TABLE.value)
    request.destination.parent.mkdir(parents=True, exist_ok=True)
    if request.destination.exists() or request.destination.is_symlink():
        raise _error(
            DiagnosticCode.EXPORT_DESTINATION_EXISTS,
            "The export destination already exists and will not be overwritten.",
        )
    estimate = estimate_export(request)
    if shutil.disk_usage(request.destination.parent).free < estimate.required_free_bytes:
        raise _error(
            DiagnosticCode.INSUFFICIENT_DISK_SPACE,
            "Insufficient incremental free space for the export.",
        )
    source_path, parquet, schema = _source(request)
    columns = schema.names
    total = parquet.metadata.num_rows
    temporary = request.destination.parent / (
        f".{request.destination.name}.{uuid.uuid4().hex}.exporting"
    )
    parquet_writer: pq.ParquetWriter | None = None
    csv_writer: pacsv.CSVWriter | None = None
    rows = 0
    try:
        if fault_hook is not None:
            fault_hook("before_open")
        if request.format is ExportFormat.PARQUET:
            parquet_writer = pq.ParquetWriter(temporary, schema, compression="zstd")
        else:
            csv_writer = pacsv.CSVWriter(
                temporary,
                schema,
                write_options=pacsv.WriteOptions(include_header=True, quoting_style="needed"),
            )
        for batch in parquet.iter_batches(batch_size=request.batch_size, columns=columns):
            check_cancelled(cancel, OperationId.EXPORT_TABLE.value)
            clean = pa.RecordBatch.from_arrays(batch.columns, schema=schema)
            if parquet_writer is not None:
                parquet_writer.write_batch(clean)
            else:
                assert csv_writer is not None
                csv_writer.write_batch(clean)
            rows += clean.num_rows
            report_progress(
                progress,
                ProgressEvent(OperationId.EXPORT_TABLE, "write", rows, total, "rows"),
            )
            if fault_hook is not None:
                fault_hook("after_batch")
        if parquet_writer is not None:
            parquet_writer.close()
            parquet_writer = None
        if csv_writer is not None:
            csv_writer.close()
            csv_writer = None
        parquet.close()
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        if fault_hook is not None:
            fault_hook("before_validate")
        _validate_export(temporary, source_path, request, schema, rows)
        check_cancelled(cancel, OperationId.EXPORT_TABLE.value)
        if fault_hook is not None:
            fault_hook("before_publish")
        try:
            os.link(temporary, request.destination)
        except FileExistsError as error:
            raise _error(
                DiagnosticCode.EXPORT_DESTINATION_EXISTS,
                "The export destination appeared during publication and was preserved.",
            ) from error
        temporary.unlink()
        receipt = ExportReceipt(
            request.destination.resolve(strict=True),
            request.format,
            rows,
            request.destination.stat().st_size,
            _sha256(request.destination),
            tuple(columns),
        )
        report_progress(
            progress,
            ProgressEvent(OperationId.EXPORT_TABLE, "complete", rows, total, "rows"),
        )
        return receipt
    except ExportError:
        raise
    except OperationCancelled:
        raise
    except Exception as error:
        raise _error(DiagnosticCode.EXPORT_IO_ERROR, "Export failed before publication.") from error
    finally:
        parquet.close()
        if parquet_writer is not None:
            with suppress(Exception):
                parquet_writer.close()
        if csv_writer is not None:
            with suppress(Exception):
                csv_writer.close()
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
