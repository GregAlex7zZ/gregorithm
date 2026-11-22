"""Source inspection and canonical import of time, current and voltage data.

Supported sources are CSV, TXT (delimited text) and Parquet files. Import converts the three
mapped columns to canonical units (seconds, amperes, volts) and cleans the series in this order:

1. remove rows whose current or voltage is missing, non-numeric or non-finite; 2. validate
elapsed time on the surviving rows (invalid time aborts the import); 3. keep the first surviving
row of each timestamp and drop later duplicates; 4. abort if elapsed time decreases.

At least two rows must remain. Positive current always means charge; the source sign is never
inverted or inferred. The source file is opened read-only and never modified. The result is an
immutable generation containing ``points/canonical.parquet`` (non-null float64 `time_s`,
`current_a`, `voltage_v`) and ``import_report.json``.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.csv as pacsv  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from .contracts import OperationId, ResourceEstimate, ResultRef
from .diagnostics import Diagnostic, DiagnosticCode, DiagnosticKind
from .identity import hash_dependencies, hash_parameters
from .progress import CancelToken, ProgressEvent, ProgressSink, check_cancelled, report_progress
from .storage import ResultStore, StagingGeneration, StorageError

CANONICAL_SCHEMA = pa.schema(
    [
        pa.field("time_s", pa.float64(), nullable=False),
        pa.field("current_a", pa.float64(), nullable=False),
        pa.field("voltage_v", pa.float64(), nullable=False),
    ]
)
PARQUET_ROW_GROUP_SIZE = 262_144
DEFAULT_BATCH_SIZE = 65_536
_FIXED_OUTPUT_OVERHEAD_BYTES = 1_048_576
_PARQUET_OVERHEAD_NUMERATOR = 105
_PARQUET_OVERHEAD_DENOMINATOR = 100
_DISK_RESERVE_NUMERATOR = 110
_DISK_RESERVE_DENOMINATOR = 100
_NUMBER_PATTERN = r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
_DURATION_PATTERN = (
    r"^(?P<h>\d+):(?P<m>[0-5]\d):(?P<s>[0-5]\d)"
    r"(?:(?P<sep>[.,])(?P<f>\d+))?$"
)
_UNIT_FACTORS: dict[str, dict[str, float]] = {
    "time": {"s": 1.0, "ms": 1e-3, "min": 60.0, "h": 3600.0},
    "current": {"A": 1.0, "mA": 1e-3, "uA": 1e-6},
    "voltage": {"V": 1.0, "mV": 1e-3},
}
_UNIT_DIVISORS = {
    ("time", "ms"): 1_000.0,
    ("current", "mA"): 1_000.0,
    ("current", "uA"): 1_000_000.0,
    ("voltage", "mV"): 1_000.0,
}
_RECOGNIZED = {
    ("time_s", "current_a", "voltage_v"): ("s", "A", "V"),
    ("time_s", "current_mA", "voltage_V"): ("s", "mA", "V"),
}


class SourceKind(StrEnum):
    """Recognized source containers, derived from the file extension."""

    CSV = "csv"
    TXT = "txt"
    PARQUET = "parquet"


class TimeFormat(StrEnum):
    """How the mapped time column is written.

    `NUMERIC` is a number in the mapped time unit. `DURATION` is elapsed-duration text
    ``H+:MM:SS[.fraction]`` (hours may exceed 23, e.g. ``179:22:08``); it is not a clock time.
    """

    NUMERIC = "numeric"
    DURATION = "duration"


@dataclass(frozen=True, slots=True)
class ColumnMapping:
    """Explicit assignment of source columns and units to canonical roles.

    Attributes:
        time_column, current_column, voltage_column: Three distinct source column names.
        time_unit: ``"s"``, ``"ms"``, ``"min"`` or ``"h"`` (must be ``"s"`` for durations).
        current_unit: ``"A"``, ``"mA"`` or ``"uA"``.
        voltage_unit: ``"V"`` or ``"mV"``.
        time_format: `TimeFormat.NUMERIC` (default) or `TimeFormat.DURATION`.

    A mapping is required unless the header is exactly recognized (see `inspect_source`). There
    is deliberately no polarity option: convert sources that use negative-for-charge first.

    Example:
        ColumnMapping("Test Time", "Current", "Voltage", "s", "mA", "V")
    """

    time_column: str
    current_column: str
    voltage_column: str
    time_unit: str
    current_unit: str
    voltage_unit: str
    time_format: TimeFormat = TimeFormat.NUMERIC

    def validate(self) -> None:
        """Raise ValueError for missing/duplicate columns or unsupported units."""
        columns = (self.time_column, self.current_column, self.voltage_column)
        if any(not value for value in columns) or len(set(columns)) != 3:
            raise ValueError("Three distinct mapped columns are required")
        for role, unit in (
            ("time", self.time_unit),
            ("current", self.current_unit),
            ("voltage", self.voltage_unit),
        ):
            if unit not in _UNIT_FACTORS[role]:
                raise ValueError(f"Unsupported {role} unit")
        if self.time_format is TimeFormat.DURATION and self.time_unit != "s":
            raise ValueError("Elapsed-duration text is already expressed in seconds")


@dataclass(frozen=True, slots=True)
class TextOptions:
    """Parsing options for CSV and TXT sources.

    Attributes:
        delimiter: Single field-separator character (default ``","``).
        decimal_separator: ``"."`` or ``","``; applies to numbers and duration fractions.
        encoding: Text encoding (default ``"utf-8"``).
        skip_rows: Lines to skip before the header row.
    """

    delimiter: str = ","
    decimal_separator: str = "."
    encoding: str = "utf-8"
    skip_rows: int = 0

    def validate(self) -> None:
        """Raise ValueError for invalid separators or negative `skip_rows`."""
        if len(self.delimiter) != 1 or self.decimal_separator not in {".", ","}:
            raise ValueError("Text separators must be explicit single characters")
        if self.skip_rows < 0:
            raise ValueError("skip_rows must not be negative")


_DEFAULT_TEXT_OPTIONS = TextOptions()


@dataclass(frozen=True, slots=True)
class SourceInspection:
    """Bounded preview of a source file, produced by `inspect_source`.

    Attributes:
        source: Resolved source path.
        kind: Detected `SourceKind`.
        columns: Column names found in the header or schema.
        row_count: Exact row count for Parquet, None for text (not counted).
        preview: First rows as strings (None for nulls).
        suggested_mapping: Mapping for a recognized header, otherwise None.
        mapping_confirmed: True when `suggested_mapping` can be used without confirmation.
    """

    source: Path
    kind: SourceKind
    columns: tuple[str, ...]
    row_count: int | None
    preview: tuple[tuple[str | None, ...], ...]
    suggested_mapping: ColumnMapping | None
    mapping_confirmed: bool


@dataclass(frozen=True, slots=True)
class ImportRequest:
    """Parameters of `import_source` and `estimate_import`.

    Attributes:
        source: CSV, TXT or Parquet file to read (never modified).
        caller_root: Result root directory owned by the caller.
        sample_id: Identifier of the sample (cell); results go below ``sample-<sample_id>``.
        job_id: Unique identifier of this run within the sample.
        generation_id: Unique identifier of the generation to publish.
        mapping: Explicit `ColumnMapping`; None accepts only recognized headers.
        text_options: `TextOptions` for CSV/TXT sources.
        batch_size: Rows processed per streaming batch.

    Identifiers must be 1-128 ASCII letters, digits, dots, underscores or hyphens.
    """

    source: Path
    caller_root: Path
    sample_id: str
    job_id: str
    generation_id: str
    mapping: ColumnMapping | None = None
    text_options: TextOptions = TextOptions()
    batch_size: int = DEFAULT_BATCH_SIZE

    def validate(self) -> None:
        """Validate batch size, text options and the mapping (if any)."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.text_options.validate()
        if self.mapping is not None:
            self.mapping.validate()


@dataclass(frozen=True, slots=True)
class ImportReport:
    """Outcome of an import.

    Attributes:
        source_rows: Data rows read from the source.
        retained_rows: Rows kept in the canonical table.
        removed_invalid_measurement: Rows dropped for invalid current or voltage.
        removed_duplicate_time: Rows dropped as later duplicates of a timestamp.
        diagnostics: Import warnings (removed rows, duplicate timestamps).
        content_hash: SHA-256 of the retained canonical rows. Use it as the input fingerprint
            for `ElaborationRequest`; it identifies content, not the file name.

    Import warnings describe the import only; they are not propagated as quality flags.
    """

    source_rows: int
    retained_rows: int
    removed_invalid_measurement: int
    removed_duplicate_time: int
    diagnostics: tuple[Diagnostic, ...]
    content_hash: str


@dataclass(frozen=True, slots=True)
class ImportResult:
    """Published canonical import: result reference `ref` and `report`."""

    ref: ResultRef
    report: ImportReport


@dataclass(frozen=True, slots=True)
class ImportAdmission:
    """Disk-space estimate for an import, returned by `estimate_import`.

    Attributes:
        estimate: Expected, pessimistic and required free bytes.
        available_bytes: Free bytes measured on the destination volume.
        estimated_source_rows: Expected rows (exact for Parquet, sampled for text).
        pessimistic_source_rows: Upper bound used for the pessimistic estimate.
        assumptions: Human-readable assumptions behind the numbers.
    """

    estimate: ResourceEstimate
    available_bytes: int
    estimated_source_rows: int
    pessimistic_source_rows: int
    assumptions: tuple[str, ...]

    @property
    def admitted(self) -> bool:
        """True when the available space covers the required free bytes."""
        return self.available_bytes >= self.estimate.required_free_bytes


class SourceImportError(RuntimeError):
    """Structured import or inspection failure; `diagnostic` holds the reason.

    Typical codes: `IMPORT_MAPPING_REQUIRED`, `IMPORT_INVALID_TIME` (with `source_row`),
    `IMPORT_TIME_DECREASE`, `IMPORT_INSUFFICIENT_ROWS`, `IMPORT_UNSUPPORTED_FORMAT`,
    `IMPORT_PARSE_ERROR`, `IMPORT_IO_ERROR` and `INSUFFICIENT_DISK_SPACE`. The original
    exception, when any, is chained as `__cause__`.
    """

    def __init__(self, diagnostic: Diagnostic) -> None:
        """Store the diagnostic and use its message as text."""
        self.diagnostic = diagnostic
        super().__init__(diagnostic.message)


def _error(code: DiagnosticCode, message: str, **context: Any) -> SourceImportError:
    """Build a `SourceImportError` with an error diagnostic and optional context."""
    return SourceImportError(
        Diagnostic(code=code, kind=DiagnosticKind.ERROR, message=message, **context)
    )


def _kind(source: Path) -> SourceKind:
    """Return the `SourceKind` for a file extension or raise `IMPORT_UNSUPPORTED_FORMAT`."""
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return SourceKind.CSV
    if suffix == ".txt":
        return SourceKind.TXT
    if suffix in {".parquet", ".pq"}:
        return SourceKind.PARQUET
    raise _error(DiagnosticCode.IMPORT_UNSUPPORTED_FORMAT, "Unsupported source format.")


def _nearest_existing(path: Path) -> Path:
    """Return the nearest existing ancestor of `path` (for disk-space probes)."""
    candidate = path.resolve(strict=False)
    while not candidate.exists():
        if candidate.parent == candidate:
            raise _error(DiagnosticCode.IMPORT_IO_ERROR, "No existing parent for result root.")
        candidate = candidate.parent
    return candidate


def _available_disk_bytes(path: Path) -> int:
    """Return free bytes on the volume that will hold `path`."""
    return shutil.disk_usage(_nearest_existing(path)).free


def _text_row_estimate(source: Path, options: TextOptions) -> int:
    """Estimate text data rows from the mean line length of a 64 KiB sample."""
    sample_bytes = min(source.stat().st_size, 64 * 1024)
    if sample_bytes == 0:
        return 0
    with source.open("rb") as stream:
        sample = stream.read(sample_bytes)
    line_count = max(1, sample.count(b"\n"))
    average_line_bytes = max(1.0, len(sample) / line_count)
    return max(0, math.ceil(source.stat().st_size / average_line_bytes) - 1 - options.skip_rows)


def _estimate_import(
    request: ImportRequest,
    *,
    available_bytes: Callable[[Path], int] = _available_disk_bytes,
) -> ImportAdmission:
    """Compute the import admission estimate (see `estimate_import`)."""
    request.validate()
    source = request.source.resolve(strict=True)
    kind = _kind(source)
    assumptions: tuple[str, ...]
    if kind is SourceKind.PARQUET:
        rows = pq.ParquetFile(source).metadata.num_rows
        pessimistic_rows = rows
        assumptions = (
            "Parquet row count is exact metadata.",
            "Canonical payload is three float64 values per retained row.",
        )
    else:
        rows = _text_row_estimate(source, request.text_options)
        pessimistic_rows = source.stat().st_size + 1
        assumptions = (
            "Expected text rows use a bounded 64 KiB mean-line-length sample.",
            "Pessimistic text rows are bounded by source bytes plus one.",
            "Canonical payload is three float64 values per retained row.",
        )
    expected = rows * 24 + _FIXED_OUTPUT_OVERHEAD_BYTES
    pessimistic_payload = pessimistic_rows * 24
    pessimistic = (
        math.ceil(pessimistic_payload * _PARQUET_OVERHEAD_NUMERATOR / _PARQUET_OVERHEAD_DENOMINATOR)
        + _FIXED_OUTPUT_OVERHEAD_BYTES
    )
    pessimistic = max(expected, pessimistic)
    required = math.ceil(pessimistic * _DISK_RESERVE_NUMERATOR / _DISK_RESERVE_DENOMINATOR)
    estimate = ResourceEstimate(expected, pessimistic, required)
    return ImportAdmission(
        estimate,
        int(available_bytes(request.caller_root)),
        rows,
        pessimistic_rows,
        (
            *assumptions,
            "Fixed overhead covers manifest, report, pointer and bounded Parquet metadata.",
            "Required free bytes are pessimistic incremental demand plus 10% reserve.",
            "Existing generations are excluded because they already occupy measured disk.",
        ),
    )


def estimate_import(
    request: ImportRequest,
    *,
    available_bytes: Callable[[Path], int] = _available_disk_bytes,
) -> ImportAdmission:
    """Estimate the disk space an import needs, before writing anything.

    The canonical payload is 24 bytes (three float64) per retained row plus 5% Parquet overhead
    and a fixed 1 MiB allowance; 10% reserve is added for the required free bytes. Text sources
    use a sampled row estimate and a pessimistic bound of one row per source byte. Check
    `ImportAdmission.admitted` before importing. Raises `SourceImportError` on I/O or parse
    errors.
    """
    try:
        return _estimate_import(request, available_bytes=available_bytes)
    except SourceImportError:
        raise
    except OSError as error:
        raise _error(
            DiagnosticCode.IMPORT_IO_ERROR,
            "Import estimation failed because of an I/O error.",
            operation=OperationId.CANONICAL_IMPORT.value,
        ) from error
    except (UnicodeError, pa.ArrowException) as error:
        raise _error(
            DiagnosticCode.IMPORT_PARSE_ERROR,
            "Import estimation failed because the source could not be parsed.",
            operation=OperationId.CANONICAL_IMPORT.value,
        ) from error


def _recognized_mapping(columns: tuple[str, ...]) -> ColumnMapping | None:
    """Return the mapping for an exactly recognized header, if any."""
    for names, units in _RECOGNIZED.items():
        if all(name in columns for name in names):
            return ColumnMapping(*names, *units)
    return None


def _inspect_source(
    source: Path, *, preview_rows: int = 100, text_options: TextOptions = _DEFAULT_TEXT_OPTIONS
) -> SourceInspection:
    """Read the schema and first rows of a source (see `inspect_source`)."""
    if preview_rows < 0:
        raise ValueError("preview_rows must not be negative")
    source = source.resolve(strict=True)
    kind = _kind(source)
    if kind is SourceKind.PARQUET:
        parquet = pq.ParquetFile(source)
        columns = tuple(parquet.schema_arrow.names)
        batches = parquet.iter_batches(batch_size=max(1, preview_rows))
        batch = next(
            batches,
            pa.record_batch(
                [pa.array([], type=field.type) for field in parquet.schema_arrow],
                schema=parquet.schema_arrow,
            ),
        )
        table = pa.Table.from_batches([batch]).slice(0, preview_rows)
        preview = tuple(
            tuple(None if value is None else str(value) for value in row.values())
            for row in table.to_pylist()
        )
        row_count: int | None = parquet.metadata.num_rows
    else:
        text_options.validate()
        read = pacsv.ReadOptions(
            block_size=64 * 1024,
            encoding=text_options.encoding,
            skip_rows=text_options.skip_rows,
        )
        parse = pacsv.ParseOptions(delimiter=text_options.delimiter)
        reader = pacsv.open_csv(source, read_options=read, parse_options=parse)
        columns = tuple(reader.schema.names)
        try:
            batch = reader.read_next_batch()
        except StopIteration:
            batch = pa.record_batch(
                [pa.array([], type=field.type) for field in reader.schema], schema=reader.schema
            )
        table = pa.Table.from_batches([batch]).slice(0, preview_rows)
        preview = tuple(
            tuple(None if value is None else str(value) for value in row.values())
            for row in table.to_pylist()
        )
        row_count = None
    mapping = _recognized_mapping(columns)
    return SourceInspection(source, kind, columns, row_count, preview, mapping, mapping is not None)


def inspect_source(
    source: Path, *, preview_rows: int = 100, text_options: TextOptions = _DEFAULT_TEXT_OPTIONS
) -> SourceInspection:
    """Preview a source without writing anything or changing it.

    Returns a `SourceInspection` with column names, up to `preview_rows` rows and, when the
    header is one of the recognized canonical forms (``time_s,current_a,voltage_v`` or
    ``time_s,current_mA,voltage_V``), a confirmed suggested mapping. Any other layout requires an
    explicit `ColumnMapping`. Raises `SourceImportError` for unsupported formats, I/O or parse
    errors.
    """
    try:
        return _inspect_source(source, preview_rows=preview_rows, text_options=text_options)
    except SourceImportError:
        raise
    except OSError as error:
        raise _error(
            DiagnosticCode.IMPORT_IO_ERROR,
            "Source inspection failed because of an I/O error.",
            operation=OperationId.INSPECT.value,
        ) from error
    except (UnicodeError, pa.ArrowException) as error:
        raise _error(
            DiagnosticCode.IMPORT_PARSE_ERROR,
            "Source inspection failed because the source could not be parsed.",
            operation=OperationId.INSPECT.value,
        ) from error


def _text_batches(
    source: Path, options: TextOptions, batch_size: int, columns: tuple[str, ...]
) -> Iterator[pa.RecordBatch]:
    """Stream the mapped text columns as string batches."""
    read = pacsv.ReadOptions(
        block_size=max(4096, batch_size * 64),
        encoding=options.encoding,
        skip_rows=options.skip_rows,
    )
    parse = pacsv.ParseOptions(delimiter=options.delimiter)
    header = pacsv.open_csv(source, read_options=read, parse_options=parse).schema.names
    missing = set(columns) - set(header)
    if missing:
        raise _error(
            DiagnosticCode.IMPORT_MAPPING_REQUIRED,
            "A mapped source column is missing.",
            column=sorted(missing)[0],
        )
    convert = pacsv.ConvertOptions(
        column_types={name: pa.string() for name in columns}, include_columns=list(columns)
    )
    yield from pacsv.open_csv(
        source, read_options=read, parse_options=parse, convert_options=convert
    )


def _source_batches(
    source: Path,
    kind: SourceKind,
    options: TextOptions,
    batch_size: int,
    columns: tuple[str, ...],
) -> Iterator[pa.RecordBatch]:
    """Stream the mapped columns from a Parquet or text source in bounded batches."""
    if kind is SourceKind.PARQUET:
        parquet = pq.ParquetFile(source)
        missing = set(columns) - set(parquet.schema_arrow.names)
        if missing:
            raise _error(
                DiagnosticCode.IMPORT_MAPPING_REQUIRED,
                "A mapped source column is missing.",
                column=sorted(missing)[0],
            )
        yield from parquet.iter_batches(batch_size=batch_size, columns=list(columns))
    else:
        for batch in _text_batches(source, options, batch_size, columns):
            for offset in range(0, batch.num_rows, batch_size):
                yield batch.slice(offset, batch_size)


def _import_source_batches(
    source: Path,
    kind: SourceKind,
    options: TextOptions,
    batch_size: int,
    columns: tuple[str, ...],
) -> Iterator[pa.RecordBatch]:
    """Wrap source streaming so read/parse errors become import errors."""
    try:
        yield from _source_batches(source, kind, options, batch_size, columns)
    except SourceImportError:
        raise
    except OSError as error:
        raise _error(
            DiagnosticCode.IMPORT_IO_ERROR,
            "Source reading failed because of an I/O error.",
            operation=OperationId.CANONICAL_IMPORT.value,
        ) from error
    except (UnicodeError, pa.ArrowException) as error:
        raise _error(
            DiagnosticCode.IMPORT_PARSE_ERROR,
            "Source reading failed because the source could not be parsed.",
            operation=OperationId.CANONICAL_IMPORT.value,
        ) from error


def _numeric(values: pa.Array, decimal_separator: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse values as finite decimal numbers; return values and a validity mask."""
    strings = pc.fill_null(values.cast(pa.string()), "")
    if decimal_separator == ",":
        strings = pc.replace_substring(strings, ",", ".")
    valid_arrow = pc.match_substring_regex(strings, _NUMBER_PATTERN)
    safe = pc.if_else(valid_arrow, strings, "nan")
    result = np.asarray(pc.cast(safe, pa.float64()))
    valid = np.asarray(valid_arrow)
    return result, valid & np.isfinite(result)


def _duration(values: pa.Array, decimal_separator: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse ``H+:MM:SS[.fraction]`` text to seconds; return values and validity."""
    strings = pc.fill_null(values.cast(pa.string()), "")
    parts = pc.extract_regex(strings, _DURATION_PATTERN)
    hours = pc.struct_field(parts, "h")
    minutes = pc.struct_field(parts, "m")
    seconds = pc.struct_field(parts, "s")
    separators = pc.struct_field(parts, "sep")
    fractions = pc.struct_field(parts, "f")
    valid_arrow = pc.fill_null(
        pc.and_(
            pc.is_valid(hours),
            pc.or_(pc.equal(separators, ""), pc.equal(separators, decimal_separator)),
        ),
        False,
    )
    safe_hours = pc.if_else(valid_arrow, hours, "0")
    safe_minutes = pc.if_else(valid_arrow, minutes, "0")
    safe_seconds = pc.if_else(valid_arrow, seconds, "0")
    safe_fraction = pc.if_else(pc.equal(fractions, ""), "0", fractions)
    fraction_text = pc.binary_join_element_wise("0.", safe_fraction, "")
    output = (
        3600.0 * np.asarray(pc.cast(safe_hours, pa.float64()))
        + 60.0 * np.asarray(pc.cast(safe_minutes, pa.float64()))
        + np.asarray(pc.cast(safe_seconds, pa.float64()))
        + np.asarray(pc.cast(fraction_text, pa.float64()))
    )
    valid = np.asarray(valid_arrow) & np.isfinite(output)
    output[~valid] = np.nan
    return output, valid


def _column(batch: pa.RecordBatch, name: str) -> pa.Array:
    """Return a mapped column of a batch or raise `IMPORT_MAPPING_REQUIRED`."""
    index = batch.schema.get_field_index(name)
    if index < 0:
        raise _error(
            DiagnosticCode.IMPORT_MAPPING_REQUIRED,
            "A mapped source column is missing.",
            column=name,
        )
    return batch.column(index)


def _convert_units(values: np.ndarray, role: str, unit: str) -> np.ndarray:
    """Convert values of `role` from `unit` to the canonical unit."""
    divisor = _UNIT_DIVISORS.get((role, unit))
    with np.errstate(over="ignore", invalid="ignore"):
        if divisor is not None:
            return values / divisor
        return values * _UNIT_FACTORS[role][unit]


def _close_parquet_writer(writer: pq.ParquetWriter) -> None:
    """Close a Parquet writer (separate function so tests can inject failures)."""
    writer.close()


def _cleanup_failed_import(
    writer: pq.ParquetWriter | None,
    staging: StagingGeneration,
    primary: BaseException,
) -> None:
    """Release resources after a failure without masking the original error."""
    cleanup_failures: list[tuple[str, Exception]] = []
    if writer is not None:
        try:
            _close_parquet_writer(writer)
        except Exception as error:
            cleanup_failures.append(("Parquet writer", error))
    try:
        staging.close()
    except Exception as error:
        cleanup_failures.append(("staging lease", error))
    for resource, cleanup_error in cleanup_failures:
        primary.add_note(
            "Additional "
            f"{resource} cleanup failure: {type(cleanup_error).__name__}: {cleanup_error}"
        )


def _canonical_bytes(time: np.ndarray, current: np.ndarray, voltage: np.ndarray) -> bytes:
    """Return rows as little-endian float64 triples for content hashing."""
    rows = np.empty((len(time), 3), dtype="<f8")
    rows[:, 0], rows[:, 1], rows[:, 2] = time, current, voltage
    return rows.tobytes(order="C")


def _report_bytes(report: ImportReport, mapping: ColumnMapping) -> bytes:
    """Serialize the import report and confirmed mapping as JSON bytes."""
    data = {
        "schema_version": 1,
        "source_rows": report.source_rows,
        "retained_rows": report.retained_rows,
        "removed_invalid_measurement": report.removed_invalid_measurement,
        "removed_duplicate_time": report.removed_duplicate_time,
        "content_hash": report.content_hash,
        "mapping": {
            "time_column": mapping.time_column,
            "current_column": mapping.current_column,
            "voltage_column": mapping.voltage_column,
            "time_unit": mapping.time_unit,
            "current_unit": mapping.current_unit,
            "voltage_unit": mapping.voltage_unit,
            "time_format": mapping.time_format.value,
            "positive_current_means": "charge",
        },
        "diagnostics": [item.to_dict() for item in report.diagnostics],
    }
    return (json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n").encode()


def import_source(
    request: ImportRequest,
    *,
    progress: ProgressSink | None = None,
    cancel: CancelToken | None = None,
    fault_hook: Callable[[str], None] | None = None,
    available_bytes: Callable[[Path], int] = _available_disk_bytes,
) -> ImportResult:
    """Import a source file into an immutable canonical generation.

    Args:
        request: `ImportRequest` with source, destination identifiers and mapping.
        progress: Optional progress sink (unit ``source_rows``).
        cancel: Optional cancellation token.
        fault_hook: Testing hook called at named checkpoints; leave it None.
        available_bytes: Free-space probe, replaceable for testing.

    Returns:
        `ImportResult`. The canonical table is ``result.ref.path / "points" /
        "canonical.parquet"`` and ``result.report.content_hash`` identifies its content.

    Raises:
        SourceImportError: invalid mapping, time errors, too few rows, parse/I-O failures or
            insufficient disk space.
        StorageError: result-store conflicts or failures.
        OperationCancelled: the cancel token was set.

    Processing is streamed in bounded batches, so files larger than memory can be imported. The
    import is refused if the source changes (size or modification time) while it is being read.
    """
    request.validate()
    try:
        source = request.source.resolve(strict=True)
        before = (source.stat().st_size, source.stat().st_mtime_ns)
        kind = _kind(source)
    except SourceImportError:
        raise
    except OSError as error:
        raise _error(
            DiagnosticCode.IMPORT_IO_ERROR,
            "Source access failed.",
            operation=OperationId.CANONICAL_IMPORT.value,
        ) from error
    mapping = request.mapping
    if mapping is None:
        mapping = inspect_source(
            source, preview_rows=0, text_options=request.text_options
        ).suggested_mapping
    if mapping is None:
        raise _error(
            DiagnosticCode.IMPORT_MAPPING_REQUIRED, "Explicit column and unit mapping is required."
        )
    mapping.validate()
    admission = estimate_import(request, available_bytes=available_bytes)
    if not admission.admitted:
        raise _error(
            DiagnosticCode.INSUFFICIENT_DISK_SPACE,
            "Insufficient free space for conservative import admission.",
            operation=OperationId.CANONICAL_IMPORT.value,
        )
    store = ResultStore(request.caller_root)
    staging = store.begin_job(request.sample_id, request.job_id, request.generation_id)
    destination = staging.path / "points" / "canonical.parquet"
    destination.parent.mkdir(parents=True)
    writer: pq.ParquetWriter | None = None
    digest = hashlib.sha256(b"gregorithm-canonical-rows-v1\0")
    source_rows = retained_rows = invalid_count = duplicate_count = 0
    previous_time: float | None = None
    pending_tables: list[pa.Table] = []
    pending_rows = 0
    try:
        writer = pq.ParquetWriter(destination, CANONICAL_SCHEMA, compression="zstd")
        columns = (mapping.time_column, mapping.current_column, mapping.voltage_column)
        for batch in _import_source_batches(
            source, kind, request.text_options, request.batch_size, columns
        ):
            check_cancelled(cancel, OperationId.CANONICAL_IMPORT.value)
            count = batch.num_rows
            row_numbers = np.arange(source_rows, source_rows + count, dtype=np.int64)
            source_rows += count
            current, current_valid = _numeric(
                _column(batch, mapping.current_column), request.text_options.decimal_separator
            )
            voltage, voltage_valid = _numeric(
                _column(batch, mapping.voltage_column), request.text_options.decimal_separator
            )
            measurement_valid = current_valid & voltage_valid
            invalid_count += int(count - np.count_nonzero(measurement_valid))
            time_values = _column(batch, mapping.time_column).filter(pa.array(measurement_valid))
            if mapping.time_format is TimeFormat.DURATION:
                time, time_valid = _duration(time_values, request.text_options.decimal_separator)
            else:
                time, time_valid = _numeric(time_values, request.text_options.decimal_separator)
                time = _convert_units(time, "time", mapping.time_unit)
                time_valid &= np.isfinite(time)
            survivor_rows = row_numbers[measurement_valid]
            if not np.all(time_valid):
                index = int(np.flatnonzero(~time_valid)[0])
                raise _error(
                    DiagnosticCode.IMPORT_INVALID_TIME,
                    "A surviving row has an invalid elapsed time.",
                    source_row=int(survivor_rows[index]),
                )
            current = _convert_units(current[measurement_valid], "current", mapping.current_unit)
            voltage = _convert_units(voltage[measurement_valid], "voltage", mapping.voltage_unit)
            if len(time):
                comparisons = np.diff(time)
                if previous_time is not None and time[0] < previous_time:
                    raise _error(
                        DiagnosticCode.IMPORT_TIME_DECREASE,
                        "Elapsed time decreased.",
                        source_row=int(survivor_rows[0]),
                        time_s=float(time[0]),
                    )
                decreasing = np.flatnonzero(comparisons < 0)
                if len(decreasing):
                    index = int(decreasing[0] + 1)
                    raise _error(
                        DiagnosticCode.IMPORT_TIME_DECREASE,
                        "Elapsed time decreased.",
                        source_row=int(survivor_rows[index]),
                        time_s=float(time[index]),
                    )
                keep = np.ones(len(time), dtype=bool)
                if previous_time is not None and time[0] == previous_time:
                    keep[0] = False
                if len(time) > 1:
                    keep[1:] = comparisons != 0
                duplicate_count += int(len(time) - np.count_nonzero(keep))
                time, current, voltage = time[keep], current[keep], voltage[keep]
                if len(time):
                    previous_time = float(time[-1])
                    retained_rows += len(time)
                    digest.update(_canonical_bytes(time, current, voltage))
                    table = pa.Table.from_arrays(
                        [pa.array(time), pa.array(current), pa.array(voltage)],
                        schema=CANONICAL_SCHEMA,
                    )
                    pending_tables.append(table)
                    pending_rows += len(time)
                    if pending_rows >= PARQUET_ROW_GROUP_SIZE:
                        combined = pa.concat_tables(pending_tables)
                        offset = 0
                        while combined.num_rows - offset >= PARQUET_ROW_GROUP_SIZE:
                            if fault_hook is not None:
                                fault_hook("before_parquet_write")
                            writer.write_table(
                                combined.slice(offset, PARQUET_ROW_GROUP_SIZE),
                                row_group_size=PARQUET_ROW_GROUP_SIZE,
                            )
                            offset += PARQUET_ROW_GROUP_SIZE
                        pending_tables = (
                            [combined.slice(offset)] if offset < combined.num_rows else []
                        )
                        pending_rows = combined.num_rows - offset
            report_progress(
                progress,
                ProgressEvent(
                    OperationId.CANONICAL_IMPORT, "read", source_rows, None, "source_rows"
                ),
            )
            if fault_hook is not None:
                fault_hook("after_batch")
        if pending_rows:
            if fault_hook is not None:
                fault_hook("before_parquet_write")
            writer.write_table(
                pa.concat_tables(pending_tables), row_group_size=PARQUET_ROW_GROUP_SIZE
            )
        _close_parquet_writer(writer)
        writer = None
        if fault_hook is not None:
            fault_hook("after_parquet_close")
        try:
            with destination.open("rb") as parquet_stream:
                parquet = pq.ParquetFile(parquet_stream)
                if (
                    parquet.schema_arrow != CANONICAL_SCHEMA
                    or parquet.metadata.num_rows != retained_rows
                ):
                    raise ValueError("Canonical Parquet schema or row count is invalid")
        except (OSError, ValueError, pa.ArrowException) as error:
            raise _error(
                DiagnosticCode.STORAGE_IO_ERROR,
                "Canonical Parquet validation failed after writing.",
                operation=OperationId.CANONICAL_IMPORT.value,
            ) from error
        if retained_rows < 2:
            raise _error(
                DiagnosticCode.IMPORT_INSUFFICIENT_ROWS,
                "Fewer than two rows remain after cleaning.",
            )
        diagnostics: list[Diagnostic] = []
        if invalid_count:
            diagnostics.append(
                Diagnostic(
                    DiagnosticCode.IMPORT_REMOVED_INVALID_MEASUREMENT,
                    DiagnosticKind.WARNING,
                    "WARNING: INVALID CURRENT OR VOLTAGE VALUES DETECTED. "
                    "AFFECTED ROWS HAVE BEEN REMOVED FROM THE DATASET. "
                    "RESULTS MAY BE AFFECTED.",
                    operation=OperationId.CANONICAL_IMPORT.value,
                    count=invalid_count,
                )
            )
        if duplicate_count:
            diagnostics.append(
                Diagnostic(
                    DiagnosticCode.IMPORT_REMOVED_DUPLICATE_TIME,
                    DiagnosticKind.WARNING,
                    "Duplicate timestamps detected. The first occurrence of each "
                    "timestamp was retained, and all subsequent occurrences were discarded.",
                    operation=OperationId.CANONICAL_IMPORT.value,
                    count=duplicate_count,
                )
            )
        report = ImportReport(
            source_rows,
            retained_rows,
            invalid_count,
            duplicate_count,
            tuple(diagnostics),
            digest.hexdigest(),
        )
        if fault_hook is not None:
            fault_hook("before_report_write")
        staging.write_bytes("import_report.json", _report_bytes(report, mapping))
        parameters: Mapping[str, Any] = {
            "mapping": {
                "time_column": mapping.time_column,
                "current_column": mapping.current_column,
                "voltage_column": mapping.voltage_column,
                "time_unit": mapping.time_unit,
                "current_unit": mapping.current_unit,
                "voltage_unit": mapping.voltage_unit,
                "time_format": mapping.time_format.value,
            }
        }
        manifest = staging.build_manifest(
            operation=OperationId.CANONICAL_IMPORT,
            parameter_hash=hash_parameters(parameters),
            dependency_hash=hash_dependencies(()),
            content_hash=report.content_hash,
            diagnostics=diagnostics,
            row_counts={"source": source_rows, "canonical": retained_rows},
            logical_schemas={
                "points/canonical.parquet": "gregorithm.canonical_measurements.v1",
                "import_report.json": "gregorithm.import_report.v1",
            },
        )
        try:
            after = (source.stat().st_size, source.stat().st_mtime_ns)
        except OSError as error:
            raise _error(
                DiagnosticCode.IMPORT_IO_ERROR,
                "Source metadata could not be read after import.",
                operation=OperationId.CANONICAL_IMPORT.value,
            ) from error
        if after != before:
            raise _error(DiagnosticCode.STORAGE_IO_ERROR, "The source changed during import.")
        ref = store.publish(staging, manifest, cancel=cancel, fault_hook=fault_hook)
        return ImportResult(ref, report)
    except (SourceImportError, StorageError) as error:
        _cleanup_failed_import(writer, staging, error)
        raise
    except (OSError, UnicodeError, pa.ArrowException) as error:
        code = (
            DiagnosticCode.STORAGE_IO_ERROR
            if isinstance(error, OSError)
            else DiagnosticCode.IMPORT_PARSE_ERROR
        )
        normalized = _error(
            code,
            "Import failed because of an expected source or filesystem error.",
            operation=OperationId.CANONICAL_IMPORT.value,
        )
        _cleanup_failed_import(writer, staging, normalized)
        raise normalized from error
    except Exception as error:
        _cleanup_failed_import(writer, staging, error)
        raise
