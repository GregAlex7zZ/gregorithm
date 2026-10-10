"""Bounded readers, immutable views and atomic exports."""

from __future__ import annotations

import csv
import hashlib
import os
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import monotonic, sleep

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import canonical, digest
from .diagnostics import (
    AmbiguousTimeSelectionError,
    BudgetExceeded,
    MaterializationLimitError,
    SchemaError,
)

REQUIRED = ("time_s", "current_mA", "voltage_V")
KEYS = ("trace_id", "result_id", "sample_id", "source_revision")
RESERVED = set(KEYS) | {
    "time_section_id",
    "step_id",
    "half_cycle_id",
    "cycle_id",
    "direction",
    "mode",
    "capacity_mAh",
}
RESERVED |= {
    "main_sign",
    "observed_sign",
    "excluded",
    "confirmed",
    "event_id",
    "event_ids",
    "mode_reason",
    "step_role",
    "q_interval_valid",
    "e_interval_valid",
    "coverage_partial",
    "c_rate_h_inv",
    "global_net_mAh",
    "global_throughput_mAh",
    "global_net_Wh",
    "global_throughput_Wh",
}


def replace_file(staged, destination, *, timeout_s=2.0):
    """Atomically replace a file, retrying only transient Windows lock failures.

    A bounded exponential delay handles WinError 5/32/33 from temporary sharing
    restrictions such as scanner activity. Other failures are raised immediately;
    persistent Windows access failures are raised after timeout_s. This never
    falls back to a non-atomic copy or deletes a destination to bypass access.
    Both paths must already be within the caller's authorized output directory.
    """
    if timeout_s < 0:
        raise ValueError("Nonnegative replacement timeout required")
    deadline = monotonic() + timeout_s
    delay = 0.02
    while True:
        try:
            os.replace(staged, destination)
            return
        except PermissionError as error:
            remaining = deadline - monotonic()
            if (
                os.name != "nt"
                or getattr(error, "winerror", None) not in {5, 32, 33}
                or remaining <= 0
            ):
                raise
            sleep(min(delay, remaining))
            delay = min(0.25, delay * 2)


RESERVED |= {
    prefix + quantity
    for prefix in ("d_", "global_", "cycling_", "excluded_d_", "cycle_", "step_")
    for quantity in ("charge_mAh", "discharge_mAh", "charge_Wh", "discharge_Wh")
}


@dataclass(frozen=True)
class ReadOptions:
    """Source aliases and bounded decoder settings. Required standardized names are time_s/current_mA/voltage_V; column_mapping maps original names to them. Text delimiter/encoding/decimal are explicit; extra text columns retain lexemes."""

    column_mapping: dict = field(default_factory=dict)
    delimiter: str = ","
    encoding: str = "utf-8-sig"
    decimal: str = "."
    batch_rows: int = 262144
    max_batch_bytes: int = 64 * 1024**2


def fingerprint(path, strong=False):
    """Return size/mtime metadata, optionally adding a streamed SHA-256 content digest. A strong digest is required for stable relocation; the quick signature alone cannot prove content identity.

    Parameters
    ----------
    path
        Input or destination path as specified; scientific sources must not be overwritten.
    strong
        If True, compute a streamed SHA-256 source-content digest."""
    p = Path(path)
    s = p.stat()
    d = {"size": s.st_size, "mtime_ns": s.st_mtime_ns}
    if strong:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda: f.read(1024**2), b""):
                h.update(b)
        d["sha256"] = h.hexdigest()
    return d


def source_batches(path, options=ReadOptions(), columns=None, row_groups=None):
    """Yield bounded Arrow batches from standardized source columns. Parquet decoders are recreated per row group to release compressed buffers; CSV/TXT is sequential with numeric NaN coercion and extra lexemes preserved as strings. Optional columns is an explicit projection.

    Parameters
    ----------
    path
        Input or destination path as specified; scientific sources must not be overwritten.
    options
        Bounded operation-specific value described by the function contract and API reference.
    columns
        Explicit column projection or normalization input names, according to this operation.
    row_groups
        Bounded operation-specific value described by the function contract and API reference."""
    p = Path(path)
    mapping = options.column_mapping
    if p.suffix.lower() == ".parquet":
        f = pq.ParquetFile(p)
        names = f.schema_arrow.names
        target = {mapping.get(n, n): n for n in names}
        wanted = None if columns is None else [target[n] for n in columns if n in target]
        bytes_per_row = max(
            64, sum(16 if pa.types.is_string(x.type) else 8 for x in f.schema_arrow)
        )
        rows = max(1, min(options.batch_rows, options.max_batch_bytes // (bytes_per_row * 4)))
        try:
            # A single iterator spanning all groups can retain compressed buffers for
            # the entire file. Recreate the decoder per group, while keeping state in
            # the analysis engine, not in the reader. This bounds file-size growth.
            groups = range(f.num_row_groups) if row_groups is None else row_groups
            for group in groups:
                for b in f.iter_batches(
                    batch_size=rows, columns=wanted, row_groups=[group], use_threads=False
                ):
                    if b.nbytes > options.max_batch_bytes:
                        raise BudgetExceeded(
                            "Decoded Arrow batch exceeds reader budget; lower batch_rows"
                        )
                    yield b.rename_columns([mapping.get(n, n) for n in b.schema.names])
                pa.default_memory_pool().release_unused()
        finally:
            f.close()
            pa.default_memory_pool().release_unused()
    elif p.suffix.lower() in {".csv", ".txt"}:
        # A bounded standard-library fallback handles invalid numeric tokens and decimal comma.
        # Other columns are strings, preserving lexemes rather than guessing types from a prefix.
        with p.open(encoding=options.encoding, newline="") as f:
            reader = csv.DictReader(f, delimiter=options.delimiter)
            if reader.fieldnames is None:
                return
            targets = [mapping.get(n, n) for n in reader.fieldnames]
            if len(set(targets)) != len(targets):
                raise SchemaError("Column mapping collision")
            use = [n for n in targets if columns is None or n in columns]
            data = {n: [] for n in use}
            size = 0
            for row in reader:
                for original, n in zip(reader.fieldnames, targets):
                    if n not in data:
                        continue
                    raw = row[original]
                    if n in REQUIRED:
                        try:
                            value = float(raw.replace(options.decimal, "."))
                        except (ValueError, AttributeError):
                            value = float("nan")
                    else:
                        value = raw
                    data[n].append(value)
                    size += len(raw or "") + 16
                if (
                    size >= options.max_batch_bytes // 4
                    or len(next(iter(data.values()), [])) >= options.batch_rows
                ):
                    yield pa.RecordBatch.from_pydict(data)
                    data = {n: [] for n in use}
                    size = 0
            if data and len(next(iter(data.values()))):
                yield pa.RecordBatch.from_pydict(data)
    else:
        raise SchemaError("Supported sources: parquet, csv, txt")


def atomic_parquet(batches, path, manifest=None, compression="zstd", compression_level=1):
    """Write Arrow batches incrementally to a staged Parquet file and publish the file plus JSON provenance sidecar after success. Never use the source path as destination. Exceptions remove incomplete staged output.

    Parameters
    ----------
    batches
        Bounded operation-specific value described by the function contract and API reference.
    path
        Input or destination path as specified; scientific sources must not be overwritten.
    manifest
        Captured provenance dictionary or saved trace-manifest path, as specified by the operation.
    compression
        Bounded operation-specific value described by the function contract and API reference.
    compression_level
        Bounded operation-specific value described by the function contract and API reference."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    writer = None
    try:
        for b in batches:
            if writer is None:
                writer = pq.ParquetWriter(
                    tmp,
                    b.schema,
                    compression=compression,
                    compression_level=compression_level if compression == "zstd" else None,
                )
            writer.write_batch(b)
        if writer is None:
            pq.write_table(pa.table({}), tmp)
        else:
            writer.close()
            writer = None
        replace_file(tmp, path)
        if manifest is not None:
            mp = path.with_suffix(path.suffix + ".manifest.json")
            mt = mp.with_suffix(mp.suffix + ".tmp")
            mt.write_text(canonical(manifest), encoding="utf8")
            replace_file(mt, mp)
    finally:
        if writer is not None:
            writer.close()
        close_source = getattr(batches, "close", None)
        if close_source is not None:
            close_source()
        if tmp.exists():
            tmp.unlink()
    return path


def selection_mask(b, clauses):
    """Evaluate equality, membership, interval, and logical-OR clauses as a Boolean row mask while retaining original row order.

    Parameters
    ----------
    b
        Bounded operation-specific value described by the function contract and API reference.
    clauses
        Bounded operation-specific value described by the function contract and API reference."""
    mask = np.ones(b.num_rows, dtype=bool)
    for clause in clauses:
        cm = np.ones(b.num_rows, dtype=bool)
        for key, value in clause.items():
            if key == "__or__":
                alternatives = np.zeros(b.num_rows, dtype=bool)
                for branch in value:
                    alternatives |= selection_mask(b, branch)
                cm &= alternatives
                continue
            if key == "columns":
                continue
            if key == "event_id" and "event_ids" in b.schema.names:
                wanted = set(value if isinstance(value, (list, tuple)) else [value])
                cm &= np.array(
                    [
                        bool(wanted.intersection(ids or []))
                        for ids in b.column("event_ids").to_pylist()
                    ]
                )
                continue
            if key not in b.schema.names:
                raise SchemaError(f"Selection requires missing column {key}")
            arr = b.column(b.schema.get_field_index(key)).to_numpy(zero_copy_only=False)
            if isinstance(value, dict):
                if value.get("min") is not None:
                    cm &= arr >= value["min"]
                if value.get("max") is not None:
                    cm &= arr <= value["max"]
            else:
                cm &= np.isin(arr, value if isinstance(value, (list, tuple)) else [value])
        mask &= cm
    return mask


def filter_batch(b, clauses):
    """Apply a declarative row mask to one Arrow batch without changing source sample identities.

    Parameters
    ----------
    b
        Bounded operation-specific value described by the function contract and API reference.
    clauses
        Bounded operation-specific value described by the function contract and API reference."""
    return b.filter(pa.array(selection_mask(b, clauses)))


def predicate_columns(clauses):
    """Find column dependencies of nested declarative selection clauses; used to choose the minimum required annotation work.

    Parameters
    ----------
    clauses
        Bounded operation-specific value described by the function contract and API reference."""
    columns = set()
    for clause in clauses:
        for key, value in clause.items():
            if key == "__or__":
                for branch in value:
                    columns.update(predicate_columns(branch))
            else:
                columns.add(key)
    return columns


def expand_clauses(clauses):
    """Expand nested OR alternatives into conjunctions for interval-domain clipping; callers merge overlapping domains before integration.

    Parameters
    ----------
    clauses
        Bounded operation-specific value described by the function contract and API reference."""
    for j, clause in enumerate(clauses):
        if "__or__" in clause:
            rest = {k: v for k, v in clause.items() if k != "__or__"}
            for branch in clause["__or__"]:
                yield from expand_clauses(
                    tuple(clauses[:j]) + (rest,) + tuple(branch) + tuple(clauses[j + 1 :])
                )
            return
    yield tuple(clauses)


def resolve_source_clauses(clauses, trace):
    """Resolve trace/file predicates against one source before applying row predicates; source names are not fabricated signal columns.

    Parameters
    ----------
    clauses
        Bounded operation-specific value described by the function contract and API reference.
    trace
        BatteryTrace source handle; raw signals are never rewritten."""
    resolved = []
    for clause in clauses:
        item = {}
        mismatch = False
        for key, value in clause.items():
            if key == "__or__":
                item[key] = [resolve_source_clauses(branch, trace) for branch in value]
            elif key in {"file", "trace_id"}:
                choices = value if isinstance(value, (list, tuple)) else [value]
                actual = trace.trace_id if key == "trace_id" else str(trace.path)
                mismatch |= actual not in choices and (
                    key != "file" or trace.path.name not in choices
                )
            else:
                item[key] = value
        resolved.append({"sample_id": []} if mismatch else item)
    return tuple(resolved)


def known_empty(clauses):
    """Recognize a declarative empty selection without scanning source rows.

    Parameters
    ----------
    clauses
        Bounded operation-specific value described by the function contract and API reference."""
    return any(
        c.get("sample_id") == []
        or ("__or__" in c and all(known_empty(branch) for branch in c["__or__"]))
        for c in clauses
    )


@dataclass(frozen=True)
class TableView:
    """Immutable lazy Arrow table plan. Use select()/head() for explicit previews, iter_batches() for large data, to_pandas() with limits, and write_parquet() for streaming exports."""

    factory: object
    manifest: dict = field(default_factory=dict)
    clauses: tuple = ()
    columns: tuple | None = None
    row_limit: int | None = None
    memory_bytes: int = 2 * 1024**3
    parquet_path: str | None = None
    declared_schema: pa.Schema | None = None

    @property
    def schema(self):
        """Return the known output schema without scanning, respecting projection.

        None means the factory did not declare its schema. Compact identity
        fields remain in explicit projections just as they do in actual batches.
        Empty materializations retain this schema when it is known.
        """
        if self.declared_schema is None or self.columns is None:
            return self.declared_schema
        names = list(
            dict.fromkeys(
                list(self.columns) + [key for key in KEYS if key in self.declared_schema.names]
            )
        )
        return pa.schema(
            [self.declared_schema.field(name) for name in names],
            metadata=self.declared_schema.metadata,
        )

    def select(self, columns=None, **predicates):
        """Return an immutable filtered view/selection; repeated predicates intersect. Equality, lists, and min/max intervals are supported. This never truncates scientific computation implicitly.

        Usage: call on the TableView object; returned selections/revisions are independent of earlier views."""
        selected = replace(
            self,
            clauses=self.clauses + ((predicates,) if predicates else ()),
            columns=self.columns if columns is None else tuple(columns),
        )
        return selected._with_selection_manifest()

    def head(self, n=5):
        """Return an explicitly row-limited preview. Negative limits are rejected; this is not a scientific downsampling operation.

        Usage: call on the TableView object; returned selections/revisions are independent of earlier views."""
        if n < 0:
            raise ValueError("head >=0")
        return replace(
            self, row_limit=n if self.row_limit is None else min(n, self.row_limit)
        )._with_selection_manifest()

    def _with_selection_manifest(self):
        """Capture the new selection digest in an independent manifest so earlier views retain their provenance."""
        selection = {"clauses": self.clauses, "columns": self.columns, "head": self.row_limit}
        return replace(
            self,
            manifest={
                **self.manifest,
                "table_selection": selection,
                "selection_id": digest(
                    {"parent_selection_id": self.manifest.get("selection_id"), **selection}
                ),
            },
        )

    def iter_batches(self, batch_rows=None):
        """Yield filtered/projected Arrow batches lazily, retaining compact identity columns and original order."""
        unscoped = [
            leaf
            for leaf in expand_clauses(self.clauses)
            if any("time_s" in c for c in leaf) and not any("time_section_id" in c for c in leaf)
        ]
        for leaf in unscoped:
            sections = {}
            for block in self._source():
                if "time_section_id" not in block.schema.names:
                    break
                selected = filter_batch(block, leaf)
                for row in selected.select(
                    [
                        name
                        for name in ("trace_id", "time_section_id")
                        if name in selected.schema.names
                    ]
                ).to_pylist():
                    tid = row.get("trace_id", "table")
                    section = row["time_section_id"]
                    if tid in sections and sections[tid] != section:
                        raise AmbiguousTimeSelectionError(
                            "Select time_section_id explicitly for repeated acquisition time domains"
                        )
                    sections[tid] = section
        left = self.row_limit
        for b in self._source():
            if left == 0:
                break
            b = filter_batch(b, self.clauses)
            if self.columns is not None:
                names = list(
                    dict.fromkeys(list(self.columns) + [k for k in KEYS if k in b.schema.names])
                )
                b = b.select(names)
            if left is not None:
                b = b.slice(0, left)
                left -= b.num_rows
            if b.num_rows:
                n = batch_rows or b.num_rows
                for start in range(0, b.num_rows, n):
                    yield b.slice(start, n)

    def _source(self):
        """Yield source batches while pruning only row groups proven irrelevant by Parquet statistics."""
        if self.parquet_path is None:
            yield from self.factory()
            return
        f = pq.ParquetFile(self.parquet_path)
        available = f.schema_arrow.names
        groups = []
        try:
            for g in range(f.num_row_groups):
                rg = f.metadata.row_group(g)
                keep = True
                for clause in self.clauses:
                    for k, v in clause.items():
                        if k not in f.schema_arrow.names:
                            continue
                        st = rg.column(f.schema_arrow.get_field_index(k)).statistics
                        if not st or not st.has_min_max:
                            continue
                        if isinstance(v, dict):
                            keep &= (v.get("min") is None or st.max >= v["min"]) and (
                                v.get("max") is None or st.min <= v["max"]
                            )
                        else:
                            keep &= any(
                                st.min <= x <= st.max
                                for x in (v if isinstance(v, (list, tuple)) else [v])
                            )
                if keep:
                    groups.append(g)
        finally:
            f.close()
        yield from source_batches(
            self.parquet_path,
            ReadOptions(max_batch_bytes=min(64 * 1024**2, self.memory_bytes // 8)),
            columns=None
            if self.columns is None
            else list(
                dict.fromkeys(
                    list(self.columns)
                    + [name for name in KEYS if name in available]
                    + list(predicate_columns(self.clauses))
                )
            ),
            row_groups=groups,
        )

    def to_pandas(self, max_rows=1_000_000, max_bytes=512 * 1024**2):
        """Materialize only this explicitly selected view, checking row and conservative copy-byte limits before concatenation. Exceeding a limit raises MaterializationLimitError without truncation.

        Usage: call on the TableView object; returned selections/revisions are independent of earlier views."""
        parts = []
        rows = size = 0
        cap = min(max_bytes, self.memory_bytes // 2)
        for b in self.iter_batches():
            rows += b.num_rows
            size += b.nbytes * 4
            if rows > max_rows or size > cap:
                raise MaterializationLimitError(
                    f"Requested view exceeds max_rows={max_rows} or conservative bytes={cap}; use iter_batches/write_parquet"
                )
            parts.append(b)
        return (
            pa.Table.from_batches(parts)
            if parts
            else pa.Table.from_batches([], schema=self.schema)
            if self.schema is not None
            else pa.table({})
        ).to_pandas()

    def write_parquet(self, path):
        """Export this view incrementally with a provenance sidecar; refuse to overwrite its original source.

        Usage: call on the TableView object; returned selections/revisions are independent of earlier views."""
        destination = Path(path).resolve()
        if (
            self.manifest.get("source_path")
            and destination == Path(self.manifest["source_path"]).resolve()
        ):
            raise ValueError("Cannot overwrite source")
        m = dict(self.manifest)
        m["export_selection"] = {
            "clauses": self.clauses,
            "columns": self.columns,
            "head": self.row_limit,
        }
        return atomic_parquet(self.iter_batches(), destination, m)

    def relative_time(self, origin=None):
        """Add relative_time_s using the supplied origin or the first selected sample. Preserve original time_s and all numerical accumulation origins."""
        view = self

        def factory():
            """Produce this captured lazy view in bounded batches; retain its source and scientific revision."""
            base = origin
            for b in view.iter_batches():
                t = b.column(b.schema.get_field_index("time_s")).to_numpy()
                if base is None and len(t):
                    base = float(t[0])
                if "relative_time_s" in b.schema.names:
                    raise SchemaError("relative_time_s collision")
                yield b.append_column("relative_time_s", pa.array(t - base))

        return TableView(factory, self.manifest, memory_bytes=self.memory_bytes)


def parquet_view(path, manifest=None, memory_bytes=2 * 1024**3):
    """Wrap an existing Parquet file as a lazy TableView with its declared schema and supplied manifest. No pandas table is materialized.

    Parameters
    ----------
    path
        Input or destination path as specified; scientific sources must not be overwritten.
    manifest
        Captured provenance dictionary or saved trace-manifest path, as specified by the operation.
    memory_bytes
        Bounded operation-specific value described by the function contract and API reference."""
    return TableView(
        lambda: source_batches(path),
        manifest or {},
        memory_bytes=memory_bytes,
        parquet_path=str(path),
        declared_schema=pq.read_schema(path),
    )
