"""Source handles, immutable selections, and per-trace results."""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import AnalysisConfig, canonical, digest
from .diagnostics import RevisionMismatchError, SchemaError, SourceChangedError
from .io import (
    REQUIRED,
    RESERVED,
    ReadOptions,
    TableView,
    fingerprint,
    predicate_columns,
    source_batches,
)
from .provenance import runtime_provenance


@dataclass
class BatteryTrace:
    """Lightweight source handle with trace/source identity, schema, reader options, metadata, and cached disk indices. Construct with open_trace(); use select() or analyze(), not a full DataFrame property."""

    path: Path
    trace_id: str
    source_revision: str
    schema: pa.Schema
    source_fingerprint: dict
    metadata: dict = field(default_factory=dict)
    read_options: ReadOptions = field(default_factory=ReadOptions)
    cache_dir: Path | None = None
    _index: object = None
    _index_config: str | None = None
    _mode_corrections: tuple = ()
    _event_selection: object = None

    def verify(self):
        """Check the source quick fingerprint against the captured revision before reading; raise SourceChangedError when it changed."""
        if fingerprint(self.path) != {
            k: v for k, v in self.source_fingerprint.items() if k != "sha256"
        }:
            raise SourceChangedError(str(self.path))

    def manifest(self):
        """Return source identity, revision, schema, metadata, reader options, and applied corrections for provenance."""
        return {
            **runtime_provenance(),
            "manifest_version": 1,
            "trace_id": self.trace_id,
            "source_revision": self.source_revision,
            "source_path": str(self.path.resolve()),
            "fingerprint": self.source_fingerprint,
            "schema": str(self.schema),
            "metadata": deepcopy(self.metadata),
            "read_options": deepcopy(self.read_options.__dict__),
            "cache_dir": str(self.cache_dir),
            "step_corrections": deepcopy(list(self._mode_corrections)),
        }

    def save_manifest(self, path, strong=False):
        """Write the trace manifest; strong=True adds a streamed content hash suitable for stable source relocation."""
        m = self.manifest()
        if strong:
            m["fingerprint"] = fingerprint(self.path, True)
        Path(path).write_text(canonical(m), encoding="utf8")

    def select(self, **kwargs):
        """Return an immutable filtered view/selection; repeated predicates intersect. Equality, lists, and min/max intervals are supported. This never truncates scientific computation implicitly.

        Usage: call on the BatteryTrace object; returned selections/revisions are independent of earlier views."""
        return TraceSelection((self,)).select(**kwargs)

    def iter_batches(self):
        """Yield filtered/projected Arrow batches lazily, retaining compact identity columns and original order."""
        return self.select().iter_batches()

    def ensure_index(self, config=None, events=None, cancellation=None, on_progress=None):
        """Build/reuse the matching scientific assignment index and capture event/correction revisions without changing raw samples."""
        from .segmentation import build_index

        config = config or AnalysisConfig()
        key = digest(
            {
                "config": config.to_dict(),
                "events": events.to_dict() if events else None,
                "step_corrections": self._mode_corrections,
            }
        )
        self.verify()
        if self._index is None or self._index_config != key:
            self._index = build_index(self, config, events, cancellation, on_progress)
            if self._mode_corrections:
                from .corrections import apply

                self._index = apply(self._index, self, self._mode_corrections, config)
            self._index_config = key
        return self._index


def open_trace(
    path, *, metadata=None, read_options=None, manifest=None, cache_dir=None, strong_verify=False
):
    """Open a standardized Parquet/CSV/TXT source as a lightweight BatteryTrace; only schema/metadata are read initially. Example: trace = open_trace("cell.parquet"). Without a saved manifest this creates a fresh trace identity.

    Parameters
    ----------
    path
        Input or destination path as specified; scientific sources must not be overwritten.
    metadata
        User cell/protocol annotations; separate from recorded signal columns.
    read_options
        ReadOptions for source aliases, text conventions, and decoder limits.
    manifest
        Captured provenance dictionary or saved trace-manifest path, as specified by the operation.
    cache_dir
        Directory for private, rebuildable source indices and intermediate results.
    strong_verify
        Request content-hash verification rather than only quick size/mtime checks."""
    p = Path(path).resolve()
    options = read_options or ReadOptions()
    if manifest:
        m = json.loads(Path(manifest).read_text(encoding="utf8"))
        trace_id = m["trace_id"]
        rev = m["source_revision"]
        options = read_options or ReadOptions(**m.get("read_options", {}))
        metadata = metadata or m.get("metadata", {})
        old = m["fingerprint"]
        current = fingerprint(p, strong_verify or "sha256" in old)
        if "sha256" in old:
            if current.get("sha256") != old["sha256"]:
                raise SourceChangedError("Strong relocation check failed")
        elif {k: current[k] for k in old} != old:
            raise SourceChangedError("Quick source check failed; relocation needs strong manifest")
    else:
        trace_id = uuid.uuid4().hex
        rev = uuid.uuid4().hex
    fp = fingerprint(p)
    if p.suffix.lower() == ".parquet":
        f = pq.ParquetFile(p)
        schema = f.schema_arrow
        f.close()
        schema = pa.schema(
            [pa.field(options.column_mapping.get(x.name, x.name), x.type) for x in schema]
        )
    else:
        first = next(source_batches(p, replace(options, batch_rows=1)), None)
        if first is None:
            raise SchemaError("Empty text file without schema")
        schema = first.schema
    if len(set(schema.names)) != len(schema.names):
        raise SchemaError("Mapping creates duplicate names")
    missing = set(REQUIRED) - set(schema.names)
    if missing:
        raise SchemaError(
            f"Missing standardized columns {sorted(missing)}; provide ReadOptions(column_mapping=...)"
        )
    collisions = set(schema.names) & RESERVED
    if collisions:
        raise SchemaError(
            f"Reserved output names present in source: {sorted(collisions)}; map them explicitly"
        )
    for n in REQUIRED:
        if not pa.types.is_floating(schema.field(n).type) and not pa.types.is_integer(
            schema.field(n).type
        ):
            raise SchemaError(f"{n} must be numeric seconds/mA/V")
    trace = BatteryTrace(
        p,
        trace_id,
        rev,
        schema,
        fp,
        metadata or {},
        options,
        Path(cache_dir) if cache_dir else p.parent / ".gregorithm-cache" / trace_id,
    )
    if manifest:
        trace._mode_corrections = tuple(m.get("step_corrections", ()))
    return trace


@dataclass(frozen=True)
class TraceSelection:
    """Immutable lazy source selection with configuration, predicates, projection, and preview limit. Use trace.select(cycle_id=2) then iter_batches(), analyze(), or bounded to_pandas()."""

    traces: tuple
    clauses: tuple = ()
    columns: tuple | None = None
    row_limit: int | None = None
    config: AnalysisConfig = field(default_factory=AnalysisConfig)
    assignment_revision: str | None = None

    def select(self, columns=None, **predicates):
        """Return an immutable filtered view/selection; repeated predicates intersect. Equality, lists, and min/max intervals are supported. This never truncates scientific computation implicitly.

        Usage: call on the TraceSelection object; returned selections/revisions are independent of earlier views."""
        if "time_range" in predicates:
            a, b = predicates.pop("time_range")
            predicates["time_s"] = {"min": a, "max": b}
        return replace(
            self,
            clauses=self.clauses + ((predicates,) if predicates else ()),
            columns=self.columns if columns is None else tuple(columns),
        )

    def head(self, n=5):
        """Return an explicitly row-limited preview. Negative limits are rejected; this is not a scientific downsampling operation.

        Usage: call on the TraceSelection object; returned selections/revisions are independent of earlier views."""
        if n < 0:
            raise ValueError("head >=0")
        return replace(self, row_limit=n if self.row_limit is None else min(n, self.row_limit))

    def with_config(self, config):
        """Return the same declarative trace selection with a replacement validated AnalysisConfig."""
        return replace(self, config=config)

    def to_dict(self):
        """Return serializable configuration/provenance fields without materializing signal data."""
        return {
            "version": 1,
            "traces": [
                {"trace_id": t.trace_id, "source_revision": t.source_revision} for t in self.traces
            ],
            "clauses": self.clauses,
            "columns": self.columns,
            "head": self.row_limit,
            "assignment_revision": self.assignment_revision,
        }

    @property
    def selection_id(self):
        """Return a canonical digest of source revisions, predicates, projection, and explicit preview limit."""
        return digest(self.to_dict())

    def intersection(self, other):
        """Combine compatible selections with logical AND; different source or assignment revisions are rejected."""
        if self.traces != other.traces or self.assignment_revision != other.assignment_revision:
            raise RevisionMismatchError("Selection revisions/sources differ")
        return replace(self, clauses=self.clauses + other.clauses)

    def union(self, other):
        """Combine compatible selections with logical OR; preserve original order and count overlapping intervals once."""
        if self.traces != other.traces or self.assignment_revision != other.assignment_revision:
            raise RevisionMismatchError("Selection revisions/sources differ")
        return CombinedSelection(self, other)

    def iter_traces(self):
        """Yield separate (trace_id, TableView) pairs; no concatenated DataFrame or simultaneous full-file materialization is created."""
        from .segmentation import annotated_batches

        for t in self.traces:
            keep = True
            clauses = []
            for c in self.clauses:
                c = dict(c)
                for name, actual in (("trace_id", t.trace_id), ("file", str(t.path))):
                    if name in c:
                        wanted = c.pop(name)
                        wanted = wanted if isinstance(wanted, (list, tuple)) else [wanted]
                        if actual not in wanted and (name != "file" or t.path.name not in wanted):
                            keep = False
                clauses.append(c)
            if not keep:
                continue
            from .io import known_empty, resolve_source_clauses

            clauses = resolve_source_clauses(clauses, t)
            if known_empty(clauses):
                continue
            selection = replace(self, traces=(t,), clauses=tuple(clauses))
            need = any(
                predicate_columns((c,))
                & {
                    "cycle_id",
                    "half_cycle_id",
                    "step_id",
                    "direction",
                    "mode",
                    "time_section_id",
                    "event_id",
                }
                for c in clauses
            )

            def factory(t=t, selection=selection, need=need):
                """Produce this captured lazy view in bounded batches; retain its source and scientific revision."""
                t.verify()
                if "time_s" in predicate_columns(selection.clauses):
                    from .segmentation import validate_time_selection

                    validate_time_selection(t.ensure_index(selection.config), selection.clauses)
                if need:
                    idx = t.ensure_index(selection.config)
                    if (
                        selection.assignment_revision
                        and idx.revision != selection.assignment_revision
                    ):
                        raise RevisionMismatchError("Obsolete assignments")
                    blocks = annotated_batches(
                        t,
                        idx,
                        selection.config,
                        columns=selection.columns,
                        clauses=selection.clauses,
                    )
                    if "event_id" in predicate_columns(selection.clauses):
                        from .analysis import _event_columns
                        from .events import detect_events

                        events = t._event_selection or detect_events(t, selection.config)
                        for block in blocks:
                            yield _event_columns(block, events)
                    else:
                        yield from blocks
                else:
                    # Projection includes columns required by predicates and compact source keys.
                    names = (
                        None
                        if selection.columns is None
                        else tuple(set(selection.columns) | predicate_columns(selection.clauses))
                    )
                    offset = 0
                    for b in source_batches(
                        t.path,
                        replace(t.read_options, batch_rows=selection.config.batch_rows),
                        names,
                    ):
                        n = b.num_rows
                        yield (
                            b.append_column(
                                "sample_id", pa.array(np.arange(offset, offset + n, dtype=np.int64))
                            )
                            .append_column("trace_id", pa.array([t.trace_id] * n))
                            .append_column("source_revision", pa.array([t.source_revision] * n))
                        )
                        offset += n
                t.verify()

            yield (
                t.trace_id,
                TableView(
                    factory,
                    {
                        **t.manifest(),
                        "selection": selection.to_dict(),
                        "selection_id": selection.selection_id,
                    },
                    tuple(clauses),
                    selection.columns,
                    selection.row_limit,
                    selection.config.managed_memory_bytes,
                ),
            )

    def iter_batches(self):
        """Yield filtered/projected Arrow batches lazily, retaining compact identity columns and original order."""
        for _, view in self.iter_traces():
            yield from view.iter_batches()

    def to_pandas(self, **kw):
        """Materialize only this explicitly selected view, checking row and conservative copy-byte limits before concatenation. Exceeding a limit raises MaterializationLimitError without truncation.

        Usage: call on the TraceSelection object; returned selections/revisions are independent of earlier views."""
        return TableView(
            self.iter_batches,
            {"selection": self.to_dict()},
            memory_bytes=self.config.managed_memory_bytes,
        ).to_pandas(**kw)

    def write_parquet(self, path):
        """Export this view incrementally with a provenance sidecar; refuse to overwrite its original source.

        Usage: call on the TraceSelection object; returned selections/revisions are independent of earlier views."""
        if any(Path(path).resolve() == t.path.resolve() for t in self.traces):
            raise ValueError("Cannot overwrite source")
        return TableView(
            self.iter_batches,
            {"selection": self.to_dict(), "sources": [t.manifest() for t in self.traces]},
        ).write_parquet(path)

    def relative_time(self, origin=None):
        """Add relative_time_s using the supplied origin or the first selected sample. Preserve original time_s and all numerical accumulation origins."""
        if len(self.traces) != 1:
            raise ValueError("Align separately via iter_traces")
        return next(self.iter_traces())[1].relative_time(origin)


@dataclass(frozen=True)
class CombinedSelection:
    """Compatible logical union of two trace selections. Original sample order and identities remain stable; scientific overlaps are integrated once."""

    left: TraceSelection
    right: TraceSelection

    def as_selection(self):
        """Represent a combined union as nested declarative OR clauses for shared selection and scientific interval clipping."""
        if self.left.row_limit is not None or self.right.row_limit is not None:
            raise ValueError(
                "Union head-limited previews only after explicit export; use head on the union instead"
            )
        columns = self.left.columns if self.left.columns == self.right.columns else None
        return replace(
            self.left,
            clauses=({"__or__": [self.left.clauses, self.right.clauses]},),
            columns=columns,
        )

    def to_dict(self):
        """Return serializable configuration/provenance fields without materializing signal data."""
        return self.as_selection().to_dict()

    def iter_traces(self):
        """Yield separate (trace_id, TableView) pairs; no concatenated DataFrame or simultaneous full-file materialization is created."""
        return self.as_selection().iter_traces()

    def iter_batches(self):
        """Yield filtered/projected Arrow batches lazily, retaining compact identity columns and original order."""
        return self.as_selection().iter_batches()

    def to_pandas(self, **kw):
        """Materialize only this explicitly selected view, checking row and conservative copy-byte limits before concatenation. Exceeding a limit raises MaterializationLimitError without truncation.

        Usage: call on the CombinedSelection object; returned selections/revisions are independent of earlier views."""
        return self.as_selection().to_pandas(**kw)

    def write_parquet(self, path):
        """Export this view incrementally with a provenance sidecar; refuse to overwrite its original source.

        Usage: call on the CombinedSelection object; returned selections/revisions are independent of earlier views."""
        return self.as_selection().write_parquet(path)

    def select(self, **kw):
        """Return an immutable filtered view/selection; repeated predicates intersect. Equality, lists, and min/max intervals are supported. This never truncates scientific computation implicitly.

        Usage: call on the CombinedSelection object; returned selections/revisions are independent of earlier views."""
        return self.as_selection().select(**kw)

    def head(self, n=5):
        """Return an explicitly row-limited preview. Negative limits are rejected; this is not a scientific downsampling operation.

        Usage: call on the CombinedSelection object; returned selections/revisions are independent of earlier views."""
        return self.as_selection().head(n)


@dataclass(frozen=True)
class TraceCollection:
    """Independent trace handles for multi-file analysis. Iterate sources or request a long BatchResult table; it deliberately has no ambiguous all-file to_pandas()."""

    traces: tuple

    def select(self, **kw):
        """Return an immutable filtered view/selection; repeated predicates intersect. Equality, lists, and min/max intervals are supported. This never truncates scientific computation implicitly.

        Usage: call on the TraceCollection object; returned selections/revisions are independent of earlier views."""
        return TraceSelection(self.traces).select(**kw)

    def iter_traces(self):
        """Yield separate (trace_id, TableView) pairs; no concatenated DataFrame or simultaneous full-file materialization is created."""
        return TraceSelection(self.traces).iter_traces()

    def __iter__(self):
        """Iterate the independent source handles, retaining failed-source placeholders where present."""
        return iter(self.traces)

    def __len__(self):
        """Return the number of independent source handles without reading data."""
        return len(self.traces)


@dataclass
class FailedTrace(BatteryTrace):
    """Retain a failed opening attempt so other files can continue in a batch."""

    open_error: str = ""

    def verify(self):
        """Check the source quick fingerprint against the captured revision before reading; raise SourceChangedError when it changed."""
        raise SchemaError(self.open_error)

    def ensure_index(self, *args, **kwargs):
        """Build/reuse the matching scientific assignment index and capture event/correction revisions without changing raw samples."""
        raise SchemaError(self.open_error)


def open_traces(paths, **kwargs):
    """Open independent sources as a TraceCollection, retaining opening failures per file rather than aborting the collection. Example: analyze(open_traces(paths)).

    Parameters
    ----------
    paths
        Iterable of independent input paths; each file retains its own identity.
    kwargs
        Keyword overrides forwarded to the documented configuration or shared pipeline."""
    traces = []
    for path in paths:
        try:
            traces.append(open_trace(path, **kwargs))
        except Exception as error:
            p = Path(path).resolve()
            trace_id = uuid.uuid4().hex
            traces.append(
                FailedTrace(
                    p,
                    trace_id,
                    uuid.uuid4().hex,
                    pa.schema([]),
                    {},
                    kwargs.get("metadata") or {},
                    kwargs.get("read_options") or ReadOptions(),
                    Path(kwargs["cache_dir"])
                    if kwargs.get("cache_dir")
                    else p.parent / ".gregorithm-cache" / trace_id,
                    open_error=f"{type(error).__name__}: {error}",
                )
            )
    return TraceCollection(tuple(traces))


@dataclass
class AnalysisResult:
    """One trace analysis outcome with lazy samples, metrics, curves, events, segments, diagnostics, manifest, and completion status. Partial does not mean experimentally validated."""

    trace: BatteryTrace
    result_id: str
    analysis_id: str
    assignment_revision: str
    status: str
    manifest: dict
    directory: Path
    samples: TableView
    metrics: TableView
    curves: TableView
    events: TableView
    segments: TableView
    diagnostics: TableView
    dcir: TableView | None = None

    def select(self, **kw):
        """Return an immutable filtered view/selection; repeated predicates intersect. Equality, lists, and min/max intervals are supported. This never truncates scientific computation implicitly.

        Usage: call on the AnalysisResult object; returned selections/revisions are independent of earlier views."""
        return self.samples.select(**kw)


@dataclass
class BatchResult:
    """Per-trace outcomes in by_trace with explicit failure/cancellation isolation. table(name) exposes long results without discarding trace identity."""

    by_trace: dict

    def table(self, name="metrics"):
        """Return a lazy long table retaining trace identity in each row; no wide pivot or all-file pandas materialization is implicit."""
        results = tuple(self.by_trace.values())

        def batches():
            """Yield bounded rows from the indexed segment/source table, retaining explicit scientific identity."""
            for r in results:
                if r is not None:
                    table = getattr(r, name, None)
                    if table:
                        yield from table.iter_batches()

        return TableView(
            batches, {"manifest_version": 1, "parents": [r.manifest for r in results if r]}
        )
