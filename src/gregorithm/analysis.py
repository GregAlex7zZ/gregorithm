"""Composable per-trace pipeline and bounded Windows spawn batch coordinator."""

import json
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import AnalysisConfig, canonical
from .diagnostics import Cancelled
from .events import EventSelection, detect_events
from .execution import CancellationToken, Reporter
from .integration import integrated_batches, integration_backend, normalize
from .io import TableView, atomic_parquet, filter_batch, parquet_view, replace_file
from .model import (
    AnalysisResult,
    BatchResult,
    BatteryTrace,
    CombinedSelection,
    TraceCollection,
    TraceSelection,
    open_trace,
)
from .provenance import runtime_provenance
from .summary import SummaryStore

REGISTRY = {
    name: {"version": "1", "units": units, "prerequisites": prereq}
    for name, units, prereq in [
        ("capacity", "mAh", ["segmentation"]),
        ("energy", "Wh", ["segmentation"]),
        ("summary", "long", ["capacity", "energy"]),
        ("dqdv", "mAh/V", ["capacity"]),
        ("events", "event", ["observed_runs"]),
        ("c_rate", "h^-1", ["capacity"]),
    ]
}


def _event_columns(block, events):
    """Attach event membership to original sample rows while retaining overlapping event IDs and a scalar first-membership selector."""
    if not events.events:
        return block
    ids = block.column("sample_id").to_numpy(zero_copy_only=False)
    matches = [[] for _ in range(block.num_rows)]
    for event in events.events:
        start = event.get("sample_t0")
        if start is None:
            start = event["sample_t1"]
        end = event.get("sample_t3")
        if end is None:
            end = event["sample_t2"]
        for j in np.flatnonzero((ids >= start) & (ids <= end)):
            matches[j].append(event["event_id"])
    return block.append_column(
        "event_ids", pa.array(matches, type=pa.list_(pa.string()))
    ).append_column("event_id", pa.array([m[0] if m else None for m in matches], type=pa.string()))


METRIC_SCHEMA = pa.schema(
    [
        (name, kind)
        for name, kind in [
            ("trace_id", pa.string()),
            ("result_id", pa.string()),
            ("assignment_revision", pa.string()),
            ("scope", pa.string()),
            ("group_id", pa.int64()),
            ("view", pa.string()),
            ("direction", pa.string()),
            ("half_cycle_id", pa.int64()),
            ("cycle_id", pa.int64()),
            ("duration_observed_s", pa.float64()),
            ("selection_partial", pa.bool_()),
            ("protocol_complete", pa.bool_()),
            ("metric", pa.string()),
            ("value", pa.float64()),
            ("unit", pa.string()),
            ("voltage_channel", pa.string()),
            ("quality", pa.string()),
            ("invalid_intervals", pa.int64()),
            ("reason", pa.string()),
        ]
    ]
)
DIAGNOSTIC_SCHEMA = pa.schema(
    [
        ("trace_id", pa.string()),
        ("result_id", pa.string()),
        ("code", pa.string()),
        ("severity", pa.string()),
        ("sample_start", pa.int64()),
        ("sample_end", pa.int64()),
        ("half_cycle_id", pa.int64()),
        ("analysis", pa.string()),
        ("consequence", pa.string()),
        ("action", pa.string()),
    ]
)


def _row_batches(rows, size=4096, schema=None):
    """Convert a row iterator into bounded Arrow record batches, preserving an explicit schema for empty results."""
    buf = []
    seen = False
    for row in rows:
        buf.append(row)
        if len(buf) >= size:
            batch = pa.RecordBatch.from_pylist(buf, schema=schema)
            schema = batch.schema
            yield batch
            seen = True
            buf = []
    if buf:
        yield pa.RecordBatch.from_pylist(buf, schema=schema)
    elif not seen and schema is not None:
        yield pa.RecordBatch.from_pylist([], schema=schema)


def _one(
    selection,
    *,
    analyses,
    output_dir=None,
    export_samples=False,
    events=None,
    auto_exclude=False,
    on_progress=None,
    cancellation=None,
):
    """Execute one trace selection and commit a completion manifest last; failures/cancellation retain an explicit noncomplete manifest."""
    trace = selection.traces[0]
    from .io import resolve_source_clauses

    selection = replace(selection, clauses=resolve_source_clauses(selection.clauses, trace))
    cfg = selection.config
    token = cancellation or CancellationToken()
    rid = uuid.uuid4().hex
    start_clock = perf_counter()
    aid = uuid.uuid4().hex
    directory = Path(output_dir or trace.cache_dir / "results") / rid
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        **trace.manifest(),
        "result_id": rid,
        "analysis_id": aid,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm_version": "1",
        "config": cfg.to_dict(),
        "config_hash": cfg.config_hash,
        "selection": selection.to_dict(),
        "selection_id": selection.selection_id,
        "parent_result_ids": [],
        "analyses": list(analyses),
        "files": {},
        "validation": "software_result_not_experimental_validation",
    }
    reporter = Reporter(on_progress, aid, trace.trace_id)
    issues = []
    diagnostic_overflow = 0
    store = None
    idx = None

    def issue(code, start, end, consequence):
        # Coalesce adjacent intervals; metadata is bounded, overflow is counted.
        """Record bounded/coalesced diagnostic support and its scientific consequence."""
        nonlocal diagnostic_overflow
        branch = start if code.startswith("dqdv_") else None
        if branch is not None:
            start = end = None
        if (
            issues
            and issues[-1]["code"] == code
            and issues[-1]["sample_end"] is not None
            and start is not None
            and issues[-1]["sample_end"] + 1 == start
        ):
            issues[-1]["sample_end"] = end
        elif len(issues) < 10000:
            issues.append(
                {
                    "trace_id": trace.trace_id,
                    "result_id": rid,
                    "code": code,
                    "severity": "warning",
                    "sample_start": start,
                    "sample_end": end,
                    "half_cycle_id": branch,
                    "analysis": "dqdv"
                    if branch is not None
                    else "c_rate"
                    if code.startswith("c_rate_")
                    else "integration",
                    "consequence": consequence,
                    "action": "Inspect original data",
                }
            )
        else:
            diagnostic_overflow += 1

    try:
        token.check()
        idx = trace.ensure_index(cfg, cancellation=token, on_progress=on_progress)
        from .segmentation import validate_time_selection

        validate_time_selection(idx, selection.clauses)
        if events is not None:
            events = events.select(trace_id=trace.trace_id)
            if any(
                event.get("source_revision") != trace.source_revision for event in events.events
            ):
                raise ValueError("Event source revision mismatch")
        from .io import predicate_columns

        if events is None and "event_id" in predicate_columns(selection.clauses):
            events = trace._event_selection or detect_events(trace, cfg)
        if "events" in analyses or auto_exclude:
            events = events or detect_events(trace, cfg)
        events = events or EventSelection()
        trace._event_selection = events
        if auto_exclude:
            events = events.auto_accept()
        if events.accepted_ranges():
            idx = trace.ensure_index(cfg, events, token, on_progress)
        manifest["assignment_revision"] = idx.revision
        manifest["event_revision"] = events.revision
        manifest["event_parent_revision"] = events.parent_revision
        manifest["event_metadata"] = events.metadata
        from .source_index import batches as source_index_batches

        atomic_parquet(
            source_index_batches(trace, idx.rows, rid), directory / "source_index.parquet", manifest
        )
        manifest["files"]["source_index"] = "source_index.parquet"

        def segment_batches():
            """Yield segment rows with result identity."""
            for block in idx.batches(table="steps"):
                yield block.append_column("result_id", pa.array([rid] * block.num_rows))

        atomic_parquet(segment_batches(), directory / "segments.parquet", manifest)
        manifest["files"]["segments"] = "segments.parquet"
        atomic_parquet(
            events.table(manifest).iter_batches(), directory / "events.parquet", manifest
        )
        manifest["files"]["events"] = "events.parquet"
        head_end = None
        if selection.row_limit:
            from .segmentation import annotated_batches

            remaining = selection.row_limit
            for block in annotated_batches(trace, idx, cfg):
                block = filter_batch(_event_columns(block, events), selection.clauses)
                if block.num_rows:
                    take = min(remaining, block.num_rows)
                    head_end = block.column("sample_id")[take - 1].as_py()
                    remaining -= take
                if not remaining:
                    break
        store = SummaryStore(
            directory / "summary.sqlite",
            selection,
            cfg,
            head_end=head_end,
            events=events,
            index=idx,
        )
        store.analyses = analyses
        spool_needed = export_samples or "dqdv" in analyses or "c_rate" in analyses
        spool_path = (
            directory / "samples.parquet"
            if export_samples
            else directory / "analysis_support.parquet"
        )
        num = 0
        partial = False
        rows_selected = 0
        completed_batches = 0
        sample_schema = None
        previous_coverage_row = None

        def scan():
            """Scan bounded original data while carrying the containing operation state across batches."""
            nonlocal \
                num, \
                partial, \
                rows_selected, \
                previous_coverage_row, \
                completed_batches, \
                sample_schema
            for nb, b in enumerate(
                integrated_batches(trace, idx, cfg, events, token, store.add, issue)
            ):
                b = _event_columns(b, events)
                num += b.num_rows
                # Invalid intervals remain missing and invalidate affected references/efficiencies.
                qvalid = b.column("q_interval_valid").to_numpy(zero_copy_only=False)
                evalid = b.column("e_interval_valid").to_numpy(zero_copy_only=False)
                invalid_groups = {}
                excluded_ranges = events.accepted_time_ranges()
                selected_mask = np.ones(b.num_rows, dtype=bool)
                if selection.clauses:
                    from .io import selection_mask

                    selected_mask = selection_mask(b, selection.clauses)
                for j in np.flatnonzero(~qvalid | ~evalid):
                    partial = True
                    group = tuple(
                        b.column(k)[j].as_py() for k in ("half_cycle_id", "cycle_id", "step_id")
                    )
                    right = {
                        k: b.column(k)[j].as_py()
                        for k in (
                            "time_s",
                            "time_section_id",
                            "sample_id",
                            "half_cycle_id",
                            "cycle_id",
                            "step_id",
                            "observed_sign",
                        )
                    }
                    left = (
                        {k: b.column(k)[j - 1].as_py() for k in right}
                        if j
                        else previous_coverage_row
                    )
                    fully_excluded = False
                    overlaps_excluded = False
                    if left and left["time_section_id"] == right["time_section_id"]:
                        a, z = left["time_s"], right["time_s"]
                        domains = sorted(
                            (max(a, lo), min(z, hi))
                            for sec, lo, hi in excluded_ranges
                            if sec == right["time_section_id"] and min(z, hi) > max(a, lo)
                        )
                        overlaps_excluded = bool(domains)
                        end = a
                        for lo, hi in domains:
                            if lo > end:
                                break
                            end = max(end, hi)
                        fully_excluded = z > a and end >= z
                    views = ["measured"]
                    if not fully_excluded:
                        views.append("cycling")
                    if overlaps_excluded:
                        views.append("excluded")
                    owners = [(group, None)]
                    if left and left["time_section_id"] == right["time_section_id"]:
                        left_group = tuple(
                            left[k] for k in ("half_cycle_id", "cycle_id", "step_id")
                        )
                        for pos, scope in enumerate(("half_cycle", "cycle", "step")):
                            if left_group[pos] != group[pos]:
                                owners.append((left_group, scope))
                    for assigned_group, only_scope in owners:
                        selected_interval = bool(selected_mask[j])
                        if (
                            left
                            and np.isfinite(right["time_s"] - left["time_s"])
                            and right["time_s"] > left["time_s"]
                        ):
                            from .io import expand_clauses

                            descriptor = (
                                *assigned_group,
                                right["time_section_id"],
                                1,
                                "measured",
                                0.0,
                                0.0,
                                right["time_s"] - left["time_s"],
                                False,
                                False,
                                right["sample_id"],
                                left["time_s"],
                                0.0,
                                0.0,
                                right["time_s"],
                                0.0,
                                0.0,
                                (),
                                left["sample_id"],
                                left["step_id"],
                                right["observed_sign"],
                                left["observed_sign"],
                            )
                            selected_interval = any(
                                store._bounds(descriptor, leaf)
                                for leaf in expand_clauses(selection.clauses)
                            )
                        for view in views:
                            for selected_flag in (0, 1):
                                if selected_flag and not selected_interval:
                                    continue
                                if (
                                    selected_flag
                                    and head_end is not None
                                    and right["sample_id"] > head_end
                                ):
                                    continue
                                key = (*assigned_group, view, selected_flag)
                                if only_scope is not None:
                                    key += (only_scope,)
                                counts = invalid_groups.setdefault(key, [0, 0])
                                counts[0] += int(not qvalid[j])
                                counts[1] += int(not evalid[j])
                if b.num_rows:
                    previous_coverage_row = {
                        k: b.column(k)[-1].as_py()
                        for k in (
                            "time_s",
                            "time_section_id",
                            "sample_id",
                            "half_cycle_id",
                            "cycle_id",
                            "step_id",
                            "observed_sign",
                        )
                    }
                store.mark_invalid_groups(invalid_groups)
                selected = filter_batch(b, selection.clauses)
                rows_selected += selected.num_rows
                if selection.row_limit is not None:
                    rows_selected = min(rows_selected, selection.row_limit)
                completed_batches = nb + 1
                b = b.append_column("result_id", pa.array([rid] * b.num_rows))
                sample_schema = b.schema
                reporter.emit(
                    "analysis",
                    rows_read=num,
                    rows_processed=num,
                    rows_selected=rows_selected,
                    batches_completed=nb + 1,
                    total=idx.rows,
                    percent=100 * num / idx.rows if idx.rows else None,
                )
                if spool_needed:
                    if not export_samples:
                        b = b.select(
                            [
                                "trace_id",
                                "result_id",
                                "sample_id",
                                "time_s",
                                "current_mA",
                                "voltage_V",
                                "time_section_id",
                                "half_cycle_id",
                                "cycle_id",
                                "direction",
                                "capacity_mAh",
                                "excluded",
                                "q_interval_valid",
                            ]
                        )
                    yield b

        if spool_needed:
            atomic_parquet(scan(), spool_path, manifest, cfg.compression, cfg.compression_level)
        else:
            for _ in scan():
                pass
        reference = store.resolve_reference(idx) if "c_rate" in analyses else None
        manifest["c_rate_reference"] = reference
        if reference and reference["value_mAh"] is None:
            issue(
                "c_rate_reference_unavailable",
                0,
                idx.rows - 1,
                "C-rate unavailable; no implicit fallback",
            )
        atomic_parquet(
            _row_batches(
                (
                    {**row, "voltage_channel": trace.metadata.get("voltage_channel", "cell")}
                    for row in store.rows(idx, trace.trace_id, rid)
                ),
                schema=METRIC_SCHEMA,
            ),
            directory / "metrics.parquet",
            manifest,
        )
        manifest["files"]["metrics"] = "metrics.parquet"

        def sample_factory():
            """Reproduce captured sample output lazily from persisted data or original integration."""
            if export_samples:
                yield from parquet_view(
                    spool_path, manifest, cfg.managed_memory_bytes
                ).iter_batches()
            else:
                for b in integrated_batches(trace, idx, cfg, events):
                    yield _event_columns(b, events).append_column(
                        "result_id", pa.array([rid] * b.num_rows)
                    )

        samples = TableView(
            sample_factory,
            manifest,
            selection.clauses,
            None,
            selection.row_limit,
            cfg.managed_memory_bytes,
            str(spool_path) if export_samples else None,
            declared_schema=sample_schema,
        )
        if cfg.mass_g is not None or cfg.area_cm2 is not None:
            samples = normalize(samples, mass_g=cfg.mass_g, area_cm2=cfg.area_cm2)
        if reference:
            base = samples

            def with_rate():
                """Append signed C-rate using the captured fixed reference without replacing source columns."""
                for b in base.iter_batches():
                    value = reference["value_mAh"]
                    yield b.append_column(
                        "c_rate_h_inv",
                        pc.divide(b.column("current_mA"), pa.scalar(value, type=pa.float64())),
                    )

            samples = TableView(
                with_rate,
                manifest,
                memory_bytes=cfg.managed_memory_bytes,
                declared_schema=base.schema.append(pa.field("c_rate_h_inv", pa.float64()))
                if base.schema is not None
                else None,
            )
        if selection.columns is not None:
            samples = samples.select(columns=selection.columns)
        if export_samples:
            if (
                cfg.mass_g is not None
                or cfg.area_cm2 is not None
                or reference
                or selection.clauses
                or selection.columns is not None
                or selection.row_limit is not None
            ):
                atomic_parquet(
                    samples.iter_batches(),
                    directory / "final_samples.parquet",
                    manifest,
                    cfg.compression,
                    cfg.compression_level,
                )
                spool_path = directory / "final_samples.parquet"
                samples = parquet_view(spool_path, manifest, cfg.managed_memory_bytes)
            manifest["files"]["samples"] = spool_path.name
        curves_meta = []

        def curves():
            """Yield independent half-cycle derivative curves and retain each curve manifest."""
            from .dqdv import compute_dqdv

            if "dqdv" not in analyses:
                return
            support = parquet_view(
                directory / ("samples.parquet" if export_samples else "analysis_support.parquet"),
                manifest,
                cfg.managed_memory_bytes,
            )
            with idx.connect() as db:
                for r in db.execute(
                    "select distinct half_cycle_id,main_sign from runs where half_cycle_id>0 and main_sign!=0 order by half_cycle_id"
                ):
                    token.check()
                    h = r[0]
                    branch = support.select(half_cycle_id=h)
                    branch = replace(branch, clauses=branch.clauses + selection.clauses)
                    if selection.row_limit is not None:
                        branch = (
                            branch.select(sample_id={"max": head_end})
                            if head_end is not None
                            else branch.select(sample_id=[])
                        )
                    try:
                        curve = compute_dqdv(
                            branch,
                            config=cfg.dqdv,
                            rest_threshold_mA=cfg.rest_threshold_mA,
                            manifest=manifest,
                        )
                        curves_meta.append(curve.manifest)
                        yield from curve.iter_batches()
                    except ValueError as e:
                        issue("dqdv_insufficient_branch", h, h, str(e))

        atomic_parquet(curves(), directory / "curves.parquet", manifest)
        manifest["files"]["curves"] = "curves.parquet"
        manifest["curve_manifests"] = curves_meta
        atomic_parquet(
            _row_batches(issues, schema=DIAGNOSTIC_SCHEMA),
            directory / "diagnostics.parquet",
            manifest,
        )
        manifest["files"]["diagnostics"] = "diagnostics.parquet"
        trace.verify()
        manifest["backend"] = integration_backend(cfg.backend)[1]
        manifest["rows_processed"] = num
        manifest["diagnostic_overflow_count"] = diagnostic_overflow
        manifest["status"] = "partial" if partial else "complete"
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_s"] = perf_counter() - start_clock
        manifest["selection_metrics_partial"] = bool(
            selection.clauses or selection.row_limit is not None
        )
        manifest["notes"] = [
            "Initial index scan is required for derived ID selections",
            "CSV/TXT range access remains sequential",
            "CC/CV uses bounded nonoverlapping observation windows; heuristic, no protocol certainty",
        ]
        store.close()
        store = None
        for filename in manifest["files"].values():
            sidecar = directory / (filename + ".manifest.json")
            staged = sidecar.with_suffix(sidecar.suffix + ".tmp")
            staged.write_text(canonical(manifest), encoding="utf8")
            replace_file(staged, sidecar)
        # The completion manifest is the final publication step for the directory.
        temp = directory / "manifest.json.tmp"
        temp.write_text(canonical(manifest), encoding="utf8")
        replace_file(temp, directory / "manifest.json")
        reporter.emit(
            "completion",
            manifest["status"],
            rows_processed=num,
            rows_read=num,
            rows_selected=rows_selected,
            batches_completed=completed_batches,
            total=idx.rows,
            percent=100.0,
        )
        return AnalysisResult(
            trace,
            rid,
            aid,
            idx.revision,
            manifest["status"],
            manifest,
            directory,
            samples,
            parquet_view(directory / "metrics.parquet", manifest, cfg.managed_memory_bytes),
            parquet_view(directory / "curves.parquet", manifest, cfg.managed_memory_bytes),
            events.table(manifest),
            parquet_view(directory / "segments.parquet", manifest, cfg.managed_memory_bytes),
            parquet_view(directory / "diagnostics.parquet", manifest, cfg.managed_memory_bytes),
        )
    except BaseException as e:
        if store:
            store.close()
        manifest["status"] = "cancelled" if isinstance(e, Cancelled) else "failed"
        manifest["failure"] = {"type": type(e).__name__, "message": str(e)}
        (directory / "manifest.json").write_text(canonical(manifest), encoding="utf8")
        reporter.emit("completion", manifest["status"])
        e.result_directory = directory
        raise


def analyze(
    source,
    analyses=("capacity", "energy", "summary"),
    *,
    config=None,
    output_dir=None,
    export_samples=False,
    events=None,
    auto_exclude=False,
    on_progress=None,
    cancellation=None,
):
    """Run the requested analyses with bounded batches and persistent provenance. Return AnalysisResult for one trace or BatchResult for a collection. Example: analyze(trace, analyses=("capacity", "energy"), config=AnalysisConfig()). Windows multi-worker scripts must use a main guard.

    Parameters
    ----------
    source
        Trace, selection, collection, or supported lazy result/view; see the return-scope contract.
    analyses
        Requested registry names: capacity, energy, summary, dqdv, events, c_rate.
    config
        Validated immutable AnalysisConfig or DQDVConfig as appropriate; None uses operational defaults.
    output_dir
        Parent directory for a new result-ID subdirectory; manifest is committed last.
    export_samples
        Persist sample output when True; otherwise retain reproducible bounded lazy views.
    events
        Explicit EventSelection with source identity and decision/boundary revision.
    auto_exclude
        Explicitly enable the inverse_return event policy; default False.
    on_progress
        Callable receiving serializable ProgressEvent records during execution.
    cancellation
        CancellationToken checked at cooperative boundaries; None creates a local token."""
    if isinstance(source, CombinedSelection):
        source = source.as_selection()
    unknown = set(analyses) - REGISTRY.keys()
    if unknown:
        raise ValueError(f"Unknown analyses: {unknown}")
    if isinstance(source, BatteryTrace):
        selection = TraceSelection((source,))
    elif isinstance(source, TraceCollection):
        selection = TraceSelection(source.traces)
    elif isinstance(source, TraceSelection):
        selection = source
    else:
        raise TypeError("BatteryTrace, TraceCollection or TraceSelection required")
    if config is not None:
        selection = selection.with_config(config)
    selected_traces = []
    for trace in selection.traces:
        keep = True
        for clause in selection.clauses:
            for key, actual in (("trace_id", trace.trace_id), ("file", str(trace.path))):
                if key in clause:
                    wanted = (
                        clause[key] if isinstance(clause[key], (list, tuple)) else [clause[key]]
                    )
                    keep &= actual in wanted or (key == "file" and trace.path.name in wanted)
        if keep:
            selected_traces.append(trace)
    selection = replace(
        selection,
        traces=tuple(selected_traces),
        clauses=tuple(
            {k: v for k, v in c.items() if k not in {"trace_id", "file"}} for c in selection.clauses
        ),
    )
    if len(selection.traces) == 1 and not isinstance(source, TraceCollection):
        return _one(
            selection,
            analyses=analyses,
            output_dir=output_dir,
            export_samples=export_samples,
            events=events,
            auto_exclude=auto_exclude,
            on_progress=on_progress,
            cancellation=cancellation,
        )
    if selection.config.workers > 1:
        return _parallel(
            selection,
            analyses,
            output_dir,
            export_samples,
            events,
            auto_exclude,
            on_progress,
            cancellation,
        )
    results = {}
    batch_reporter = Reporter(on_progress, uuid.uuid4().hex)
    for trace in selection.traces:
        if cancellation and cancellation.cancelled:
            results[trace.trace_id] = failed_result(
                trace, "Cancelled before start", output_dir, "cancelled"
            )
            continue
        sel = replace(selection, traces=(trace,))
        try:
            results[trace.trace_id] = _one(
                sel,
                analyses=analyses,
                output_dir=output_dir,
                export_samples=export_samples,
                events=events,
                auto_exclude=auto_exclude,
                on_progress=on_progress,
                cancellation=cancellation,
            )
        except Exception as e:
            results[trace.trace_id] = failed_result(
                trace,
                str(e),
                getattr(e, "result_directory", output_dir),
                "cancelled" if isinstance(e, Cancelled) else "failed",
            )
        batch_reporter.emit(
            "batch",
            files_completed=sum(r.status in {"complete", "partial"} for r in results.values()),
            files_failed=sum(r.status == "failed" for r in results.values()),
            files_total=len(selection.traces),
            rows_processed=sum(r.manifest.get("rows_processed", 0) for r in results.values()),
        )
    batch_reporter.emit(
        "completion",
        "cancelled"
        if cancellation and cancellation.cancelled
        else "failed"
        if results and all(r.status == "failed" for r in results.values())
        else "partial"
        if any(r.status in {"partial", "failed"} for r in results.values())
        else "complete",
        files_completed=sum(r.status in {"complete", "partial"} for r in results.values()),
        files_failed=sum(r.status == "failed" for r in results.values()),
        files_total=len(selection.traces),
        rows_processed=sum(r.manifest.get("rows_processed", 0) for r in results.values()),
    )
    return BatchResult(results)


def failed_result(trace, reason, output_dir=None, status="failed"):
    """Construct an explicit per-trace failure outcome with no successful scientific tables.

    Parameters
    ----------
    trace
        BatteryTrace source handle; raw signals are never rewritten.
    reason
        Required human-readable scientific/protocol justification retained in provenance.
    output_dir
        Parent directory for a new result-ID subdirectory; manifest is committed last.
    status
        Bounded operation-specific value described by the function contract and API reference."""
    empty = TableView(lambda: iter([]))
    manifest = {**trace.manifest(), "status": status, "failure": reason}
    return AnalysisResult(
        trace,
        "",
        "",
        "",
        status,
        manifest,
        Path(output_dir or trace.cache_dir),
        empty,
        empty,
        empty,
        empty,
        empty,
        empty,
    )


def _worker(payload):
    # Importable worker, only paths/config/small metadata cross the spawn boundary.
    """Reconstruct one trace in a Windows-spawn child using serializable input and a shared cancellation event; return only small result metadata."""
    manifest, sel, analyses, output, export, events, auto, shared_cancel, progress_queue = payload
    token = CancellationToken(shared_cancel)
    trace = open_trace(
        manifest["source_path"],
        metadata=manifest["metadata"],
        read_options=__import__("gregorithm.io", fromlist=["ReadOptions"]).ReadOptions(
            **manifest["read_options"]
        ),
        cache_dir=manifest.get("cache_dir"),
    )
    trace.trace_id = manifest["trace_id"]
    trace.source_revision = manifest["source_revision"]
    trace.source_fingerprint = manifest["fingerprint"]
    trace._mode_corrections = tuple(manifest.get("step_corrections", ()))
    selection = TraceSelection(
        (trace,),
        tuple(sel["clauses"]),
        None if sel["columns"] is None else tuple(sel["columns"]),
        sel["head"],
        AnalysisConfig.from_dict(sel["config"]),
    )
    try:

        def send_progress(event):
            """Forward serializable worker progress to the coordinating process."""
            from queue import Full

            while not token.cancelled:
                try:
                    progress_queue.put(event.to_dict(), timeout=0.1)
                    break
                except Full:
                    continue

        r = _one(
            selection,
            analyses=analyses,
            output_dir=output,
            export_samples=export,
            events=EventSelection.from_dict(events) if events else None,
            auto_exclude=auto,
            cancellation=token,
            on_progress=send_progress,
        )
        return str(r.directory)
    except Exception as e:
        return {
            "failed": str(e),
            "status": "cancelled" if isinstance(e, Cancelled) else "failed",
            "directory": str(getattr(e, "result_directory", output or trace.cache_dir)),
        }


def reopen_result(directory, trace=None):
    """Reopen a completed local result directory, verify its source, and restore persisted or reproducible lazy sample views. Example: result = reopen_result("results/result-id"). Failure/cancellation manifests are not completed results.

    Parameters
    ----------
    directory
        Bounded operation-specific value described by the function contract and API reference.
    trace
        BatteryTrace source handle; raw signals are never rewritten."""
    p = Path(directory)
    m = json.loads((p / "manifest.json").read_text(encoding="utf8"))
    if trace is None:
        trace = open_trace(
            m["source_path"],
            metadata=m["metadata"],
            read_options=__import__("gregorithm.io", fromlist=["ReadOptions"]).ReadOptions(
                **m["read_options"]
            ),
            cache_dir=m.get("cache_dir"),
        )
        trace.trace_id = m["trace_id"]
        trace.source_revision = m["source_revision"]
        trace.source_fingerprint = m["fingerprint"]
        trace._mode_corrections = tuple(m.get("step_corrections", ()))
        trace.verify()
    tables = {k: parquet_view(p / name, m) for k, name in m["files"].items()}
    empty = TableView(lambda: iter([]), m)
    if "samples" not in tables and m["status"] in {"complete", "partial"}:
        current_runtime = runtime_provenance()
        if (
            m.get("source_file_sha256")
            and m["source_file_sha256"] != current_runtime["source_file_sha256"]
        ):
            from .diagnostics import RevisionMismatchError

            raise RevisionMismatchError(
                "Lazy reconstruction requires the original library code; use persisted samples or the original environment"
            )
        for dependency in ("numpy", "pyarrow", "numba"):
            saved = m.get("runtime_versions", {})
            if (
                dependency in saved
                and saved[dependency] != current_runtime["runtime_versions"][dependency]
            ):
                from .diagnostics import RevisionMismatchError

                raise RevisionMismatchError(
                    "Lazy reconstruction requires original numerical dependency versions"
                )
        cfg = AnalysisConfig.from_dict(m["config"])
        event_rows = (
            tuple(
                {k: v for k, v in row.items() if k != "result_id"}
                for row in pq.read_table(p / m["files"]["events"]).to_pylist()
            )
            if "events" in m["files"]
            else ()
        )
        event_selection = EventSelection(
            event_rows,
            m.get("event_revision", "initial"),
            m.get("event_parent_revision"),
            metadata=m.get("event_metadata", {}),
        )
        trace._event_selection = event_selection
        idx = trace.ensure_index(
            cfg, event_selection if event_selection.accepted_ranges() else None
        )

        def sample_batches():
            """Reconstruct original-source sample analysis from the captured completed manifest."""
            for block in integrated_batches(trace, idx, cfg, event_selection):
                yield _event_columns(block, event_selection).append_column(
                    "result_id", pa.array([m["result_id"]] * block.num_rows)
                )

        selected = m["selection"]
        view = TableView(
            sample_batches,
            m,
            tuple(selected["clauses"]),
            None,
            selected["head"],
            cfg.managed_memory_bytes,
        )
        if cfg.mass_g is not None or cfg.area_cm2 is not None:
            view = normalize(view, mass_g=cfg.mass_g, area_cm2=cfg.area_cm2)
        ref = m.get("c_rate_reference")
        if ref is not None:
            original = view

            def rate_batches():
                """Append reconstructed signed C-rate using the saved reference choice."""
                for block in original.iter_batches():
                    capacity = ref["value_mAh"]
                    yield block.append_column(
                        "c_rate_h_inv",
                        pc.divide(
                            block.column("current_mA"), pa.scalar(capacity, type=pa.float64())
                        ),
                    )

            view = TableView(rate_batches, m, memory_bytes=cfg.managed_memory_bytes)
        if selected["columns"] is not None:
            view = view.select(columns=selected["columns"])
        tables["samples"] = view
    return AnalysisResult(
        trace,
        m["result_id"],
        m["analysis_id"],
        m.get("assignment_revision", ""),
        m["status"],
        m,
        p,
        tables.get("samples", empty),
        tables.get("metrics", empty),
        tables.get("curves", empty),
        tables.get("events", empty),
        tables.get("segments", empty),
        tables.get("diagnostics", empty),
    )


def _parallel(selection, analyses, output, export, events, auto, progress, token):
    """Coordinate independent trace tasks, divide the managed budget, forward bounded progress, and retain per-file failure/cancellation outcomes."""
    import multiprocessing
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    from queue import Empty

    cfg = selection.config
    per_worker = cfg.managed_memory_bytes // cfg.workers
    if per_worker < 64 * 1024**2:
        raise ValueError("Insufficient global memory for requested worker count")
    ctx = multiprocessing.get_context("spawn")
    by = {}
    latest = {}
    reporter = Reporter(progress, uuid.uuid4().hex)
    with ctx.Manager() as manager:
        shared_cancel = manager.Event()
        messages = manager.Queue(maxsize=2)
        payloads = []
        for t in selection.traces:
            d = {
                **selection.to_dict(),
                "config": replace(cfg, workers=1, managed_memory_bytes=per_worker).to_dict(),
            }
            payloads.append(
                (
                    t.manifest(),
                    d,
                    tuple(analyses),
                    str(output) if output else None,
                    export,
                    events.select(trace_id=t.trace_id).to_dict() if events else None,
                    auto,
                    shared_cancel,
                    messages,
                )
            )
        with ProcessPoolExecutor(max_workers=cfg.workers, mp_context=ctx) as pool:
            futures = {pool.submit(_worker, p): t for p, t in zip(payloads, selection.traces)}
            pending = set(futures)
            while pending:
                if token and token.cancelled:
                    shared_cancel.set()
                    for f in pending:
                        f.cancel()
                while True:
                    try:
                        event = messages.get_nowait()
                    except Empty:
                        break
                    latest[event["trace_id"]] = event
                    # Worker completion is represented by the coordinator's file counters;
                    # callbacks and widgets never cross the process boundary.
                    reporter.emit(
                        event["phase"],
                        "running",
                        trace_id=event["trace_id"],
                        rows_read=event["rows_read"],
                        rows_selected=event["rows_selected"],
                        rows_processed=event["rows_processed"],
                        batches_completed=event["batches_completed"],
                        total=event["total"],
                        percent=event["percent"],
                    )
                done, pending = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
                for future in done:
                    t = futures[future]
                    if future.cancelled():
                        by[t.trace_id] = failed_result(
                            t, "Cancelled before start", output, "cancelled"
                        )
                        continue
                    try:
                        answer = future.result()
                        by[t.trace_id] = (
                            failed_result(
                                t,
                                answer["failed"],
                                answer.get("directory", output),
                                answer.get("status", "failed"),
                            )
                            if isinstance(answer, dict)
                            else reopen_result(answer, t)
                        )
                    except Exception as e:
                        by[t.trace_id] = failed_result(t, str(e), output)
                if done:
                    reporter.emit(
                        "batch",
                        rows_processed=sum(e["rows_processed"] for e in latest.values()),
                        files_completed=sum(
                            r.status in {"complete", "partial"} for r in by.values()
                        ),
                        files_failed=sum(r.status == "failed" for r in by.values()),
                        files_total=len(selection.traces),
                    )
    reporter.emit(
        "completion",
        "cancelled"
        if token and token.cancelled
        else "failed"
        if by and all(r.status == "failed" for r in by.values())
        else "partial"
        if any(r.status in {"partial", "failed"} for r in by.values())
        else "complete",
        rows_read=sum(e["rows_read"] for e in latest.values()),
        rows_selected=sum(e["rows_selected"] for e in latest.values()),
        batches_completed=sum(e["batches_completed"] for e in latest.values()),
        rows_processed=sum(e["rows_processed"] for e in latest.values()),
        files_completed=sum(r.status in {"complete", "partial"} for r in by.values()),
        files_failed=sum(r.status == "failed" for r in by.values()),
        files_total=len(selection.traces),
    )
    return BatchResult({t.trace_id: by[t.trace_id] for t in selection.traces})
