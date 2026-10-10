"""Disk-backed deterministic observed runs and assignment index."""

import sqlite3
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import digest
from .execution import CancellationToken, Reporter
from .io import atomic_parquet, source_batches

DIR = {1: "charge", -1: "discharge", 0: "rest", 2: "unknown"}


class _ClosingConnection(sqlite3.Connection):
    """Internal SQLite context that closes handles after transaction completion, including Windows rename safety."""

    def __exit__(self, exc_type, exc_value, traceback):
        """Commit or roll back the SQLite context and close its file handle even on failure, permitting Windows-safe replacement."""
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def assignment_config(config):
    """Extract scientific assignment parameters from resource/display-independent configuration for stable segmentation revision hashing.

    Parameters
    ----------
    config
        Validated immutable AnalysisConfig or DQDVConfig as appropriate; None uses operational defaults."""
    names = (
        "rest_threshold_mA",
        "min_samples",
        "min_duration_s",
        "cycle_order",
        "rest_assignment",
        "cc_window_s",
        "cc_absolute_mA",
        "cc_relative",
        "cv_band_V",
    )
    return {name: getattr(config, name) for name in names}


def validate_time_selection(index, clauses):
    """Reject an absolute-time selection matching multiple acquisition sections unless explicit section IDs disambiguate it. This prevents silently mixing repeated timestamps after resets.

    Parameters
    ----------
    index
        Disk-backed SegmentIndex captured for the scientific assignment revision.
    clauses
        Bounded operation-specific value described by the function contract and API reference."""
    from .diagnostics import AmbiguousTimeSelectionError
    from .io import expand_clauses

    with index.connect() as db:
        for leaf in expand_clauses(clauses):
            windows = [c["time_s"] for c in leaf if "time_s" in c]
            if not windows or any("time_section_id" in c for c in leaf):
                continue
            lo = max((w.get("min") for w in windows if w.get("min") is not None), default=-np.inf)
            hi = min((w.get("max") for w in windows if w.get("max") is not None), default=np.inf)
            sections = set()
            for r in db.execute(
                "select * from runs where t_end>=? and t_start<=? order by start", (lo, hi)
            ):
                keep = True
                for clause in leaf:
                    for key, value in clause.items():
                        if key in {"cycle_id", "half_cycle_id"}:
                            keep &= r[key] in (
                                value if isinstance(value, (list, tuple)) else [value]
                            )
                if keep:
                    sections.add(r["time_section_id"])
                if len(sections) > 1:
                    raise AmbiguousTimeSelectionError(
                        "Select time_section_id explicitly for a time window spanning acquisition sections"
                    )


def classify_current(i, threshold):
    """Map point current to charge(+1), discharge(-1), rest(0), or unknown(2). Rest comparison is inclusive; raw small currents are retained for integration.

    Parameters
    ----------
    i
        Matching point current batch in mA.
    threshold
        Bounded operation-specific value described by the function contract and API reference."""
    i = np.asarray(i, dtype=np.float64)
    return np.where(
        ~np.isfinite(i), 2, np.where(i > threshold, 1, np.where(i < -threshold, -1, 0))
    ).astype(np.int64)


@dataclass
class SegmentIndex:
    """Disk-backed observed-run and step assignments for one source/scientific revision; connections and iterators are bounded and Windows-safe."""

    path: Path
    revision: str
    rows: int
    trace_id: str
    source_revision: str = ""

    def connect(self):
        """Open a row-aware bounded-cache SQLite connection; the context closes it explicitly on exit."""
        c = sqlite3.connect(self.path, factory=_ClosingConnection)
        c.row_factory = sqlite3.Row
        return c

    def batches(self, table="runs", batch_rows=4096):
        """Yield bounded rows from the indexed segment/source table, retaining explicit scientific identity."""
        with self.connect() as c:
            cur = c.execute(f"select * from {table} order by start")
            while records := cur.fetchmany(batch_rows):
                rows = []
                for r in records:
                    row = {
                        **dict(r),
                        "trace_id": self.trace_id,
                        "source_revision": self.source_revision,
                        "assignment_revision": self.revision,
                        "direction": DIR[r["sign"]],
                    }
                    if table == "steps":
                        owner = c.execute(
                            "select half_cycle_id,cycle_id from runs "
                            "where start<=? and end>=? limit 1",
                            (r["start"], r["start"]),
                        ).fetchone()
                        row["half_cycle_id"] = owner[0] if owner else 0
                        row["cycle_id"] = owner[1] if owner else 0
                    rows.append(row)
                yield pa.RecordBatch.from_pylist(rows)

    def ranges(self, clauses):
        # Exact range pruning for assignment IDs. Time predicates are reapplied per sample.
        """Query indexed original sample ranges matching derived selection predicates without loading all source rows."""
        with self.connect() as c:
            for r in c.execute("select * from runs order by start"):
                keep = True
                for clause in clauses:
                    for k, v in clause.items():
                        if k == "sample_id" and isinstance(v, dict):
                            keep &= (v.get("min") is None or r["end"] >= v["min"]) and (
                                v.get("max") is None or r["start"] <= v["max"]
                            )
                        if k in {"cycle_id", "half_cycle_id", "time_section_id", "direction"}:
                            vals = v if isinstance(v, (list, tuple)) else [v]
                            value = DIR[r["sign"]] if k == "direction" else r[k]
                            keep &= value in vals
                if keep:
                    yield r["start"], r["end"]


def _mode(sign, count, imin, imax, vmin, vmax, config):
    """Infer CC/CV/rest/variable/unknown from bounded observation statistics; this does not certify the experimental protocol."""
    if sign == 0:
        return "rest", "observed_rest"
    if sign == 2 or count < 2 or not all(np.isfinite(x) for x in (imin, imax, vmin, vmax)):
        return "unknown", "insufficient_support"
    stable_i = imax - imin <= max(
        config.cc_absolute_mA, config.cc_relative * max(abs(imin), abs(imax))
    )
    stable_v = vmax - vmin <= config.cv_band_V
    if stable_i and not stable_v:
        return "CC", "current_band"
    if stable_v and not stable_i:
        return "CV", "voltage_band"
    if stable_i and stable_v:
        return "unknown", "both_signals_stable"
    return "variable", "both_signals_vary"


def build_index(trace, config, events=None, cancellation=None, on_progress=None):
    """Scan original samples with persistent cross-batch state into SQLite runs/steps and deterministic cycle assignments. Confirmation requires both configured support limits; end-of-file does not shorten them.

    Parameters
    ----------
    trace
        BatteryTrace source handle; raw signals are never rewritten.
    config
        Validated immutable AnalysisConfig or DQDVConfig as appropriate; None uses operational defaults.
    events
        Explicit EventSelection with source identity and decision/boundary revision.
    cancellation
        CancellationToken checked at cooperative boundaries; None creates a local token.
    on_progress
        Callable receiving serializable ProgressEvent records during execution."""
    trace.verify()
    token = cancellation or CancellationToken()
    directory = trace.cache_dir
    directory.mkdir(parents=True, exist_ok=True)
    revision = digest(
        {
            "source": trace.source_revision,
            "algorithm": "segmentation-3",
            "config": assignment_config(config),
            "schema": str(trace.schema),
            "read_options": {
                name: getattr(trace.read_options, name)
                for name in ("column_mapping", "delimiter", "encoding", "decimal")
            },
            "events": events.to_dict() if events else None,
        }
    )
    path = directory / (revision + ".sqlite")
    if path.exists():
        with sqlite3.connect(path) as c:
            try:
                rows = c.execute("select rows from completed").fetchone()[0]
            except sqlite3.Error:
                rows = None
        if rows is not None:
            return SegmentIndex(path, revision, rows, trace.trace_id, trace.source_revision)
    tmp = path.with_suffix(".working.sqlite")
    if tmp.exists():
        tmp.unlink()
    db = sqlite3.connect(tmp)
    db.execute("pragma cache_size=-8192")
    db.execute(
        "create table runs (run_id integer primary key,start integer,end integer,t_start real,t_end real,time_section_id integer,sign integer,n integer,mode text,reason text,half_cycle_id integer default 0,cycle_id integer default 0,main_sign integer default 0,confirmed integer default 0,excluded integer default 0)"
    )
    db.execute(
        "create table steps (step_id integer primary key,start integer,end integer,t_start real,t_end real,time_section_id integer,sign integer,mode text,reason text,i_level real)"
    )
    split_points = {p for a, z in (events.accepted_ranges() if events else ()) for p in (a, z + 1)}
    run = None
    window = None
    step = None
    section = 0
    previous_t = None
    row = 0
    run_id = 0
    step_id = 0
    reporter = Reporter(on_progress, uuid.uuid4().hex, trace.trace_id)

    def close_run():
        """Flush one observed action to the disk index without losing batch-boundary state."""
        nonlocal run_id, run
        if run is None:
            return
        run_id += 1
        mode, reason = _mode(
            run["sign"], run["n"], run["imin"], run["imax"], run["vmin"], run["vmax"], config
        )
        db.execute(
            "insert into runs(run_id,start,end,t_start,t_end,time_section_id,sign,n,mode,reason) values(?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                run["start"],
                run["end"],
                run["t0"],
                run["t1"],
                run["section"],
                run["sign"],
                run["n"],
                mode,
                reason,
            ),
        )
        run = None

    def close_step():
        """Flush one inferred step and original support to the disk index."""
        nonlocal step_id, step
        if step is None:
            return
        step_id += 1
        db.execute(
            "insert into steps values(?,?,?,?,?,?,?,?,?,?)",
            (
                step_id,
                step["start"],
                step["end"],
                step["t0"],
                step["t1"],
                step["section"],
                step["sign"],
                step["mode"],
                step["reason"],
                step["level"],
            ),
        )
        step = None

    def close_window():
        """Finalize one bounded CC/CV observation window and reset only its statistics."""
        nonlocal window, step
        if window is None:
            return
        mode, reason = _mode(
            window["sign"],
            window["n"],
            window["imin"],
            window["imax"],
            window["vmin"],
            window["vmax"],
            config,
        )
        level = (window["imin"] + window["imax"]) / 2 if window["sign"] in (-1, 1) else 0
        change = (
            step is None
            or step["sign"] != window["sign"]
            or step["section"] != window["section"]
            or step["mode"] != mode
        )
        if step is not None and mode == "CC":
            change |= abs(level - step["level"]) > max(
                config.cc_absolute_mA, config.cc_relative * abs(step["level"])
            )
        if change:
            close_step()
            step = {**window, "mode": mode, "reason": reason, "level": level}
        else:
            step["end"] = window["end"]
            step["t1"] = window["t1"]
        window = None

    def stat(ints, floats):
        """Update bounded observation statistics from one original sample."""
        return {
            "start": int(ints[0]),
            "end": int(ints[1]),
            "section": int(ints[2]),
            "sign": int(ints[3]),
            "n": int(ints[4]),
            "t0": float(floats[0]),
            "t1": float(floats[1]),
            "imin": float(floats[2]),
            "imax": float(floats[3]),
            "vmin": float(floats[4]),
            "vmax": float(floats[5]),
        }

    try:
        opts = replace(
            trace.read_options,
            batch_rows=min(config.batch_rows, max(1, config.managed_memory_bytes // 1024)),
            max_batch_bytes=min(
                trace.read_options.max_batch_bytes, config.managed_memory_bytes // 16
            ),
        )
        from ._segment_scan import backend as scan_backend

        scanner = scan_backend(config.backend)
        ri = np.zeros(5, dtype=np.int64)
        ri[0] = -1
        wi = ri.copy()
        rf = np.zeros(6)
        wf = rf.copy()
        previous_t = np.nan
        splits = np.array(sorted(split_points), dtype=np.int64)
        for nb, b in enumerate(
            source_batches(trace.path, opts, ("time_s", "current_mA", "voltage_V"))
        ):
            token.check()
            t, i, v = [
                b.column(b.schema.get_field_index(n)).to_numpy(zero_copy_only=False).astype(float)
                for n in ("time_s", "current_mA", "voltage_V")
            ]
            signs = classify_current(i, config.rest_threshold_mA)
            runs_i, runs_f, wins_i, wins_f, row, section, previous_t = scanner(
                t, i, v, signs, row, section, previous_t, ri, rf, wi, wf, config.cc_window_s, splits
            )
            for ints, floats in zip(runs_i, runs_f):
                run = stat(ints, floats)
                close_run()
            for ints, floats in zip(wins_i, wins_f):
                window = stat(ints, floats)
                close_window()
            if nb % 16 == 0:
                db.commit()
            reporter.emit(
                "segmentation", rows_read=row, rows_processed=row, batches_completed=nb + 1
            )
        if ri[0] >= 0:
            run = stat(ri, rf)
        if wi[0] >= 0:
            window = stat(wi, wf)
        close_run()
        close_window()
        close_step()
        db.commit()
        excluded = events.accepted_ranges() if events else ()
        half = cycle = 0
        main = 0
        last_section = -1
        first_sign = 1 if config.cycle_order == "charge_then_discharge" else -1
        db.row_factory = sqlite3.Row
        for r in db.execute("select * from runs order by start"):
            token.check()
            if r["time_section_id"] != last_section:
                main = 0
                local_half = local_cycle = 0
                last_section = r["time_section_id"]
            exc = any(a <= r["start"] and r["end"] <= z for a, z in excluded)
            s = r["sign"]
            confirmed = False
            if (
                not exc
                and s in (-1, 1)
                and s != main
                and r["n"] >= config.min_samples
                and r["t_end"] is not None
                and r["t_start"] is not None
                and r["t_end"] - r["t_start"] >= config.min_duration_s
            ):
                half += 1
                local_half = half
                if main == 0 or s == first_sign:
                    cycle += 1
                local_cycle = cycle
                main = s
                confirmed = True
            assignment = (local_half, local_cycle, main)
            if s == 0 and config.rest_assignment == "unassigned":
                assignment = (0, 0, 0)
            db.execute(
                "update runs set half_cycle_id=?,cycle_id=?,main_sign=?,confirmed=?,excluded=? where run_id=?",
                (*assignment, int(confirmed), int(exc), r["run_id"]),
            )
        db.execute("create index run_ranges on runs(start,end)")
        db.execute("create index step_ranges on steps(start,end)")
        db.execute("create table completed(rows integer)")
        db.execute("insert into completed values(?)", (row,))
        db.commit()
        trace.verify()
        db.close()
        tmp.replace(path)
        idx = SegmentIndex(path, revision, row, trace.trace_id, trace.source_revision)
        # Sidecar remains disk-backed even for pathological numbers of tiny actions.
        atomic_parquet(
            idx.batches(),
            directory / (revision + ".segments.parquet"),
            {**trace.manifest(), "assignment_revision": revision},
        )
        reporter.emit("segmentation", "complete", rows_read=row, rows_processed=row)
        return idx
    except BaseException:
        db.close()
        if tmp.exists():
            tmp.unlink()
        raise


def annotated_batches(trace, index, config, columns=None, clauses=()):
    """Join bounded source batches with disk-backed run/step assignments and preserve original sample IDs. Predicate ranges may prune Parquet groups; numerical state does not live in the decoder.

    Parameters
    ----------
    trace
        BatteryTrace source handle; raw signals are never rewritten.
    index
        Disk-backed SegmentIndex captured for the scientific assignment revision.
    config
        Validated immutable AnalysisConfig or DQDVConfig as appropriate; None uses operational defaults.
    columns
        Explicit column projection or normalization input names, according to this operation.
    clauses
        Bounded operation-specific value described by the function contract and API reference."""
    trace.verify()
    required = {k for c in clauses for k in c}
    names = (
        None
        if columns is None
        else tuple(
            (set(columns) | required | {"time_s", "current_mA", "voltage_V"})
            & set(trace.schema.names)
        )
    )
    opts = replace(
        trace.read_options,
        batch_rows=config.batch_rows,
        max_batch_bytes=min(trace.read_options.max_batch_bytes, config.managed_memory_bytes // 16),
    )
    groups = None
    group_offsets = None
    if trace.path.suffix.lower() == ".parquet" and clauses:
        f = pq.ParquetFile(trace.path)
        ranges = iter(index.ranges(clauses))
        interval = next(ranges, None)
        offset = 0
        groups = []
        group_offsets = []
        for g in range(f.num_row_groups):
            n = f.metadata.row_group(g).num_rows
            while interval and interval[1] < offset:
                interval = next(ranges, None)
            if interval and interval[0] < offset + n:
                groups.append(g)
                group_offsets.append(offset)
            offset += n
        f.close()
        if not groups:
            return
    with index.connect() as db:
        # Only selected row groups are read. Each original offset is retained.
        requests = [(None, 0)] if groups is None else list(zip(groups, group_offsets))
        for group, offset in requests:
            rc = db.execute("select * from runs where end>=? order by start", (offset,))
            r = rc.fetchone()
            sc = db.execute("select * from steps where end>=? order by start", (offset,))
            s = sc.fetchone()
            for b in source_batches(trace.path, opts, names, None if group is None else [group]):
                n = b.num_rows
                ids = np.arange(offset, offset + n, dtype=np.int64)
                fields = {
                    k: np.empty(n, dtype=np.int64)
                    for k in (
                        "time_section_id",
                        "half_cycle_id",
                        "cycle_id",
                        "main_sign",
                        "observed_sign",
                        "excluded",
                        "confirmed",
                        "step_id",
                    )
                }
                modes = []
                reasons = []
                roles = []
                j = 0
                while j < n:
                    while r and offset + j > r["end"]:
                        r = rc.fetchone()
                    stop = min(n, r["end"] - offset + 1)
                    for k in (
                        "time_section_id",
                        "half_cycle_id",
                        "cycle_id",
                        "main_sign",
                        "excluded",
                        "confirmed",
                    ):
                        fields[k][j:stop] = r[k]
                    fields["observed_sign"][j:stop] = r["sign"]
                    j = stop
                j = 0
                while j < n:
                    while s and offset + j > s["end"]:
                        s = sc.fetchone()
                    stop = min(n, s["end"] - offset + 1)
                    fields["step_id"][j:stop] = s["step_id"]
                    modes.extend([s["mode"]] * (stop - j))
                    reasons.extend([s["reason"]] * (stop - j))
                    roles.extend([s["role"] if "role" in s.keys() else "ordinary"] * (stop - j))
                    j = stop
                for k, a in fields.items():
                    b = b.append_column(k, pa.array(a))
                b = (
                    b.append_column("sample_id", pa.array(ids))
                    .append_column("direction", pa.array([DIR[x] for x in fields["observed_sign"]]))
                    .append_column("mode", pa.array(modes))
                    .append_column("mode_reason", pa.array(reasons))
                    .append_column("step_role", pa.array(roles))
                    .append_column("trace_id", pa.array([trace.trace_id] * n))
                    .append_column("source_revision", pa.array([trace.source_revision] * n))
                )
                yield b
                offset += n
    trace.verify()
