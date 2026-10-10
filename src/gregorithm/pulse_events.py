"""Bounded discovery and protocol-profile assessment of current-pulse events."""

from __future__ import annotations

import math
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import asdict, dataclass, field, replace
from numbers import Integral
from pathlib import Path

import numpy as np
import pyarrow as pa

from ._result_tables import bounded_frame
from .diagnostics import BudgetExceeded, SourceChangedError
from .execution import CancellationToken, MemoryBudget
from .io import atomic_parquet, fingerprint
from .point_calculations import _effective_batches


@dataclass(frozen=True)
class PulseEventConfig:
    """Configure broad pulse discovery without asserting a diagnostic protocol.

    A supported broad candidate has immediately adjacent measured rest runs and at
    least ``min_real_samples`` real active-current measurements spanning
    ``min_real_span_s``. Incomplete opportunities remain visible as unresolved.
    """

    rest_threshold_mA: float = 0.0001
    min_real_samples: int = 2
    min_real_span_s: float = 0.1
    missing_span_s: float = 10.0
    pulse_direction: str = "both"

    def __post_init__(self):
        """Reject invalid operational settings before opening the source."""
        if (
            isinstance(self.min_real_samples, (bool, np.bool_))
            or not isinstance(self.min_real_samples, Integral)
            or self.min_real_samples < 2
        ):
            raise ValueError("min_real_samples must be an integer >= 2")
        object.__setattr__(self, "min_real_samples", int(self.min_real_samples))
        for name in ("rest_threshold_mA", "min_real_span_s", "missing_span_s"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, float(value))
        if self.pulse_direction not in {"charge", "discharge", "both"}:
            raise ValueError("pulse_direction must be charge, discharge or both")


@dataclass(frozen=True)
class PulseProfile:
    """Known rest-pulse-rest durations used to assess discovered opportunities.

    Durations and tolerances are seconds. The initial pulse tolerance is 5 s and
    each rest tolerance is 2 s. A nominal zero rest may be absent. Current
    stability uses all real pulse samples and has no nominal-current constraint.
    """

    rest_before_s: float
    pulse_s: float
    rest_after_s: float
    rest_before_tolerance_s: float = 2.0
    pulse_tolerance_s: float = 5.0
    rest_after_tolerance_s: float = 2.0
    current_relative_range: float = 0.05

    def __post_init__(self):
        """Require finite nonnegative protocol durations and tolerances."""
        for name, value in asdict(self).items():
            if isinstance(value, (bool, np.bool_)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, float(value))


_EVENT_SCHEMA = pa.schema(
    [
        ("event", pa.int64()),
        ("direction", pa.string()),
        ("relation_to_previous_active", pa.string()),
        ("t0_s", pa.float64()),
        ("t1_s", pa.float64()),
        ("t2_s", pa.float64()),
        ("t3_s", pa.float64()),
        ("observed_pulse_span_s", pa.float64()),
        ("pulse_duration_min_s", pa.float64()),
        ("pulse_duration_max_s", pa.float64()),
        ("rest_before_duration_min_s", pa.float64()),
        ("rest_before_duration_max_s", pa.float64()),
        ("rest_after_duration_min_s", pa.float64()),
        ("rest_after_duration_max_s", pa.float64()),
        ("n_active_samples", pa.int64()),
        ("n_real_active_current", pa.int64()),
        ("has_missing_active_current", pa.bool_()),
        ("current_min_mA", pa.float64()),
        ("current_max_mA", pa.float64()),
        ("current_relative_range", pa.float64()),
        ("candidate_status", pa.string()),
        ("candidate_reason", pa.string()),
        ("profile_status", pa.string()),
        ("profile_reason", pa.string()),
        ("decision", pa.string()),
        ("decision_reason", pa.string()),
        ("diagnostic_cycle_mark", pa.bool_()),
    ]
)


def _duration_status(lo, hi, nominal, tolerance, *, present):
    """Apply the approved full-bracket containment rule to one profile step."""
    allowed_lo, allowed_hi = max(0.0, nominal - tolerance), nominal + tolerance
    if not present:
        return (
            ("matched", "zero-duration rest absent as allowed")
            if nominal == 0
            else (
                "unresolved",
                "required rest is absent",
            )
        )
    if lo is None or hi is None:
        return "unresolved", "duration boundary support is incomplete"
    if lo >= allowed_lo and hi <= allowed_hi:
        return "matched", "duration bracket is inside the configured range"
    if hi < allowed_lo or lo > allowed_hi:
        return "not_matched", "duration bracket is outside the configured range"
    return "unresolved", "duration bracket only partially overlaps the configured range"


def _profile_assessment(event, profile):
    """Assess durations and whole-pulse measured-current stability independently."""
    checks = [
        _duration_status(
            event["rest_before_duration_min_s"],
            event["rest_before_duration_max_s"],
            profile.rest_before_s,
            profile.rest_before_tolerance_s,
            present=event["t0_s"] is not None,
        ),
        _duration_status(
            event["pulse_duration_min_s"],
            event["pulse_duration_max_s"],
            profile.pulse_s,
            profile.pulse_tolerance_s,
            present=True,
        ),
        _duration_status(
            event["rest_after_duration_min_s"],
            event["rest_after_duration_max_s"],
            profile.rest_after_s,
            profile.rest_after_tolerance_s,
            present=event["t3_s"] is not None,
        ),
    ]
    if event["has_missing_active_current"]:
        checks.append(("unresolved", "missing active current prevents stability confirmation"))
    elif event["current_relative_range"] is None:
        checks.append(("unresolved", "whole-pulse current range is unavailable"))
    elif event["current_relative_range"] <= profile.current_relative_range:
        checks.append(("matched", "whole-pulse current range is within the configured limit"))
    else:
        checks.append(("not_matched", "whole-pulse current range exceeds the configured limit"))
    status = (
        "not_matched"
        if any(value == "not_matched" for value, _ in checks)
        else "unresolved"
        if any(value == "unresolved" for value, _ in checks)
        else "matched"
    )
    return status, "; ".join(
        reason for value, reason in checks if value != "matched"
    ) or "all criteria matched"


def _run_rows(source, config, token):
    """Yield compact rest/active/unknown runs from bounded effective-current batches."""
    time_column = source.report["time_column"]
    reader = _effective_batches(
        source,
        [time_column, "current_mA"],
        config.missing_span_s,
        token,
        rest_threshold_mA=config.rest_threshold_mA,
    )
    run, previous_time = None, None
    try:
        for times, raw, effective, _, estimated, _ in reader:
            for t, measured, value, is_estimated in zip(
                times, raw, effective, estimated, strict=True
            ):
                token.check()
                real = math.isfinite(measured)
                usable = math.isfinite(value)
                if not usable:
                    state = "unknown"
                elif abs(value) <= config.rest_threshold_mA:
                    state = "rest"
                else:
                    state = "charge" if value > 0 else "discharge"
                if run is None or state != run["state"]:
                    if run is not None:
                        run["following_s"] = float(t)
                        yield run
                    run = dict(
                        state=state,
                        start_s=float(t),
                        end_s=float(t),
                        preceding_s=previous_time,
                        following_s=None,
                        n_samples=0,
                        n_real=0,
                        missing=False,
                        minimum=None,
                        maximum=None,
                    )
                run["end_s"] = float(t)
                run["n_samples"] += 1
                run["missing"] |= bool(is_estimated) or not real
                if real:
                    run["n_real"] += 1
                    run["minimum"] = (
                        measured if run["minimum"] is None else min(run["minimum"], measured)
                    )
                    run["maximum"] = (
                        measured if run["maximum"] is None else max(run["maximum"], measured)
                    )
                previous_time = float(t)
        if run is not None:
            yield run
    finally:
        reader.close()


def _event_from_runs(number, before, pulse, after, prior_active, config, profile):
    """Create one readable event record without a row identifier."""
    has_before = before is not None and before["state"] == "rest"
    has_after = after is not None and after["state"] == "rest"
    t0 = before["end_s"] if has_before else None
    t3 = after["start_s"] if has_after else None
    supported = (
        has_before
        and has_after
        and pulse["n_real"] >= config.min_real_samples
        and pulse["end_s"] - pulse["start_s"] >= config.min_real_span_s
    )
    reasons = []
    if not has_before:
        reasons.append("preceding rest absent")
    if not has_after:
        reasons.append("following rest absent")
    if pulse["n_real"] < config.min_real_samples:
        reasons.append("insufficient real active-current samples")
    if pulse["end_s"] - pulse["start_s"] < config.min_real_span_s:
        reasons.append("real active-current span is too short")
    scale = max(abs(pulse["minimum"] or 0.0), abs(pulse["maximum"] or 0.0))
    relative_range = (
        None
        if scale == 0 or pulse["minimum"] is None
        else (pulse["maximum"] - pulse["minimum"]) / scale
    )
    event = dict(
        event=number,
        direction=pulse["state"],
        relation_to_previous_active=(
            "unknown"
            if prior_active is None
            else "same"
            if prior_active == pulse["state"]
            else "opposite"
        ),
        t0_s=t0,
        t1_s=pulse["start_s"],
        t2_s=pulse["end_s"],
        t3_s=t3,
        observed_pulse_span_s=pulse["end_s"] - pulse["start_s"],
        pulse_duration_min_s=pulse["end_s"] - pulse["start_s"],
        pulse_duration_max_s=None if t0 is None or t3 is None else t3 - t0,
        rest_before_duration_min_s=(
            before["end_s"] - before["start_s"] if has_before and not before["missing"] else None
        ),
        rest_before_duration_max_s=(
            None
            if not has_before or before["missing"] or before["preceding_s"] is None
            else pulse["start_s"] - before["preceding_s"]
        ),
        rest_after_duration_min_s=(
            after["end_s"] - after["start_s"] if has_after and not after["missing"] else None
        ),
        rest_after_duration_max_s=(
            None
            if not has_after or after["missing"] or after["following_s"] is None
            else after["following_s"] - pulse["end_s"]
        ),
        n_active_samples=pulse["n_samples"],
        n_real_active_current=pulse["n_real"],
        has_missing_active_current=pulse["missing"],
        current_min_mA=pulse["minimum"],
        current_max_mA=pulse["maximum"],
        current_relative_range=relative_range,
        candidate_status="supported" if supported else "unresolved",
        candidate_reason="minimum measured support and adjacent rests available"
        if supported
        else "; ".join(reasons),
        profile_status="not_assessed",
        profile_reason="no pulse profile supplied",
        decision="unreviewed",
        decision_reason=None,
        diagnostic_cycle_mark=False,
    )
    if profile is not None:
        event["profile_status"], event["profile_reason"] = _profile_assessment(event, profile)
    return event


@dataclass(frozen=True)
class PulseEventResult:
    """Disk-backed event candidates with immutable human decisions layered on top."""

    source: object = field(repr=False)
    path: Path = field(repr=False)
    report: dict
    _workspace: object = field(repr=False)
    _budget: object = field(repr=False)
    _decisions: tuple = ()
    _statuses: tuple | None = None

    def statuses(self, values):
        """Select candidate statuses without rerunning discovery."""
        selected = (values,) if isinstance(values, str) else tuple(values)
        if any(value not in {"supported", "unresolved"} for value in selected):
            raise ValueError("Unknown candidate status")
        return replace(self, _statuses=selected)

    def decide(self, events, decision, *, reason, diagnostic_cycle_mark=False):
        """Return a decision view; discovery itself never accepts or marks an event.

        ``events`` contains readable event numbers from this result. Marking is a
        cycle-interpretation request only; applying it to half-cycle counting is a
        later explicit operation and never removes physical Q/E.
        """
        if decision not in {"accepted", "rejected", "unreviewed"}:
            raise ValueError("decision must be accepted, rejected or unreviewed")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("A nonempty decision reason is required")
        if diagnostic_cycle_mark and decision != "accepted":
            raise ValueError("Only an accepted event can be marked for cycle interpretation")
        numbers = (
            (events,)
            if isinstance(events, Integral) and not isinstance(events, bool)
            else tuple(events)
        )
        numbers = tuple(int(value) for value in numbers)
        with closing(sqlite3.connect(self.path)) as db:
            found = (
                {
                    row[0]
                    for row in db.execute(
                        "SELECT event FROM events WHERE event IN ("
                        + ",".join("?" for _ in numbers)
                        + ")",
                        numbers,
                    )
                }
                if numbers
                else set()
            )
        if found != set(numbers):
            raise KeyError(set(numbers) - found)
        updates = dict(self._decisions)
        for number in numbers:
            updates[number] = (decision, reason.strip(), bool(diagnostic_cycle_mark))
        return replace(self, _decisions=tuple(sorted(updates.items())))

    def iter_batches(self, batch_rows=512):
        """Yield bounded candidate summaries in natural event order."""
        if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows < 1:
            raise ValueError("Positive batch_rows required")
        rows = min(batch_rows, (self._budget.limit - self._budget.used - 2 * 1024**2) // 4096)
        if rows < 1:
            raise BudgetExceeded("Insufficient event-summary reader budget")
        reservation = 2 * 1024**2 + rows * 4096
        self._budget.reserve(reservation)
        self._workspace.active_readers = getattr(self._workspace, "active_readers", 0) + 1
        decisions = dict(self._decisions)
        try:
            if fingerprint(self.source.path) != self.report["prepared_source_metadata"]:
                raise SourceChangedError("Prepared source changed since event discovery")
            where = (
                ""
                if self._statuses is None
                else " WHERE candidate_status IN (" + ",".join("?" for _ in self._statuses) + ")"
            )
            with closing(sqlite3.connect(self.path)) as db:
                cursor = db.execute(
                    "SELECT "
                    + ",".join(_EVENT_SCHEMA.names)
                    + " FROM events"
                    + where
                    + " ORDER BY event",
                    self._statuses or (),
                )
                emitted = False
                while values := cursor.fetchmany(rows):
                    records = []
                    for values_row in values:
                        record = dict(zip(_EVENT_SCHEMA.names, values_row, strict=True))
                        record["has_missing_active_current"] = bool(
                            record["has_missing_active_current"]
                        )
                        record["diagnostic_cycle_mark"] = bool(record["diagnostic_cycle_mark"])
                        if record["event"] in decisions:
                            (
                                record["decision"],
                                record["decision_reason"],
                                record["diagnostic_cycle_mark"],
                            ) = decisions[record["event"]]
                        records.append(record)
                    emitted = True
                    yield pa.RecordBatch.from_pylist(records, schema=_EVENT_SCHEMA)
                if not emitted:
                    yield pa.RecordBatch.from_pylist([], schema=_EVENT_SCHEMA)
        finally:
            self._workspace.active_readers -= 1
            self._budget.release(reservation)

    def table(self, max_rows=100000):
        """Return a guarded pandas summary suitable for notebook review."""
        return bounded_frame(self, max_rows)

    def plot(self, *, y="observed_pulse_span_s", max_points=250000, mode="auto"):
        """Plot discovered opportunities by event number and current direction."""
        from .prepared_plotting import plot_compact_result

        return plot_compact_result(
            self,
            x="event",
            y=y,
            series_column="direction",
            series_values=("charge", "discharge"),
            max_points=max_points,
            mode=mode,
        )

    def plot_progressive(self, *, y="observed_pulse_span_s", max_points=250000, mode="auto"):
        """Create an idle progressive pulse-event plot session."""
        from .prepared_plotting import plot_compact_result

        return plot_compact_result(
            self,
            x="event",
            y=y,
            series_column="direction",
            series_values=("charge", "discharge"),
            progressive=True,
            max_points=max_points,
            mode=mode,
        )

    def write_parquet(self, path):
        """Stream the selected event summary and its single metadata report."""
        if Path(path).exists():
            raise FileExistsError("Choose a new event-summary export path")
        return atomic_parquet(
            self.iter_batches(),
            path,
            manifest={**self.report, "selected_candidate_statuses": self._statuses},
        )

    def dcir(self, delay_s, **settings):
        """Measure one requested DCIR delay for every event in this selected view."""
        from .pulse_dcir import measure_pulse_dcir

        return measure_pulse_dcir(self.source, delay_s, events=self, **settings)

    def diagnostic_windows(self):
        """Return active time ranges explicitly accepted for cycle-only interpretation."""
        ranges = []
        reader = self.iter_batches()
        try:
            for batch in reader:
                for row in batch.select(
                    ["t1_s", "t2_s", "decision", "diagnostic_cycle_mark"]
                ).to_pylist():
                    if row["decision"] == "accepted" and row["diagnostic_cycle_mark"]:
                        ranges.append((row["t1_s"], row["t2_s"]))
        finally:
            reader.close()
        return tuple(ranges)

    def capacity(self, **settings):
        """Apply accepted diagnostic marks to cycle counting while retaining all Q/E."""
        if "diagnostic_windows" in settings:
            raise ValueError("Use this result's accepted diagnostic marks, not separate windows")
        return self.source.capacity(
            diagnostic_windows=self.diagnostic_windows(),
            **settings,
        )

    def close(self):
        """Remove verified owned temporary storage after all iterators close."""
        if getattr(self._workspace, "active_readers", 0):
            raise ValueError("Close active event iterators first")
        if Path(self._workspace.name).resolve() != self.path.parent:
            raise ValueError("Event cleanup target changed")
        self._workspace.cleanup()


def detect_pulse_events(
    source,
    *,
    config=None,
    profile=None,
    memory_budget=None,
    cancellation=None,
    scratch_dir=None,
):
    """Discover bounded rest/current opportunities without accepting or measuring them.

    Use ``data.events()`` for defaults or pass a ``PulseProfile`` to compare known
    protocol durations. Detection creates one compact row per active run and never
    appends identifiers or labels to measurement rows.
    """
    cfg = config or PulseEventConfig()
    if not isinstance(cfg, PulseEventConfig):
        raise TypeError("config must be PulseEventConfig")
    if profile is not None and not isinstance(profile, PulseProfile):
        raise TypeError("profile must be PulseProfile")
    budget = memory_budget or MemoryBudget(2 * 1024**3)
    token = cancellation or CancellationToken()
    initial = fingerprint(source.path)
    cache = 2 * 1024**2
    reader_bytes = min(
        source.read_options.max_batch_bytes, (budget.limit - budget.used - cache) // 16
    )
    if reader_bytes < 64:
        raise BudgetExceeded("Insufficient shared event-discovery budget")
    reservation = reader_bytes * 16 + cache
    budget.reserve(reservation)
    workspace = None
    try:
        parent = Path(scratch_dir or source.path.parent / ".gregorithm-work").resolve()
        parent.mkdir(parents=True, exist_ok=True)
        workspace = tempfile.TemporaryDirectory(prefix="events-", dir=parent)
        folder = Path(workspace.name).resolve()
        if not folder.is_relative_to(parent):
            raise ValueError("Event workspace escaped its intended directory")
        path = folder / "events.sqlite"
        view = replace(
            source,
            read_options=replace(source.read_options, max_batch_bytes=reader_bytes),
        )
        runs = _run_rows(view, cfg, token)
        previous, current = None, next(runs, None)
        following = next(runs, None)
        prior_active, number = None, 0
        with closing(sqlite3.connect(path)) as db:
            db.execute("PRAGMA cache_size=-2048")
            db.execute("PRAGMA temp_store=FILE")
            db.execute(
                "CREATE TABLE events ("
                + ",".join(
                    field.name
                    + (
                        " TEXT"
                        if pa.types.is_string(field.type)
                        else " INTEGER"
                        if pa.types.is_integer(field.type) or pa.types.is_boolean(field.type)
                        else " REAL"
                    )
                    + (" PRIMARY KEY" if field.name == "event" else "")
                    for field in _EVENT_SCHEMA
                )
                + ")"
            )
            while current is not None:
                token.check()
                if current["state"] in {"charge", "discharge"}:
                    if cfg.pulse_direction == "both" or cfg.pulse_direction == current["state"]:
                        number += 1
                        event = _event_from_runs(
                            number, previous, current, following, prior_active, cfg, profile
                        )
                        db.execute(
                            "INSERT INTO events VALUES ("
                            + ",".join("?" for _ in _EVENT_SCHEMA)
                            + ")",
                            tuple(event[name] for name in _EVENT_SCHEMA.names),
                        )
                    prior_active = current["state"]
                previous, current = current, following
                following = next(runs, None)
            db.commit()
        if fingerprint(source.path) != initial:
            raise SourceChangedError("Prepared source changed during event discovery")
        report = dict(
            source_file=source.report["source_file"],
            prepared_source_metadata=initial,
            operation="pulse_event_discovery",
            settings=asdict(cfg),
            profile=None if profile is None else asdict(profile),
            events=number,
            generated_measurement_identifiers=False,
            appended_measurement_columns=False,
            discovery_accepts_events=False,
            discovery_measures_resistance=False,
        )
        return PulseEventResult(source, path, report, workspace, budget)
    except BaseException:
        if workspace is not None:
            workspace.cleanup()
        raise
    finally:
        budget.release(reservation)
