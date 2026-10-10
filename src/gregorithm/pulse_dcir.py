"""Signed DCIR measurement on pulse events using only real source rows."""

from __future__ import annotations

import math
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path

import numpy as np
import pyarrow as pa

from ._result_tables import bounded_frame
from .diagnostics import BudgetExceeded, SourceChangedError
from .execution import CancellationToken, MemoryBudget
from .io import atomic_parquet, fingerprint


@dataclass(frozen=True)
class ManualPulse:
    """Exact retained timestamps for last-rest, active start/end and optional next rest."""

    t0_s: float
    t1_s: float
    t2_s: float
    t3_s: float | None = None

    def __post_init__(self):
        """Require finite ordered boundaries; no boundary is snapped or rounded."""
        required = (self.t0_s, self.t1_s, self.t2_s)
        if any(isinstance(x, (bool, np.bool_)) or not math.isfinite(x) for x in required):
            raise ValueError("Manual pulse boundaries must be finite measured times")
        if not self.t0_s < self.t1_s <= self.t2_s:
            raise ValueError("Manual boundaries must satisfy t0 < t1 <= t2")
        if self.t3_s is not None and (
            isinstance(self.t3_s, (bool, np.bool_))
            or not math.isfinite(self.t3_s)
            or self.t3_s <= self.t2_s
        ):
            raise ValueError("Manual t3 must be finite and later than t2")


_SCHEMA = pa.schema(
    [
        ("event", pa.int64()),
        ("direction", pa.string()),
        ("event_context", pa.string()),
        ("t0_s", pa.float64()),
        ("t1_s", pa.float64()),
        ("t2_s", pa.float64()),
        ("t3_s", pa.float64()),
        ("requested_delay_s", pa.float64()),
        ("target_time_s", pa.float64()),
        ("actual_time_s", pa.float64()),
        ("actual_delay_s", pa.float64()),
        ("timing_error_s", pa.float64()),
        ("sampling_choice", pa.string()),
        ("baseline_current_mA", pa.float64()),
        ("baseline_voltage_V", pa.float64()),
        ("target_current_mA", pa.float64()),
        ("target_voltage_V", pa.float64()),
        ("delta_current_A", pa.float64()),
        ("delta_voltage_V", pa.float64()),
        ("resistance_ohm", pa.float64()),
        ("valid", pa.bool_()),
        ("quality", pa.string()),
        ("reason", pa.string()),
    ]
)


def _scalar(name, value, *, positive=False, optional=False):
    """Validate and normalize one scalar setting."""
    if value is None and optional:
        return None
    if isinstance(value, (bool, np.bool_, list, tuple, np.ndarray)):
        raise ValueError(f"{name} must be one finite scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be one finite scalar") from error
    if not math.isfinite(result) or result < 0 or positive and result <= 0:
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return result


def _event_records(events, manual, domain=None):
    """Yield selected discovered or manual compact event records."""
    if manual is not None:
        previous_end = -math.inf
        for number, pulse in enumerate(tuple(manual), 1):
            if not isinstance(pulse, ManualPulse):
                raise TypeError("manual entries must be ManualPulse objects")
            if pulse.t0_s <= previous_end:
                raise ValueError("Manual pulses must be ordered and nonoverlapping")
            previous_end = pulse.t3_s if pulse.t3_s is not None else pulse.t2_s
            yield {
                "event": number,
                "direction": "unknown",
                "event_context": "manual",
                "t0_s": float(pulse.t0_s),
                "t1_s": float(pulse.t1_s),
                "t2_s": float(pulse.t2_s),
                "t3_s": None if pulse.t3_s is None else float(pulse.t3_s),
            }
        return
    reader = events.iter_batches()
    try:
        for batch in reader:
            names = ["event", "direction", "t0_s", "t1_s", "t2_s", "t3_s"]
            for row in batch.select(names).to_pylist():
                if domain is not None and not domain[0] <= row["t1_s"] <= domain[1]:
                    continue
                yield {
                    **row,
                    "event_context": "complete" if row["t3_s"] is not None else "incomplete",
                }
    finally:
        reader.close()


def _blank(event, delay):
    """Build one result that remains explicit if support is unavailable."""
    return {
        **event,
        "requested_delay_s": delay,
        "target_time_s": None if event["t0_s"] is None else event["t0_s"] + delay,
        "actual_time_s": None,
        "actual_delay_s": None,
        "timing_error_s": None,
        "sampling_choice": None,
        "baseline_current_mA": None,
        "baseline_voltage_V": None,
        "target_current_mA": None,
        "target_voltage_V": None,
        "delta_current_A": None,
        "delta_voltage_V": None,
        "resistance_ohm": None,
        "valid": False,
        "quality": "unavailable",
        "reason": None,
    }


def _finish(record, state, max_error, minimum):
    """Select exact/excess/defect support and calculate signed V/A resistance."""
    if record["t0_s"] is None:
        record["reason"] = "preceding last-rest baseline is absent"
        return record
    required = 3 if record["t3_s"] is None else 4
    if not all(state["found"][:required]):
        record["reason"] = "boundary is not an exact retained timestamp"
        return record
    baseline = state["baseline"]
    if baseline is None:
        record["reason"] = "last-rest baseline lacks real finite current/voltage support"
        return record
    if state["direction_conflict"] or state["direction"] is None:
        record["reason"] = "active pulse direction is unavailable or changes within the boundaries"
        return record
    if record["direction"] == "unknown":
        record["direction"] = state["direction"]
    if record["t3_s"] is not None and record["target_time_s"] >= record["t3_s"]:
        record["reason"] = "requested target is at or after the following rest"
        return record
    target, choice = state["exact"], "exact"
    if target is None:
        target, choice = state["excess"], "excess"
    if target is None:
        if record["t3_s"] is None and record["target_time_s"] > record["t2_s"]:
            record["reason"] = "requested target lies beyond truncated observed support"
            return record
        target, choice = state["defect"], "defect"
    if target is None:
        record["reason"] = "no real finite current/voltage target exists in the active pulse"
        return record
    actual_delay = target[0] - record["t0_s"]
    timing_error = actual_delay - record["requested_delay_s"]
    record.update(
        actual_time_s=target[0],
        actual_delay_s=actual_delay,
        timing_error_s=timing_error,
        sampling_choice=choice,
        baseline_current_mA=baseline[0],
        baseline_voltage_V=baseline[1],
        target_current_mA=target[1],
        target_voltage_V=target[2],
    )
    if max_error is not None and abs(timing_error) > max_error:
        record["reason"] = "maximum absolute target-time error exceeded"
        return record
    delta_i_mA = target[1] - baseline[0]
    delta_v = target[2] - baseline[1]
    record["delta_current_A"] = delta_i_mA / 1000.0
    record["delta_voltage_V"] = delta_v
    if abs(delta_i_mA) < minimum:
        record["reason"] = "absolute current difference is below the configured minimum"
        return record
    record["resistance_ohm"] = delta_v / record["delta_current_A"]
    record["valid"] = True
    record["quality"] = "complete" if record["t3_s"] is not None else "incomplete_event"
    record["reason"] = "measured from real baseline and target rows"
    return record


@dataclass(frozen=True)
class DCIRResult:
    """Disk-backed one-row-per-event DCIR output, including invalid outcomes."""

    source: object = field(repr=False)
    path: Path = field(repr=False)
    report: dict
    _workspace: object = field(repr=False)
    _budget: object = field(repr=False)

    def iter_batches(self, batch_rows=512):
        """Yield bounded results in natural event order."""
        if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows < 1:
            raise ValueError("Positive batch_rows required")
        rows = min(batch_rows, (self._budget.limit - self._budget.used - 2 * 1024**2) // 4096)
        if rows < 1:
            raise BudgetExceeded("Insufficient DCIR reader budget")
        reservation = 2 * 1024**2 + rows * 4096
        self._budget.reserve(reservation)
        self._workspace.active_readers = getattr(self._workspace, "active_readers", 0) + 1
        try:
            if fingerprint(self.source.path) != self.report["prepared_source_metadata"]:
                raise SourceChangedError("Prepared source changed since DCIR measurement")
            with closing(sqlite3.connect(self.path)) as db:
                cursor = db.execute(
                    "SELECT " + ",".join(_SCHEMA.names) + " FROM results ORDER BY event"
                )
                emitted = False
                while values := cursor.fetchmany(rows):
                    rows_out = []
                    for values_row in values:
                        record = dict(zip(_SCHEMA.names, values_row, strict=True))
                        record["valid"] = bool(record["valid"])
                        rows_out.append(record)
                    emitted = True
                    yield pa.RecordBatch.from_pylist(rows_out, schema=_SCHEMA)
                if not emitted:
                    yield pa.RecordBatch.from_pylist([], schema=_SCHEMA)
        finally:
            self._workspace.active_readers -= 1
            self._budget.release(reservation)

    def table(self, max_rows=100000):
        """Return a guarded pandas table for ordinary notebook inspection."""
        return bounded_frame(self, max_rows)

    def plot(self, *, y="resistance_ohm", max_points=250000, mode="auto"):
        """Plot one-delay DCIR outcomes by readable event number and pulse direction."""
        from .prepared_plotting import plot_compact_result

        return plot_compact_result(
            self,
            x="event",
            y=y,
            series_column="direction",
            series_values=("charge", "discharge", "unknown"),
            max_points=max_points,
            mode=mode,
        )

    def plot_progressive(self, *, y="resistance_ohm", max_points=250000, mode="auto"):
        """Create an idle progressive DCIR plot session."""
        from .prepared_plotting import plot_compact_result

        return plot_compact_result(
            self,
            x="event",
            y=y,
            series_column="direction",
            series_values=("charge", "discharge", "unknown"),
            progressive=True,
            max_points=max_points,
            mode=mode,
        )

    def write_parquet(self, path):
        """Stream results with result-level settings stored once."""
        if Path(path).exists():
            raise FileExistsError("Choose a new DCIR export path")
        return atomic_parquet(self.iter_batches(), path, manifest=self.report)

    def close(self):
        """Remove verified owned temporary storage after active iterators close."""
        if getattr(self._workspace, "active_readers", 0):
            raise ValueError("Close active DCIR iterators first")
        if Path(self._workspace.name).resolve() != self.path.parent:
            raise ValueError("DCIR cleanup target changed")
        self._workspace.cleanup()


def measure_pulse_dcir(
    source,
    delay_s,
    *,
    events=None,
    manual=None,
    max_timing_error_s=None,
    delta_current_min_mA=0.001,
    rest_threshold_mA=0.0001,
    memory_budget=None,
    cancellation=None,
    scratch_dir=None,
    cycles=None,
    capacity=None,
):
    """Measure one signed finite-delay resistance for every event in one bounded scan.

    The target is always one actual retained row: exact first, then excess, then a
    permitted defect fallback. Current is converted from mA to A exactly once. No
    interpolation, smoothing, rest averaging, extrapolation or absolute value is used.
    """
    delay = _scalar("delay_s", delay_s, positive=True)
    max_error = _scalar("max_timing_error_s", max_timing_error_s, optional=True)
    minimum = _scalar("delta_current_min_mA", delta_current_min_mA)
    rest = _scalar("rest_threshold_mA", rest_threshold_mA)
    if events is not None and manual is not None:
        raise ValueError("Choose discovered events or manual boundaries, not both")
    if manual is not None and cycles is not None:
        raise ValueError("Cycle filtering and exact manual boundaries are separate workflows")
    domain = None
    owned_capacity = None
    if cycles is not None:
        values = (
            (cycles,)
            if isinstance(cycles, Integral) and not isinstance(cycles, bool)
            else tuple(cycles)
        )
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or value < 1
            for value in values
        ):
            raise ValueError("Cycle numbers must be positive integers")
        requested = tuple(sorted({int(value) for value in values}))
        if not requested or requested != tuple(range(requested[0], requested[-1] + 1)):
            raise ValueError("DCIR cycle selection must contain consecutive cycle numbers")
        selected_capacity = capacity
        if selected_capacity is None:
            owned_capacity = source.capacity(memory_budget=memory_budget, cancellation=cancellation)
            selected_capacity = owned_capacity
        frame = selected_capacity.cycles(requested).details().table()
        if set(frame.cycle) != set(requested):
            if owned_capacity is not None:
                owned_capacity.close()
            raise ValueError("One or more requested cycles are unavailable")
        domain = (float(frame.start_s.min()), float(frame.end_s.max()))
    owned = None
    if events is None and manual is None:
        owned = source.events(memory_budget=memory_budget, cancellation=cancellation)
        events = owned
    event_source = getattr(events, "source", source)
    if Path(event_source.path).resolve() != Path(source.path).resolve():
        if owned is not None:
            owned.close()
        if owned_capacity is not None:
            owned_capacity.close()
        raise ValueError("Events belong to a different prepared source")
    budget = memory_budget or getattr(events, "_budget", None) or MemoryBudget(2 * 1024**3)
    token = cancellation or CancellationToken()
    initial = fingerprint(source.path)
    try:
        records = [_blank(event, delay) for event in _event_records(events, manual, domain)]
    except BaseException:
        if owned is not None:
            owned.close()
        if owned_capacity is not None:
            owned_capacity.close()
        raise
    records.sort(key=lambda record: record["t0_s"] if record["t0_s"] is not None else -math.inf)
    available = budget.limit - budget.used
    reservation = min(source.read_options.max_batch_bytes * 4 + 2 * 1024**2, available)
    if reservation < 64:
        if owned is not None:
            owned.close()
        if owned_capacity is not None:
            owned_capacity.close()
        raise BudgetExceeded("Insufficient DCIR scan budget")
    budget.reserve(reservation)
    workspace = None
    try:
        parent = Path(scratch_dir or source.path.parent / ".gregorithm-work").resolve()
        parent.mkdir(parents=True, exist_ok=True)
        workspace = tempfile.TemporaryDirectory(prefix="dcir-", dir=parent)
        folder = Path(workspace.name).resolve()
        if not folder.is_relative_to(parent):
            raise ValueError("DCIR workspace escaped its intended directory")
        path = folder / "dcir.sqlite"
        states = [
            {
                "baseline": None,
                "exact": None,
                "excess": None,
                "defect": None,
                "found": [False] * 4,
                "direction": None,
                "direction_conflict": False,
            }
            for _ in records
        ]
        columns = [source.report["time_column"], "current_mA", "voltage_V"]
        reader = source.iter_batches(columns=columns)
        cursor = 0
        try:
            for batch in reader:
                token.check()
                arrays = [
                    np.asarray(batch.column(name).to_numpy(zero_copy_only=False), dtype=np.float64)
                    for name in columns
                ]
                for t, current, voltage in zip(*arrays, strict=True):
                    while cursor < len(records):
                        end = records[cursor]["t3_s"] or records[cursor]["t2_s"]
                        if t <= end:
                            break
                        cursor += 1
                    if cursor >= len(records):
                        continue
                    record, state = records[cursor], states[cursor]
                    if record["t0_s"] is None or t < record["t0_s"]:
                        continue
                    for k, boundary in enumerate(
                        (record["t0_s"], record["t1_s"], record["t2_s"], record["t3_s"])
                    ):
                        if boundary is not None and t == boundary:
                            state["found"][k] = True
                    finite_i = math.isfinite(current)
                    finite = finite_i and math.isfinite(voltage)
                    if t == record["t0_s"] and finite and abs(current) <= rest:
                        state["baseline"] = (float(current), float(voltage))
                    if record["t1_s"] <= t <= record["t2_s"] and finite_i and abs(current) > rest:
                        direction = "charge" if current > 0 else "discharge"
                        if state["direction"] is None:
                            state["direction"] = direction
                        elif state["direction"] != direction:
                            state["direction_conflict"] = True
                    if (
                        record["t1_s"] <= t <= record["t2_s"]
                        and finite
                        and abs(current) > rest
                        and not state["direction_conflict"]
                    ):
                        point = (float(t), float(current), float(voltage))
                        if t == record["target_time_s"]:
                            state["exact"] = point
                        elif t > record["target_time_s"] and state["excess"] is None:
                            state["excess"] = point
                        elif t < record["target_time_s"]:
                            state["defect"] = point
        finally:
            reader.close()
        results = [
            _finish(record, state, max_error, minimum)
            for record, state in zip(records, states, strict=True)
        ]
        with closing(sqlite3.connect(path)) as db:
            declaration = []
            for field_def in _SCHEMA:
                kind = (
                    "TEXT"
                    if pa.types.is_string(field_def.type)
                    else "INTEGER"
                    if pa.types.is_integer(field_def.type) or pa.types.is_boolean(field_def.type)
                    else "REAL"
                )
                declaration.append(field_def.name + " " + kind)
            db.execute("CREATE TABLE results (" + ",".join(declaration) + ")")
            db.executemany(
                "INSERT INTO results VALUES (" + ",".join("?" for _ in _SCHEMA) + ")",
                (tuple(result[name] for name in _SCHEMA.names) for result in results),
            )
            db.commit()
        if fingerprint(source.path) != initial:
            raise SourceChangedError("Prepared source changed during DCIR measurement")
        report = {
            "source_file": source.report["source_file"],
            "prepared_source_metadata": initial,
            "operation": "dcir_measurement",
            "delay_s": delay,
            "max_timing_error_s": max_error,
            "delta_current_min_mA": minimum,
            "rest_threshold_mA": rest,
            "event_origin": "manual" if manual is not None else "discovered",
            "events": len(results),
            "valid_results": sum(result["valid"] for result in results),
            "generated_measurement_identifiers": False,
            "target_interpolation": False,
        }
        return DCIRResult(source, path, report, workspace, budget)
    except BaseException:
        if workspace is not None:
            workspace.cleanup()
        raise
    finally:
        budget.release(reservation)
        if owned is not None:
            owned.close()
        if owned_capacity is not None:
            owned_capacity.close()
