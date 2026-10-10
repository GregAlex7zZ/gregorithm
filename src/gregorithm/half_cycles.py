"""Bounded, identity-free half-cycle capacity with explicit numerical ownership.

Half-cycle/cycle numbers are readable local physical segment labels, not sample
identifiers. Two passes store only confirmed boundaries and summary rows on disk.
Current direction uses real observations; numerical estimates never confirm a
reversal. Capacity calculations do not run CC/CV or event discovery.
"""

from __future__ import annotations

import math
import sqlite3
import tempfile
import warnings
from contextlib import closing
from dataclasses import asdict, dataclass, field, replace
from numbers import Integral
from pathlib import Path

import numpy as np
import pyarrow as pa

from .diagnostics import BudgetExceeded, MaterializationLimitError, SourceChangedError
from .execution import CancellationToken, MemoryBudget
from .io import atomic_parquet, fingerprint
from .point_calculations import (
    EstimatedMeasurementWarning,
    IncompleteIntegrationWarning,
    _effective_batches,
    directional_integrals,
)


@dataclass(frozen=True)
class HalfCycleConfig:
    """Reviewed operational settings, configurable and not experimentally calibrated.

    cycle_order pairs charge then discharge by default. An initial discharge
    occupies incomplete cycle 1. Rest belongs to the preceding half-cycle;
    initial unassigned contributions remain in the result report.
    """

    rest_threshold_mA: float = 0.0001
    min_confirm_samples: int = 2
    min_confirm_duration_s: float = 0.1
    missing_span_s: float = 10.0
    cycle_order: str = "charge_then_discharge"

    def __post_init__(self):
        """Reject invalid thresholds before reading measurements or allocating state."""
        for name in ("rest_threshold_mA", "min_confirm_duration_s", "missing_span_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            isinstance(self.min_confirm_samples, bool)
            or not isinstance(self.min_confirm_samples, int)
            or self.min_confirm_samples < 1
        ):
            raise ValueError("min_confirm_samples must be a positive integer")
        if self.cycle_order not in {"charge_then_discharge", "discharge_then_charge"}:
            raise ValueError("Unsupported cycle order")


_SCHEMA = pa.schema(
    [
        ("half_cycle", pa.int64()),
        ("cycle", pa.int64()),
        ("direction", pa.string()),
        ("capacity_mAh", pa.float64()),
        ("capacity_quality", pa.string()),
        ("cycle_pair_complete", pa.bool_()),
        ("start_s", pa.float64()),
        ("end_s", pa.float64()),
        ("first_sample_s", pa.float64()),
        ("last_sample_s", pa.float64()),
        ("charge_mAh", pa.float64()),
        ("discharge_mAh", pa.float64()),
        ("capacity_covered_s", pa.float64()),
        ("duration_s", pa.float64()),
        ("boundary_quality", pa.string()),
        ("energy_Wh", pa.float64()),
        ("energy_quality", pa.string()),
        ("charge_Wh", pa.float64()),
        ("discharge_Wh", pa.float64()),
        ("energy_covered_s", pa.float64()),
    ]
)
_BASIC = (
    "half_cycle",
    "cycle",
    "direction",
    "capacity_mAh",
    "capacity_quality",
    "cycle_pair_complete",
)


def _cycle_numbers(numbers):
    """Accept Python/NumPy integer labels, rejecting booleans and nonpositive cycles."""
    values = (numbers,) if isinstance(numbers, Integral) else tuple(numbers)
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1
        for value in values
    ):
        raise ValueError("Cycle numbers must be positive integers")
    return tuple(sorted({int(value) for value in values}))


@dataclass(frozen=True)
class HalfCycleResult:
    """Disk-backed half-cycle summaries; table(), cycles() and details() are simple views.

    table() has explicit row/byte limits; iter_batches() reads arbitrarily many
    summaries without a full dataframe. close() closes shared selected views too.
    Tables already materialized remain independent of the temporary workspace.
    """

    path: Path
    report: dict
    _workspace: object = field(repr=False)
    _cycles: tuple | None = None
    _details: bool = False
    _materialize_bytes: int = 64 * 1024**2
    _budget: object = field(default=None, repr=False)
    _source_path: Path | None = field(default=None, repr=False)

    def __repr__(self):
        """Show readable source/count information instead of the technical scratch pathname."""
        return (
            f"HalfCycleResult(source={self.report['source_file']!r}, "
            f"half_cycles={self.report['half_cycles']}, selected_cycles={self._cycles!r})"
        )

    def cycles(self, numbers):
        """Select original positive cycle numbers without renumbering or joining data gaps."""
        values = _cycle_numbers(numbers)
        if self._cycles is not None:
            values = tuple(value for value in values if value in self._cycles)
        return replace(self, _cycles=values)

    def details(self):
        """Include directional quantities, numerical/measured bounds and coverage columns."""
        return replace(self, _details=True)

    def summary(self):
        """Return one row per cycle with paired net quantities and quality-aware efficiencies."""
        from .cycle_summary import SummaryView

        return SummaryView(self, "cycle_summary")

    def normalize(self, **settings):
        """Add per-mass and/or per-area columns lazily without replacing original columns or null masks. Example: normalize(view, mass_g=0.005, area_cm2=1.13). Output-name collisions and nonpositive denominators raise ValueError."""
        from .cycle_summary import normalize_summary

        return normalize_summary(self, **settings)

    def iter_batches(self, batch_rows=1024):
        """Read summary rows in original half-cycle order, with no source-wide dataframe."""
        if not isinstance(batch_rows, int) or isinstance(batch_rows, bool) or batch_rows < 1:
            raise ValueError("Positive batch_rows required")
        budget = self._budget or MemoryBudget(2 * 1024**3)
        cache = 2 * 1024**2
        available = budget.limit - budget.used - cache
        if available < 4096:
            raise BudgetExceeded("Insufficient shared budget for summary reading")
        batch_rows = min(batch_rows, max(1, self._materialize_bytes // 4096), available // 4096)
        reservation = cache + 4096 * batch_rows
        budget.reserve(reservation)
        self._workspace.active_readers = getattr(self._workspace, "active_readers", 0) + 1
        try:
            yield from self._read_batches(batch_rows)
        finally:
            self._workspace.active_readers -= 1
            budget.release(reservation)

    def _read_batches(self, batch_rows):
        """Read SQLite rows under the public iterator's already reserved buffer allowance."""
        if not self.path.is_file():
            raise ValueError("Half-cycle result is closed or no longer available")
        if not isinstance(batch_rows, int) or isinstance(batch_rows, bool) or batch_rows < 1:
            raise ValueError("Positive batch_rows required")
        names = list(_SCHEMA.names if self._details else _BASIC)
        if self.report["include_energy"] and not self._details:
            names += ["energy_Wh", "energy_quality"]
        if not self.report["include_energy"]:
            names = [
                name
                for name in names
                if name
                not in {
                    "energy_Wh",
                    "energy_quality",
                    "charge_Wh",
                    "discharge_Wh",
                    "energy_covered_s",
                }
            ]
        schema = pa.schema([_SCHEMA.field(name) for name in names])
        with closing(sqlite3.connect(self.path)) as database:
            database.execute("PRAGMA cache_size=-2048")
            database.execute("PRAGMA temp_store=FILE")
            where = ""
            if self._cycles is not None:
                database.execute("CREATE TEMP TABLE requested_cycles (cycle INTEGER PRIMARY KEY)")
                database.executemany(
                    "INSERT INTO requested_cycles VALUES (?)", ((value,) for value in self._cycles)
                )
                where = " WHERE cycle IN (SELECT cycle FROM requested_cycles)"
            cursor = database.execute(
                "SELECT " + ",".join(names) + " FROM summaries" + where + " ORDER BY half_cycle"
            )
            emitted = False
            while rows := cursor.fetchmany(batch_rows):
                emitted = True
                data = {name: [row[k] for row in rows] for k, name in enumerate(names)}
                for name in names:
                    if pa.types.is_boolean(schema.field(name).type):
                        data[name] = [
                            bool(value) if value is not None else None for value in data[name]
                        ]
                yield pa.RecordBatch.from_pydict(data, schema=schema)
            if not emitted:
                yield pa.RecordBatch.from_pylist([], schema=schema)

    def table(self, max_rows=100000):
        """Return a bounded pandas table; excessive requests raise rather than truncate."""
        if not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows < 0:
            raise ValueError("Nonnegative max_rows required")
        budget = self._budget or MemoryBudget(2 * 1024**3)
        budget.reserve(self._materialize_bytes)
        try:
            return self._table(max_rows)
        finally:
            budget.release(self._materialize_bytes)

    def _table(self, max_rows):
        """Materialize bounded Arrow/pandas data with an owned managed-buffer reservation."""
        blocks, rows, size = [], 0, 0
        reader = self.iter_batches()
        try:
            for block in reader:
                rows += block.num_rows
                size += block.nbytes
                if rows > max_rows or size * 3 > self._materialize_bytes:
                    raise MaterializationLimitError(
                        "Half-cycle table exceeds materialization limits; use iter_batches()"
                    )
                blocks.append(block)
        finally:
            reader.close()
        if blocks:
            return pa.Table.from_batches(blocks).to_pandas()
        names = list(_SCHEMA.names if self._details else _BASIC)
        if self.report["include_energy"] and not self._details:
            names += ["energy_Wh", "energy_quality"]
        if not self.report["include_energy"]:
            names = [
                name
                for name in names
                if not name.endswith("Wh") and name not in {"energy_quality", "energy_covered_s"}
            ]
        return pa.Table.from_batches(
            [], schema=pa.schema([_SCHEMA.field(name) for name in names])
        ).to_pandas()

    def write_parquet(self, path):
        """Export the requested summary view incrementally, with metadata once in a sidecar."""
        return atomic_parquet(
            self.iter_batches(), path, manifest={**self.report, "selected_cycles": self._cycles}
        )

    def plot(self, *, y="capacity_mAh", max_points=250000, mode="auto"):
        """Plot charge/discharge half-cycle results against readable half-cycle numbers."""
        from .prepared_plotting import plot_compact_result

        return plot_compact_result(
            self,
            x="half_cycle",
            y=y,
            series_column="direction",
            series_values=("charge", "discharge"),
            max_points=max_points,
            mode=mode,
        )

    def plot_progressive(self, *, y="capacity_mAh", max_points=250000, mode="auto"):
        """Create an idle progressive half-cycle plot session."""
        from .prepared_plotting import plot_compact_result

        return plot_compact_result(
            self,
            x="half_cycle",
            y=y,
            series_column="direction",
            series_values=("charge", "discharge"),
            progressive=True,
            max_points=max_points,
            mode=mode,
        )

    def close(self):
        """Remove this operation's verified temporary storage; shared views become closed."""
        if getattr(self._workspace, "active_readers", 0):
            raise ValueError("Close active summary iterators before closing the result")
        if Path(self._workspace.name).resolve() != self.path.parent:
            raise ValueError("Temporary cleanup target changed; refusing recursive cleanup")
        self._workspace.cleanup()


@dataclass(frozen=True)
class PreparedCycleSelection:
    """Lightweight cycle request; selection retains global numbering during calculation."""

    source: object
    numbers: tuple

    def __post_init__(self):
        """Validate a cycle request before any scientific scan and retain sorted unique labels."""
        object.__setattr__(self, "numbers", _cycle_numbers(self.numbers))

    def capacity(self, **settings):
        """Compute half-cycle net capacity and select requested source cycle numbers."""
        return half_cycle_capacity(self.source, **settings).cycles(self.numbers)


class _Accumulator:
    """Constant-size directional sums and independent support quality for one domain."""

    def __init__(self, energy):
        """Allocate only two or four scalar sums, never a per-source array."""
        self.energy = energy
        self.sums = np.zeros(4 if energy else 2)
        self.correction = np.zeros_like(self.sums)
        self.duration = self.q_duration = self.e_duration = 0.0
        self.q_valid = self.e_valid = self.q_missing = self.e_missing = 0
        self.q_estimated = self.e_estimated = False

    def add(self, values, duration, estimated_i, estimated_v):
        """Merge bounded original-interval portions without double counting boundaries."""
        if not len(duration):
            return
        q_ok = np.isfinite(values[:, 0]) & np.isfinite(values[:, 1])
        self.q_valid += int(np.count_nonzero(q_ok))
        self.q_missing += int(np.count_nonzero(~q_ok))
        self.q_duration += float(np.sum(duration[q_ok]))
        self.duration += float(np.sum(duration))
        self.q_estimated |= bool(np.any(estimated_i & q_ok))
        if self.energy:
            e_ok = np.isfinite(values[:, 2]) & np.isfinite(values[:, 3])
            self.e_valid += int(np.count_nonzero(e_ok))
            self.e_missing += int(np.count_nonzero(~e_ok))
            self.e_duration += float(np.sum(duration[e_ok]))
            self.e_estimated |= bool(np.any((estimated_i | estimated_v) & e_ok))
        subtotal = np.nansum(values, axis=0)
        combined = self.sums + subtotal
        self.correction += np.where(
            np.abs(self.sums) >= np.abs(subtotal),
            (self.sums - combined) + subtotal,
            (subtotal - combined) + self.sums,
        )
        self.sums = combined

    def result(self):
        """Return known amounts and coverage; no supported interval means None, not zero."""
        totals = self.sums + self.correction
        q_quality = (
            "unavailable"
            if not self.q_valid
            else (
                "partial" if self.q_missing else ("estimated" if self.q_estimated else "complete")
            )
        )
        e_quality = (
            "unavailable"
            if not self.e_valid
            else (
                "partial" if self.e_missing else ("estimated" if self.e_estimated else "complete")
            )
        )
        return {
            "charge_mAh": float(totals[0]) if self.q_valid else None,
            "discharge_mAh": float(totals[1]) if self.q_valid else None,
            "capacity_quality": q_quality,
            "capacity_covered_s": self.q_duration,
            "duration_s": self.duration,
            "charge_Wh": float(totals[2]) if self.energy and self.e_valid else None,
            "discharge_Wh": float(totals[3]) if self.energy and self.e_valid else None,
            "energy_quality": e_quality if self.energy else None,
            "energy_covered_s": self.e_duration if self.energy else None,
        }


def _sign(current, tolerance):
    """Classify real finite current; unknown is not rest and tolerance does not zero data."""
    if not math.isfinite(current):
        return 3
    return 1 if current > tolerance else -1 if current < -tolerance else 0


class _DirectionScan:
    """Run-based measured confirmation, storing only confirmed half-cycle boundaries."""

    def __init__(self, database, config):
        """Start with initial unassigned support, no invented active half-cycle."""
        self.database, self.config = database, config
        self.direction = self.half = self.cycle = 0
        self.pending = None
        self.last_real = self.last_raw_time = None
        self.missing_run = 0
        self.first_time = self.last_time = None

    def boundary(self, time, current):
        """Resolve onset/root support from real brackets, never bridge an unresolved gap."""
        if self.last_real is None:
            return time, "observed_start"
        previous_t, previous_i = self.last_real
        if self.missing_run and (
            self.missing_run != 1 or time - previous_t > self.config.missing_span_s
        ):
            return time, "unknown_transition"
        if previous_i * current < 0:
            return previous_t + (time - previous_t) * (
                -previous_i / (current - previous_i)
            ), "linear_zero_crossing"
        return previous_t, "rest_onset" if _sign(
            previous_i, self.config.rest_threshold_mA
        ) == 0 else "observed_start"

    def consume(self, times, raw, effective, estimated, ignored):
        """Process uniform direction runs in batches; estimates never count as observations."""
        if not len(times):
            return
        if self.first_time is None:
            self.first_time = float(times[0])
        self.last_time = float(times[-1])
        labels = np.where(
            ~np.isfinite(raw),
            3,
            np.where(
                raw > self.config.rest_threshold_mA,
                1,
                np.where(raw < -self.config.rest_threshold_mA, -1, 0),
            ),
        )
        labels[ignored] = 4
        edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(times)]
        for lo, hi in zip(edges[:-1], edges[1:]):
            kind = int(labels[lo])
            if kind in {0, 4}:
                self.pending = None
            elif kind == 3:
                if (
                    hi - lo != 1
                    or not estimated[lo]
                    or self.pending is None
                    or _sign(effective[lo], self.config.rest_threshold_mA) != self.pending[0]
                ):
                    self.pending = None
                self.missing_run += hi - lo
            elif kind == self.direction:
                self.pending = None
            else:
                if self.pending is None or self.pending[0] != kind:
                    start, quality = self.boundary(float(times[lo]), float(raw[lo]))
                    self.pending = [kind, float(times[lo]), start, quality, 0, self.last_raw_time]
                self.pending[4] += hi - lo
                if self.pending[4] >= self.config.min_confirm_samples and times[
                    hi - 1
                ] >= np.nextafter(self.pending[1] + self.config.min_confirm_duration_s, -np.inf):
                    _, first, start, quality, _, old_last = self.pending
                    if self.half:
                        self.database.execute(
                            "UPDATE boundaries SET end_s=?,last_sample_s=? WHERE half_cycle=?",
                            (start, old_last, self.half),
                        )
                    opening = 1 if self.config.cycle_order == "charge_then_discharge" else -1
                    if not self.half or kind == opening:
                        self.cycle += 1
                    paired = bool(self.half and kind != opening and self.direction == opening)
                    if paired:
                        self.database.execute(
                            "UPDATE boundaries SET cycle_pair_complete=1 WHERE half_cycle=?",
                            (self.half,),
                        )
                    self.half += 1
                    self.database.execute(
                        "INSERT INTO boundaries VALUES (?,?,?,?,?,?,?,?,?)",
                        (
                            self.half,
                            self.cycle,
                            "charge" if kind > 0 else "discharge",
                            start,
                            first,
                            None,
                            None,
                            int(paired),
                            quality,
                        ),
                    )
                    self.direction = kind
                    self.pending = None
            if kind != 3:
                finite = np.flatnonzero(np.isfinite(raw[lo:hi]))
                if len(finite):
                    last = lo + int(finite[-1])
                    self.last_real = (float(times[last]), float(raw[last]))
                    self.missing_run = hi - 1 - last
                else:
                    self.missing_run += hi - lo
            self.last_raw_time = float(times[hi - 1])

    def finish(self):
        """End the final confirmed domain at the last real timestamp, with no extrapolation."""
        if self.half:
            self.database.execute(
                "UPDATE boundaries SET end_s=?,last_sample_s=? WHERE half_cycle=?",
                (self.last_time, self.last_time, self.half),
            )


def _diagnostic_ranges(windows):
    """Validate/merge explicit time windows; these change counting only, never integration."""
    result, endpoints = [], []
    for lo, hi in sorted(windows):
        if not math.isfinite(lo) or not math.isfinite(hi) or hi < lo:
            raise ValueError("Diagnostic windows require finite ordered measured times")
        endpoints.extend((float(lo), float(hi)))
        if result and lo <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], hi))
        else:
            result.append((float(lo), float(hi)))
    return np.array(result, dtype=float).reshape(-1, 2), np.unique(endpoints)


def _split(t0, i0, v0, t1, i1, v1, at, energy):
    """Split an original linear interval numerically; no source row is inserted."""
    fraction = (at - t0) / (t1 - t0)
    im = i0 + (i1 - i0) * fraction
    vm = None if not energy else v0 + (v1 - v0) * fraction
    left = directional_integrals([t0, at], [i0, im], None if not energy else [v0, vm])
    right = directional_integrals([at, t1], [im, i1], None if not energy else [vm, v1])
    return left, right


def half_cycle_capacity(
    source,
    *,
    config=None,
    include_energy=False,
    diagnostic_windows=(),
    current_column="current_mA",
    voltage_column="voltage_V",
    memory_budget=None,
    cancellation=None,
    scratch_dir=None,
):
    """Calculate direction-oriented net capacity for confirmed half-cycles in bounded passes.

    Use data.capacity().table(), optionally include_energy=True, or
    data.cycles([2,3]).capacity().table(). Original source numbering is preserved.
    Explicit diagnostic_windows use exact retained endpoint times, ignore only
    cycle creation and retain all physical Q/E in the surrounding half-cycle.
    They do not invoke detection or automatically mark events. Result summaries
    live in owned temporary disk storage; use close() or export when appropriate.
    """
    cfg = config or HalfCycleConfig()
    if not isinstance(cfg, HalfCycleConfig):
        raise TypeError("config must be HalfCycleConfig")
    windows, endpoints = _diagnostic_ranges(diagnostic_windows)
    budget = memory_budget or MemoryBudget(2 * 1024**3)
    cache_bytes = 2 * 1024**2
    reader_bytes = min(
        source.read_options.max_batch_bytes, (budget.limit - budget.used - cache_bytes) // 16
    )
    if reader_bytes < 64:
        raise BudgetExceeded("Insufficient shared half-cycle calculation budget")
    reservation = reader_bytes * 16 + cache_bytes
    view = replace(source, read_options=replace(source.read_options, max_batch_bytes=reader_bytes))
    token = cancellation or CancellationToken()
    columns = [source.report["time_column"], current_column]
    initial = fingerprint(source.path)
    parent = Path(scratch_dir or source.path.parent / ".gregorithm-work").resolve()
    parent.mkdir(parents=True, exist_ok=True)
    workspace = tempfile.TemporaryDirectory(prefix="capacity-", dir=parent)
    work = Path(workspace.name).resolve()
    if not work.is_relative_to(parent):
        raise ValueError("Calculation scratch path escaped its intended directory")
    path = work / "half_cycles.sqlite"
    try:
        budget.reserve(reservation)
    except BaseException:
        workspace.cleanup()
        raise
    try:
        with closing(sqlite3.connect(path)) as database:
            database.execute("PRAGMA cache_size=-2048")
            database.execute("PRAGMA temp_store=FILE")
            database.execute(
                "CREATE TABLE boundaries (half_cycle INTEGER PRIMARY KEY, cycle INTEGER, direction TEXT, start_s REAL, first_sample_s REAL, last_sample_s REAL, end_s REAL, cycle_pair_complete INTEGER, boundary_quality TEXT)"
            )
            database.execute(
                "CREATE TABLE summaries ("
                + ",".join(
                    f"{name} {'TEXT' if pa.types.is_string(_SCHEMA.field(name).type) else 'REAL' if pa.types.is_floating(_SCHEMA.field(name).type) else 'INTEGER'}"
                    + (" PRIMARY KEY" if name == "half_cycle" else "")
                    for name in _SCHEMA.names
                )
                + ")"
            )
            scanner = _DirectionScan(database, cfg)
            found = np.zeros(len(endpoints), dtype=bool)
            reader = _effective_batches(view, columns, cfg.missing_span_s, token)
            try:
                for t, raw_i, effective_i, _, estimates, _ in reader:
                    ignored = np.zeros(len(t), dtype=bool)
                    if len(windows):
                        position = np.searchsorted(windows[:, 0], t, side="right") - 1
                        ignored = (position >= 0) & (t <= windows[np.maximum(position, 0), 1])
                        locations = np.searchsorted(t, endpoints)
                        found |= (locations < len(t)) & (
                            t[np.minimum(locations, len(t) - 1)] == endpoints
                        )
                    scanner.consume(t, raw_i, effective_i, estimates, ignored)
            finally:
                reader.close()
            if not np.all(found):
                raise ValueError("Diagnostic window endpoints must be retained measured timestamps")
            scanner.finish()
            database.commit()
            cursor = database.execute(
                "SELECT half_cycle,cycle,direction,start_s,first_sample_s,last_sample_s,end_s,cycle_pair_complete,boundary_quality FROM boundaries ORDER BY half_cycle"
            )
            current = None
            following = cursor.fetchone()
            accumulator = _Accumulator(include_energy)
            unassigned = None

            def save():
                """Persist one completed domain and retain initial unassigned amounts once."""
                nonlocal unassigned
                values = accumulator.result()
                if current is None:
                    unassigned = values
                    return
                h, c, direction, start, first, last, end, pair, quality = current
                sign = 1 if direction == "charge" else -1
                capacity = (
                    None
                    if values["charge_mAh"] is None
                    else sign * (values["charge_mAh"] - values["discharge_mAh"])
                )
                energy = (
                    None
                    if values["charge_Wh"] is None
                    else sign * (values["charge_Wh"] - values["discharge_Wh"])
                )
                row = {
                    **values,
                    "half_cycle": h,
                    "cycle": c,
                    "direction": direction,
                    "capacity_mAh": capacity,
                    "energy_Wh": energy,
                    "start_s": start,
                    "end_s": end,
                    "first_sample_s": first,
                    "last_sample_s": last,
                    "cycle_pair_complete": bool(pair),
                    "boundary_quality": quality,
                }
                database.execute(
                    "INSERT INTO summaries VALUES (" + ",".join("?" for _ in _SCHEMA.names) + ")",
                    [row[name] for name in _SCHEMA.names],
                )

            if include_energy:
                columns += [voltage_column]
            reader = _effective_batches(view, columns, cfg.missing_span_s, token)
            previous = None
            estimated_points = 0
            try:
                for t, _, i, v, ei, ev in reader:
                    estimated_points += int(np.count_nonzero(ei)) + (
                        int(np.count_nonzero(ev)) if include_energy else 0
                    )
                    if previous is not None:
                        t, i = np.r_[previous[0], t], np.r_[previous[1], i]
                        if include_energy:
                            v = np.r_[previous[2], v]
                        ei, ev = np.r_[previous[3], ei], np.r_[previous[4], ev]
                    values = directional_integrals(t, i, v if include_energy else None)
                    duration = np.diff(t)
                    qi, qe = ei[:-1] | ei[1:], ev[:-1] | ev[1:]
                    offset = 0
                    while offset < len(duration):
                        while following is not None and following[3] <= t[offset]:
                            save()
                            current, following = following, cursor.fetchone()
                            accumulator = _Accumulator(include_energy)
                        if following is None:
                            end_index = len(duration)
                        else:
                            end_index = int(np.searchsorted(t[1:], following[3], side="right"))
                        accumulator.add(
                            values[offset:end_index],
                            duration[offset:end_index],
                            qi[offset:end_index],
                            qe[offset:end_index],
                        )
                        offset = end_index
                        if (
                            offset < len(duration)
                            and following is not None
                            and t[offset] < following[3] < t[offset + 1]
                        ):
                            boundary = following[3]
                            left, right = _split(
                                t[offset],
                                i[offset],
                                None if not include_energy else v[offset],
                                t[offset + 1],
                                i[offset + 1],
                                None if not include_energy else v[offset + 1],
                                boundary,
                                include_energy,
                            )
                            accumulator.add(
                                left,
                                np.array([boundary - t[offset]]),
                                qi[offset : offset + 1],
                                qe[offset : offset + 1],
                            )
                            save()
                            current, following = following, cursor.fetchone()
                            accumulator = _Accumulator(include_energy)
                            accumulator.add(
                                right,
                                np.array([t[offset + 1] - boundary]),
                                qi[offset : offset + 1],
                                qe[offset : offset + 1],
                            )
                            offset += 1
                    previous = (
                        t[-1],
                        i[-1],
                        None if not include_energy else v[-1],
                        bool(ei[-1]),
                        bool(ev[-1]),
                    )
            finally:
                reader.close()
            save()
            database.execute("CREATE INDEX cycle_selection ON summaries(cycle)")
            database.commit()
            token.check()
            if fingerprint(source.path) != initial:
                raise SourceChangedError("Prepared source changed during half-cycle calculation")
            partial = database.execute(
                "SELECT COUNT(*) FROM summaries WHERE capacity_quality IN ('partial','unavailable') OR energy_quality IN ('partial','unavailable')"
            ).fetchone()[0]
            report = {
                "source_file": source.report["source_file"],
                "prepared_source_metadata": initial,
                "units": {"time": "s", "current": "mA", "voltage": "V"},
                "measurement_columns": {
                    "time": columns[0],
                    "current": current_column,
                    "voltage": voltage_column if include_energy else None,
                },
                "signal_model": "linear point current; independent linear point voltage",
                "operation": "half_cycle_capacity",
                "settings": asdict(cfg),
                "include_energy": include_energy,
                "diagnostic_windows": windows.tolist(),
                "half_cycles": scanner.half,
                "cycles": scanner.cycle,
                "unassigned": unassigned,
                "generated_measurement_identifiers": False,
                "managed_reservation_bytes": reservation,
            }
        if estimated_points:
            warnings.warn(
                "Half-cycle integration uses isolated-point estimates; original measurements unchanged",
                EstimatedMeasurementWarning,
                stacklevel=2,
            )
        if partial or (
            unassigned
            and unassigned["duration_s"] > 0
            and (
                unassigned["capacity_quality"] in {"partial", "unavailable"}
                or (include_energy and unassigned["energy_quality"] in {"partial", "unavailable"})
            )
        ):
            warnings.warn(
                "Some half-cycle totals have unavailable interval support",
                IncompleteIntegrationWarning,
                stacklevel=2,
            )
        return HalfCycleResult(
            path,
            report,
            workspace,
            _materialize_bytes=min(64 * 1024**2, budget.limit // 4),
            _budget=budget,
            _source_path=source.path.resolve(),
        )
    except BaseException:
        workspace.cleanup()
        raise
    finally:
        budget.release(reservation)
