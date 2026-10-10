"""Bounded capacity and energy integration over an existing compact step result."""

from __future__ import annotations

import sqlite3
import tempfile
import warnings
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa

from ._result_tables import bounded_frame
from .diagnostics import BudgetExceeded, SourceChangedError
from .execution import CancellationToken
from .io import atomic_parquet, fingerprint
from .point_calculations import (
    EstimatedMeasurementWarning,
    IncompleteIntegrationWarning,
    _effective_batches,
    directional_integrals,
)

_FIELDS = [
    ("step", pa.int64()),
    ("step_type", pa.string()),
    ("direction", pa.string()),
    ("start_s", pa.float64()),
    ("end_s", pa.float64()),
    ("charge_mAh", pa.float64()),
    ("discharge_mAh", pa.float64()),
    ("net_mAh", pa.float64()),
    ("capacity_mAh", pa.float64()),
    ("capacity_quality", pa.string()),
    ("capacity_covered_s", pa.float64()),
    ("charge_Wh", pa.float64()),
    ("discharge_Wh", pa.float64()),
    ("net_Wh", pa.float64()),
    ("energy_Wh", pa.float64()),
    ("energy_quality", pa.string()),
    ("energy_covered_s", pa.float64()),
    ("duration_s", pa.float64()),
]
_SCHEMA = pa.schema(_FIELDS)


def _quality(invalid, estimated, intervals):
    """Return the shared complete/estimated/partial/unavailable quality vocabulary."""
    if intervals == invalid:
        return "unavailable"
    if invalid:
        return "partial"
    return "estimated" if estimated else "complete"


@dataclass(frozen=True)
class StepQuantities:
    """Disk-backed step Q/E summaries calculated without changing step labels."""

    source: object = field(repr=False)
    path: Path = field(repr=False)
    report: dict
    _workspace: object = field(repr=False)
    _budget: object = field(repr=False)

    def iter_batches(self, batch_rows=512):
        """Yield bounded summaries in natural step order."""
        if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows < 1:
            raise ValueError("Positive batch_rows required")
        rows = min(batch_rows, (self._budget.limit - self._budget.used - 2 * 1024**2) // 4096)
        if rows < 1:
            raise BudgetExceeded("Insufficient step-quantity reader budget")
        reservation = 2 * 1024**2 + rows * 4096
        self._budget.reserve(reservation)
        self._workspace.active_readers = getattr(self._workspace, "active_readers", 0) + 1
        try:
            if fingerprint(self.source.path) != self.report["prepared_source_metadata"]:
                raise SourceChangedError("Prepared source changed since step integration")
            with closing(sqlite3.connect(self.path)) as db:
                cursor = db.execute(
                    "SELECT " + ",".join(_SCHEMA.names) + " FROM quantities ORDER BY step"
                )
                emitted = False
                while values := cursor.fetchmany(rows):
                    emitted = True
                    yield pa.RecordBatch.from_pylist(
                        [dict(zip(_SCHEMA.names, row, strict=True)) for row in values],
                        schema=_SCHEMA,
                    )
                if not emitted:
                    yield pa.RecordBatch.from_pylist([], schema=_SCHEMA)
        finally:
            self._workspace.active_readers -= 1
            self._budget.release(reservation)

    def table(self, max_rows=100000):
        """Return a guarded pandas step table."""
        return bounded_frame(self, max_rows)

    def write_parquet(self, path):
        """Stream summaries with calculation settings stored once."""
        if Path(path).exists():
            raise FileExistsError("Choose a new step-quantity export path")
        return atomic_parquet(self.iter_batches(), path, manifest=self.report)

    def plot(self, *, y="capacity_mAh", max_points=250000, mode="auto"):
        """Plot step quantities by readable step number and classified type."""
        from .prepared_plotting import plot_compact_result

        return plot_compact_result(
            self,
            x="step",
            y=y,
            series_column="step_type",
            series_values=("CC", "CV", "rest", "variable", "unknown"),
            max_points=max_points,
            mode=mode,
        )

    def plot_progressive(self, *, y="capacity_mAh", max_points=250000, mode="auto"):
        """Create an idle progressive step-quantity plot session."""
        from .prepared_plotting import plot_compact_result

        return plot_compact_result(
            self,
            x="step",
            y=y,
            series_column="step_type",
            series_values=("CC", "CV", "rest", "variable", "unknown"),
            progressive=True,
            max_points=max_points,
            mode=mode,
        )

    def close(self):
        """Remove verified owned storage after active iterators close."""
        if getattr(self._workspace, "active_readers", 0):
            raise ValueError("Close active step-quantity iterators first")
        if Path(self._workspace.name).resolve() != self.path.parent:
            raise ValueError("Step-quantity cleanup target changed")
        self._workspace.cleanup()


def integrate_steps(steps, *, include_energy=True, cancellation=None, scratch_dir=None):
    """Integrate Q/E for existing steps with transition intervals owned by the left step.

    The interval ending at the first sample of a new step belongs to the preceding
    step. No interval crosses unavailable current support. Current zero crossings and
    energy use the same exact point-signal model as whole-source and half-cycle totals.
    """
    if not isinstance(include_energy, (bool, np.bool_)):
        raise ValueError("include_energy must be boolean")
    source, budget = steps.source, steps._budget
    token = cancellation or CancellationToken()
    initial = fingerprint(source.path)
    with closing(sqlite3.connect(steps.path)) as db:
        rows = db.execute(
            "SELECT step,step_type,direction,start_s,end_s FROM steps ORDER BY step"
        ).fetchall()
    count = len(rows)
    starts = np.array([row[3] for row in rows], dtype=np.float64)
    ends = np.array([row[4] for row in rows], dtype=np.float64)
    sums = np.zeros((count, 4), dtype=np.float64)
    covered = np.zeros((count, 3), dtype=np.float64)
    intervals = np.zeros(count, dtype=np.int64)
    invalid_q = np.zeros(count, dtype=np.int64)
    invalid_e = np.zeros(count, dtype=np.int64)
    estimated_q = np.zeros(count, dtype=bool)
    estimated_e = np.zeros(count, dtype=bool)
    cache = max(2 * 1024**2, count * 160)
    reader_bytes = min(
        source.read_options.max_batch_bytes, (budget.limit - budget.used - cache) // 16
    )
    if reader_bytes < 64:
        raise BudgetExceeded("Insufficient shared step-integration budget")
    reservation = reader_bytes * 16 + cache
    budget.reserve(reservation)
    workspace = None
    try:
        parent = Path(scratch_dir or source.path.parent / ".gregorithm-work").resolve()
        parent.mkdir(parents=True, exist_ok=True)
        workspace = tempfile.TemporaryDirectory(prefix="step-quantities-", dir=parent)
        folder = Path(workspace.name).resolve()
        if not folder.is_relative_to(parent):
            raise ValueError("Step-quantity workspace escaped its intended directory")
        path = folder / "step_quantities.sqlite"
        columns = [source.report["time_column"], "current_mA"]
        if include_energy:
            columns.append("voltage_V")
        reader = _effective_batches(
            source, columns, steps.report["settings"]["missing_span_s"], token
        )
        previous = None
        try:
            for values in reader:
                times, _, current, voltage, estimate_i, estimate_v = values
                if previous is not None:
                    times = np.r_[previous[0], times]
                    current = np.r_[previous[1], current]
                    estimate_i = np.r_[previous[3], estimate_i]
                    if include_energy:
                        voltage = np.r_[previous[2], voltage]
                        estimate_v = np.r_[previous[4], estimate_v]
                if len(times) < 2:
                    if len(times):
                        previous = (
                            times[-1],
                            current[-1],
                            None if voltage is None else voltage[-1],
                            estimate_i[-1],
                            False if estimate_v is None else estimate_v[-1],
                        )
                    continue
                increments = directional_integrals(
                    times, current, voltage if include_energy else None
                )
                left = times[:-1]
                owner = np.searchsorted(starts, left, side="right") - 1
                valid_owner = (owner >= 0) & (owner < count)
                clipped = np.clip(owner, 0, max(count - 1, 0))
                if count:
                    valid_owner &= left <= ends[clipped]
                dt = np.diff(times)
                for position in np.flatnonzero(valid_owner):
                    k = owner[position]
                    intervals[k] += 1
                    covered[k, 0] += dt[position]
                    q_valid = np.all(np.isfinite(increments[position, :2]))
                    if q_valid:
                        sums[k, :2] += increments[position, :2]
                        covered[k, 1] += dt[position]
                        estimated_q[k] |= bool(estimate_i[position] or estimate_i[position + 1])
                    else:
                        invalid_q[k] += 1
                    if include_energy:
                        e_valid = np.all(np.isfinite(increments[position, 2:]))
                        if e_valid:
                            sums[k, 2:] += increments[position, 2:]
                            covered[k, 2] += dt[position]
                            estimated_e[k] |= bool(
                                estimate_i[position]
                                or estimate_i[position + 1]
                                or estimate_v[position]
                                or estimate_v[position + 1]
                            )
                        else:
                            invalid_e[k] += 1
                previous = (
                    times[-1],
                    current[-1],
                    None if voltage is None else voltage[-1],
                    estimate_i[-1],
                    False if estimate_v is None else estimate_v[-1],
                )
        finally:
            reader.close()
        output = []
        for k, (step, kind, direction, start, end) in enumerate(rows):
            q_quality = _quality(invalid_q[k], estimated_q[k], intervals[k])
            e_quality = (
                _quality(invalid_e[k], estimated_e[k], intervals[k])
                if include_energy
                else "unavailable"
            )
            charge, discharge = sums[k, :2]
            e_charge, e_discharge = sums[k, 2:]
            orient = -1.0 if direction == "discharge" else 1.0
            output.append(
                dict(
                    step=step,
                    step_type=kind,
                    direction=direction,
                    start_s=start,
                    end_s=end,
                    charge_mAh=None if q_quality == "unavailable" else charge,
                    discharge_mAh=None if q_quality == "unavailable" else discharge,
                    net_mAh=None if q_quality == "unavailable" else charge - discharge,
                    capacity_mAh=None
                    if q_quality == "unavailable"
                    else orient * (charge - discharge),
                    capacity_quality=q_quality,
                    capacity_covered_s=covered[k, 1],
                    charge_Wh=None if e_quality == "unavailable" else e_charge,
                    discharge_Wh=None if e_quality == "unavailable" else e_discharge,
                    net_Wh=None if e_quality == "unavailable" else e_charge - e_discharge,
                    energy_Wh=None
                    if e_quality == "unavailable"
                    else orient * (e_charge - e_discharge),
                    energy_quality=e_quality,
                    energy_covered_s=covered[k, 2],
                    duration_s=covered[k, 0],
                )
            )
        with closing(sqlite3.connect(path)) as db:
            declaration = []
            for definition in _SCHEMA:
                kind = (
                    "TEXT"
                    if pa.types.is_string(definition.type)
                    else "INTEGER"
                    if pa.types.is_integer(definition.type)
                    else "REAL"
                )
                declaration.append(definition.name + " " + kind)
            db.execute("CREATE TABLE quantities (" + ",".join(declaration) + ")")
            db.executemany(
                "INSERT INTO quantities VALUES (" + ",".join("?" for _ in _SCHEMA) + ")",
                (tuple(record[name] for name in _SCHEMA.names) for record in output),
            )
            db.commit()
        if fingerprint(source.path) != initial:
            raise SourceChangedError("Prepared source changed during step integration")
        if np.any(estimated_q) or np.any(estimated_e):
            warnings.warn(
                "Step Q/E uses approved isolated-point estimates",
                EstimatedMeasurementWarning,
                stacklevel=2,
            )
        if np.any(invalid_q) or np.any(invalid_e):
            warnings.warn(
                "Some step intervals have unavailable Q/E support",
                IncompleteIntegrationWarning,
                stacklevel=2,
            )
        report = {
            **steps.report,
            "operation": "step_capacity_energy",
            "include_energy": bool(include_energy),
            "transition_interval_owner": "preceding_step",
            "generated_measurement_identifiers": False,
        }
        return StepQuantities(source, path, report, workspace, budget)
    except BaseException:
        if workspace is not None:
            workspace.cleanup()
        raise
    finally:
        budget.release(reservation)
