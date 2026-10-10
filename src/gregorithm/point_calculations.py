"""Identity-free point integration, with bounded isolated-missing-point estimates.

This module implements global source totals, not half-cycle summaries. Scientific
step/cycle interpretation remains a separate dependency; no classification runs
as a hidden side effect. Currents are explicitly mA, voltages V and times seconds.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .diagnostics import BudgetExceeded, SchemaError, SourceChangedError
from .execution import CancellationToken, MemoryBudget
from .io import fingerprint
from .preparation import TimestampOrderError


class EstimatedMeasurementWarning(UserWarning):
    """An isolated missing measurement was estimated for this calculation only."""


class IncompleteIntegrationWarning(UserWarning):
    """Unavailable interval support makes the reported known total partial."""


@dataclass(frozen=True)
class SourceTotals:
    """Small global calculation result; table() returns one readable pandas row.

    Charge/discharge are separate directional quantities; net is charge minus
    discharge. These are source totals, not the net capacity of a half-cycle.
    Settings and quality are stored once in report, without generated IDs.
    """

    values: dict
    report: dict

    def arrow(self):
        """Return the one-row result as Arrow without requiring pandas or a GUI."""
        return pa.table({key: [value] for key, value in self.values.items()})

    def table(self):
        """Return a one-row dataframe; install the notebook extra for pandas support."""
        return self.arrow().to_pandas()


def estimate_isolated(times, values, max_span_s=10.0):
    """Estimate only isolated nonfinite values between immediate real finite neighbors.

    The complete neighbor time span must be positive and <= max_span_s. Endpoints,
    consecutive missing values and longer brackets remain unavailable. Return a
    new calculation array and an estimate mask; input arrays are never modified.
    Supply original neighboring endpoints when calling across batch boundaries.
    """
    result = np.asarray(values, dtype=np.float64).copy()
    times = np.asarray(times, dtype=np.float64)
    estimated = np.zeros(len(result), dtype=bool)
    if len(result) >= 3:
        span = times[2:] - times[:-2]
        mask = (
            ~np.isfinite(result[1:-1])
            & np.isfinite(result[:-2])
            & np.isfinite(result[2:])
            & (span > 0)
            & (span <= max_span_s)
        )
        positions = np.flatnonzero(mask) + 1
        result[positions] = result[positions - 1] + (
            (times[positions] - times[positions - 1])
            / (times[positions + 1] - times[positions - 1])
        ) * (result[positions + 1] - result[positions - 1])
        estimated[positions] = True
    return result, estimated


def directional_integrals(times, current, voltage=None):
    """Return one interval row of charge/discharge mAh and optional directional Wh.

    Actual time spacing is used with no measured-gap cutoff. Opposite-sign
    current is split at its linear zero crossing. Energy is the exact integral
    of separately linear current and voltage, not endpoint-power trapezoids.
    Invalid current makes Q and E unavailable; invalid V affects E only.
    No absolute-value/clamping operation alters voltage or energy signs.
    """
    t, i = np.asarray(times, dtype=np.float64), np.asarray(current, dtype=np.float64)
    dt = np.diff(t)
    if not np.all(np.isfinite(t)) or np.any(dt <= 0):
        raise TimestampOrderError(
            "Prepared measurement times must be finite and strictly increasing"
        )
    a, b = i[:-1], i[1:]
    valid = np.isfinite(a) & np.isfinite(b)
    positive = (a >= 0) & (b >= 0)
    negative = (a <= 0) & (b <= 0) & ~positive
    crossing = ((a < 0) & (b > 0)) | ((a > 0) & (b < 0))
    delta = b - a
    net = dt * (a + b) / 7200.0
    charge = np.where(positive, net, 0.0)
    discharge = np.where(negative, -net, 0.0)
    denominator = np.where(crossing, np.abs(delta), 1.0)
    charge = np.where(crossing, dt * np.maximum(a, b) ** 2 / (7200.0 * denominator), charge)
    discharge = np.where(crossing, dt * np.minimum(a, b) ** 2 / (7200.0 * denominator), discharge)
    charge[~valid] = discharge[~valid] = np.nan
    if voltage is None:
        return np.column_stack((charge, discharge))
    v = np.asarray(voltage, dtype=np.float64)
    va, dv = v[:-1], np.diff(v)
    ev = valid & np.isfinite(v[:-1]) & np.isfinite(v[1:])
    root = np.divide(-a, delta, out=np.zeros_like(a), where=crossing)
    total = dt * (va * a + (va * delta + a * dv) / 2 + delta * dv / 3) / 3600000.0
    before = (
        dt
        * (va * a * root + (va * delta + a * dv) * root**2 / 2 + delta * dv * root**3 / 3)
        / 3600000.0
    )
    e_charge = np.where(positive, total, 0.0)
    e_discharge = np.where(negative, -total, 0.0)
    e_charge = np.where(crossing, np.where(a > 0, before, total - before), e_charge)
    e_discharge = np.where(crossing, np.where(a < 0, -before, -(total - before)), e_discharge)
    e_charge[~ev] = e_discharge[~ev] = np.nan
    return np.column_stack((charge, discharge, e_charge, e_discharge))


def _numeric(block, name):
    """Read one bounded numeric column, requiring explicitly configured physical units."""
    if name not in block.schema.names:
        raise SchemaError(f"Missing measurement column: {name}")
    try:
        return pc.cast(block.column(block.schema.get_field_index(name)), pa.float64()).to_numpy(
            zero_copy_only=False
        )
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as error:
        raise SchemaError(f"Measurement column must be numeric: {name}") from error


def _effective_batches(
    source,
    columns,
    missing_span_s,
    cancellation,
    *,
    extra_columns=(),
    include_original=False,
    rest_threshold_mA=0.0001,
):
    """Yield bounded raw/effective point arrays with original-neighbor lookahead.

    The caller owns the shared memory reservation. Tuple fields are times,
    original current, effective current, optional effective voltage, current
    estimate mask and voltage estimate mask. No identities are generated.
    include_original additionally returns the projected original RecordBatch and
    an isolated-estimate continuity mask requiring both real neighbours to have
    the same active direction. extra_columns are retained without numeric casting.
    """
    reader = source.iter_batches(columns=list(dict.fromkeys([*columns, *extra_columns])))
    previous = None
    try:
        block = next(reader, None)
        while block is not None:
            cancellation.check()
            following = next(reader, None)
            raw = [_numeric(block, name) for name in columns]
            if not block.num_rows:
                block = following
                continue
            after = (
                None
                if following is None
                else [_numeric(following.slice(0, 1), name)[0] for name in columns]
            )
            offset = int(previous is not None)
            padded = [
                np.concatenate(
                    (
                        [] if previous is None else [previous[k]],
                        values,
                        [] if after is None else [after[k]],
                    )
                )
                for k, values in enumerate(raw)
            ]
            signals, masks = [], []
            for values in padded[1:]:
                effective, mask = estimate_isolated(padded[0], values, missing_span_s)
                signals.append(effective[offset : offset + block.num_rows])
                masks.append(mask[offset : offset + block.num_rows])
            result = (
                raw[0],
                raw[1],
                signals[0],
                signals[1] if len(signals) > 1 else None,
                masks[0],
                masks[1] if len(masks) > 1 else np.zeros(block.num_rows, dtype=bool),
            )
            if include_original:
                neighbors = np.zeros(len(padded[1]), dtype=bool)
                neighbors[1:-1] = (
                    (padded[1][:-2] > rest_threshold_mA) & (padded[1][2:] > rest_threshold_mA)
                ) | ((padded[1][:-2] < -rest_threshold_mA) & (padded[1][2:] < -rest_threshold_mA))
                result += (block, masks[0] & neighbors[offset : offset + block.num_rows])
            yield result
            previous = [values[-1] for values in raw]
            block = following
    finally:
        reader.close()


def source_totals(
    source,
    *,
    include_energy=True,
    current_column="current_mA",
    voltage_column="voltage_V",
    missing_span_s=10.0,
    memory_budget=None,
    cancellation=None,
):
    """Calculate bounded global directional/net capacity and optionally energy.

    Call on a PreparedSource, or use its totals() convenience method. Missing
    current/voltage is linearly estimated only for isolated points with immediate
    real neighbors spanning <= missing_span_s. Original source columns stay
    unchanged. Defaults are reviewed operational settings, not calibration.
    include_energy=False reads only time/current and performs no voltage work.
    Returned totals cover known support; report and table distinguish partial
    quantities from complete or estimated quantities. No half-cycle is inferred.
    """
    if not math.isfinite(missing_span_s) or missing_span_s < 0:
        raise ValueError("missing_span_s must be finite and nonnegative")
    budget = memory_budget or MemoryBudget(2 * 1024**3)
    available = (budget.limit - budget.used) // 16
    reader_bytes = min(source.read_options.max_batch_bytes, available)
    if reader_bytes < 64:
        raise BudgetExceeded("Insufficient shared budget for point calculations")
    view = replace(source, read_options=replace(source.read_options, max_batch_bytes=reader_bytes))
    time_column = source.report["time_column"]
    columns = [time_column, current_column] + ([voltage_column] if include_energy else [])
    token = cancellation or CancellationToken()
    initial = fingerprint(source.path)
    reservation = reader_bytes * 16
    budget.reserve(reservation)
    reader = view.iter_batches(columns=columns)
    sums = np.zeros(4 if include_energy else 2)
    compensation = np.zeros_like(sums)
    counts = {
        "rows": 0,
        "intervals": 0,
        "estimated_current_points": 0,
        "estimated_voltage_points": 0,
        "unavailable_capacity_intervals": 0,
        "unavailable_energy_intervals": 0,
    }
    durations = np.zeros(3)  # observed, capacity-covered, energy-covered seconds
    previous_raw = previous_effective = None
    try:
        block = next(reader, None)
        while block is not None:
            token.check()
            following = next(reader, None)
            raw = [_numeric(block, name) for name in columns]
            after = (
                None
                if following is None
                else [_numeric(following.slice(0, 1), name)[0] for name in columns]
            )
            prefix = 0 if previous_raw is None else 1
            padded = [
                np.concatenate(
                    (
                        [] if previous_raw is None else [previous_raw[k]],
                        values,
                        [] if after is None else [after[k]],
                    )
                )
                for k, values in enumerate(raw)
            ]
            signals, masks = [], []
            for values in padded[1:]:
                effective, mask = estimate_isolated(padded[0], values, missing_span_s)
                signals.append(effective[prefix : prefix + block.num_rows])
                masks.append(mask[prefix : prefix + block.num_rows])
            counts["estimated_current_points"] += int(np.count_nonzero(masks[0]))
            if include_energy:
                counts["estimated_voltage_points"] += int(np.count_nonzero(masks[1]))
            effective = [raw[0], *signals]
            interval_arrays = (
                effective
                if previous_effective is None
                else [np.r_[previous_effective[k], values] for k, values in enumerate(effective)]
            )
            increments = directional_integrals(*interval_arrays)
            dt = np.diff(interval_arrays[0])
            counts["rows"] += block.num_rows
            counts["intervals"] += len(dt)
            q_valid = np.isfinite(increments[:, 0]) & np.isfinite(increments[:, 1])
            counts["unavailable_capacity_intervals"] += int(np.count_nonzero(~q_valid))
            durations[0] += np.sum(dt)
            durations[1] += np.sum(dt[q_valid])
            if include_energy:
                e_valid = np.isfinite(increments[:, 2]) & np.isfinite(increments[:, 3])
                counts["unavailable_energy_intervals"] += int(np.count_nonzero(~e_valid))
                durations[2] += np.sum(dt[e_valid])
            subtotal = np.nansum(increments, axis=0)
            combined = sums + subtotal
            compensation += np.where(
                np.abs(sums) >= np.abs(subtotal),
                (sums - combined) + subtotal,
                (subtotal - combined) + sums,
            )
            sums = combined
            previous_raw = [values[-1] for values in raw]
            previous_effective = [values[-1] for values in effective]
            block = following
        token.check()
        if fingerprint(source.path) != initial:
            raise SourceChangedError("Prepared source changed during calculation")
    finally:
        reader.close()
        budget.release(reservation)
    sums += compensation
    q_quality = (
        "partial"
        if counts["unavailable_capacity_intervals"]
        else ("estimated" if counts["estimated_current_points"] else "complete")
    )
    has_capacity = counts["intervals"] > counts["unavailable_capacity_intervals"]
    if not has_capacity:
        q_quality = "unavailable"
    values = {
        "charge_mAh": float(sums[0]),
        "discharge_mAh": float(sums[1]),
        "net_mAh": float(sums[0] - sums[1]),
        "capacity_quality": q_quality,
        "duration_s": float(durations[0]),
        "capacity_covered_s": float(durations[1]),
    }
    if include_energy:
        e_quality = (
            "partial"
            if counts["unavailable_energy_intervals"]
            else (
                "estimated"
                if counts["estimated_current_points"] or counts["estimated_voltage_points"]
                else "complete"
            )
        )
        values.update(
            charge_Wh=float(sums[2]),
            discharge_Wh=float(sums[3]),
            net_Wh=float(sums[2] - sums[3]),
            energy_quality=e_quality,
            energy_covered_s=float(durations[2]),
        )
        if counts["intervals"] == counts["unavailable_energy_intervals"]:
            values.update(
                charge_Wh=None, discharge_Wh=None, net_Wh=None, energy_quality="unavailable"
            )
    if not has_capacity:
        values.update(charge_mAh=None, discharge_mAh=None, net_mAh=None)
    if counts["estimated_current_points"] or counts["estimated_voltage_points"]:
        warnings.warn(
            "Isolated missing measurements estimated; original columns unchanged",
            EstimatedMeasurementWarning,
            stacklevel=2,
        )
    if counts["unavailable_capacity_intervals"] or counts["unavailable_energy_intervals"]:
        warnings.warn(
            "Unavailable measurement intervals; inspect quantity coverage and quality",
            IncompleteIntegrationWarning,
            stacklevel=2,
        )
    return SourceTotals(
        values,
        {
            "source_file": source.report["source_file"],
            "scope": "entire prepared source",
            "measurement_columns": {
                "time": time_column,
                "current": current_column,
                "voltage": voltage_column if include_energy else None,
            },
            "counts": counts,
            "missing_span_s": missing_span_s,
            "units": {"time": "s", "current": "mA", "voltage": "V"},
            "include_energy": include_energy,
            "managed_reservation_bytes": reservation,
            "generated_measurement_identifiers": False,
        },
    )
