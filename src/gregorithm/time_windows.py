"""Exact selected-time Q/E totals using surrounding point-signal support."""

from __future__ import annotations

import math
import warnings

import numpy as np

from .diagnostics import BudgetExceeded, SourceChangedError
from .execution import CancellationToken, MemoryBudget
from .integration import interval_parts
from .io import fingerprint
from .point_calculations import (
    EstimatedMeasurementWarning,
    IncompleteIntegrationWarning,
    SourceTotals,
    _effective_batches,
)


def integrate_time_window(
    source,
    start_s,
    end_s,
    *,
    include_energy=True,
    missing_span_s=10.0,
    memory_budget=None,
    cancellation=None,
):
    """Integrate one exact `[start_s, end_s]` window without rounding source rows.

    Boundaries inside an original interval use the approved linear point model and
    analytic I(V) product integral. The operation does not insert source rows or reset
    the cumulative coordinate exposed by ``data.cumulative()``.
    """
    for name, value in (("start_s", start_s), ("end_s", end_s)):
        if isinstance(value, (bool, np.bool_)) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    start, end = float(start_s), float(end_s)
    if end <= start:
        raise ValueError("end_s must be later than start_s")
    if not isinstance(include_energy, (bool, np.bool_)):
        raise ValueError("include_energy must be boolean")
    if (
        isinstance(missing_span_s, (bool, np.bool_))
        or not math.isfinite(missing_span_s)
        or missing_span_s < 0
    ):
        raise ValueError("missing_span_s must be finite and nonnegative")
    budget = memory_budget or MemoryBudget(2 * 1024**3)
    token = cancellation or CancellationToken()
    initial = fingerprint(source.path)
    cache = 2 * 1024**2
    reader_bytes = min(
        source.read_options.max_batch_bytes, (budget.limit - budget.used - cache) // 16
    )
    if reader_bytes < 64:
        raise BudgetExceeded("Insufficient exact-window integration budget")
    reservation = reader_bytes * 16 + cache
    budget.reserve(reservation)
    columns = [source.report["time_column"], "current_mA"]
    if include_energy:
        columns.append("voltage_V")
    sums = np.zeros(4, dtype=np.float64)
    correction = np.zeros(4, dtype=np.float64)
    counts = {
        "intervals": 0,
        "unavailable_capacity_intervals": 0,
        "unavailable_energy_intervals": 0,
        "estimated_current_points": 0,
        "estimated_voltage_points": 0,
    }
    total_duration = q_covered = e_covered = 0.0
    found_start = found_end = False
    previous = None
    reader = _effective_batches(source, columns, missing_span_s, token)
    try:
        for times, _, current, voltage, estimate_i, estimate_v in reader:
            for position in range(len(times)):
                point = (
                    times[position],
                    current[position],
                    None if voltage is None else voltage[position],
                    estimate_i[position],
                    False if estimate_v is None else estimate_v[position],
                )
                if previous is None:
                    previous = point
                    continue
                t0, i0, v0, ei0, ev0 = previous
                t1, i1, v1, ei1, ev1 = point
                previous = point
                if t1 < start or t0 > end:
                    continue
                overlap_start, overlap_end = max(t0, start), min(t1, end)
                if overlap_end <= overlap_start:
                    continue
                found_start |= t0 <= start <= t1
                found_end |= t0 <= end <= t1
                lo = (overlap_start - t0) / (t1 - t0)
                hi = (overlap_end - t0) / (t1 - t0)
                qc, qd, ec, ed = interval_parts(
                    t0,
                    i0,
                    np.nan if v0 is None else v0,
                    t1,
                    i1,
                    np.nan if v1 is None else v1,
                    lo=lo,
                    hi=hi,
                )
                increment = np.array([qc, qd, ec, ed])
                duration = overlap_end - overlap_start
                counts["intervals"] += 1
                total_duration += duration
                if np.all(np.isfinite(increment[:2])):
                    q_covered += duration
                    counts["estimated_current_points"] += int(ei0) + int(ei1)
                    for k in range(2):
                        adjusted = increment[k] - correction[k]
                        combined = sums[k] + adjusted
                        correction[k] = (combined - sums[k]) - adjusted
                        sums[k] = combined
                else:
                    counts["unavailable_capacity_intervals"] += 1
                if include_energy:
                    if np.all(np.isfinite(increment[2:])):
                        e_covered += duration
                        counts["estimated_voltage_points"] += int(ev0) + int(ev1)
                        for k in range(2, 4):
                            adjusted = increment[k] - correction[k]
                            combined = sums[k] + adjusted
                            correction[k] = (combined - sums[k]) - adjusted
                            sums[k] = combined
                    else:
                        counts["unavailable_energy_intervals"] += 1
        if not found_start or not found_end:
            raise ValueError("Exact time window must lie inside observed source support")
        token.check()
        if fingerprint(source.path) != initial:
            raise SourceChangedError("Prepared source changed during exact-window integration")
    finally:
        reader.close()
        budget.release(reservation)
    q_invalid = counts["unavailable_capacity_intervals"]
    e_invalid = counts["unavailable_energy_intervals"]
    q_quality = (
        "unavailable"
        if q_invalid == counts["intervals"]
        else "partial"
        if q_invalid
        else "estimated"
        if counts["estimated_current_points"]
        else "complete"
    )
    e_quality = (
        "unavailable"
        if not include_energy or e_invalid == counts["intervals"]
        else "partial"
        if e_invalid
        else "estimated"
        if counts["estimated_current_points"] or counts["estimated_voltage_points"]
        else "complete"
    )
    values = {
        "charge_mAh": None if q_quality == "unavailable" else float(sums[0]),
        "discharge_mAh": None if q_quality == "unavailable" else float(sums[1]),
        "net_mAh": None if q_quality == "unavailable" else float(sums[0] - sums[1]),
        "capacity_quality": q_quality,
        "duration_s": total_duration,
        "capacity_covered_s": q_covered,
    }
    if include_energy:
        values.update(
            charge_Wh=None if e_quality == "unavailable" else float(sums[2]),
            discharge_Wh=None if e_quality == "unavailable" else float(sums[3]),
            net_Wh=None if e_quality == "unavailable" else float(sums[2] - sums[3]),
            energy_quality=e_quality,
            energy_covered_s=e_covered,
        )
    if counts["estimated_current_points"] or counts["estimated_voltage_points"]:
        warnings.warn(
            "Exact-window Q/E uses approved isolated-point estimates",
            EstimatedMeasurementWarning,
            stacklevel=2,
        )
    if q_invalid or e_invalid:
        warnings.warn(
            "Exact window contains unavailable Q/E intervals",
            IncompleteIntegrationWarning,
            stacklevel=2,
        )
    return SourceTotals(
        values,
        {
            "source_file": source.report["source_file"],
            "scope": "exact selected time window",
            "start_s": start,
            "end_s": end,
            "boundary_policy": "linear_point_signals_without_source_rows",
            "counts": counts,
            "missing_span_s": float(missing_span_s),
            "include_energy": bool(include_energy),
            "generated_measurement_identifiers": False,
        },
    )
