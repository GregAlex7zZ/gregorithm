"""Data elaboration: half-cycles, cycles, step types and cumulative quantities.

`elaborate` streams a canonical import into an immutable generation containing:

- ``points/elaborated.parquet``: every retained point with `time_s`, `current_a`, `voltage_v`,
  the four cumulative quantities (`charge_capacity_ah`, `discharge_capacity_ah`,
  `charge_energy_wh`, `discharge_energy_wh`, reset at each half-cycle), `cycle`, `half_cycle`,
  `rest`, `step` (see `Step`) and `source_row`;
- ``indexes/half_cycles.parquet`` and ``indexes/segments.parquet``: compact half-cycle totals
  and contiguous step segments used by downstream analyses;
- ``overview/level-<width>.parquet``: real measured rows (first, last and extrema per bucket of
  64, 4096 and 262144 points) for fast plotting;
- ``reports/elaboration.json``: counts and the cycle order used.

Rules in brief. A point is rest when ``|I| <= rest_current_threshold_a`` (default 0 A). The first
non-rest point starts half-cycle 1; a new half-cycle starts only after three consecutive
opposite-sign points (rest resets the count), so brief sign excursions stay in the current half-
cycle. Half-cycles are paired into cycles in the selected `CycleOrder`; leading rest is cycle 0.
Steps are classified within each half-cycle: CC when the current range of at least three
consecutive points is within `cc_relative_tolerance` of the absolute mean current; CV when the
voltage range is within `cv_absolute_tolerance_v` and the current departs from the preceding CC
level (CV then lasts until the next rest or half-cycle); otherwise Other. Capacity and energy are
trapezoidal integrals of current and of ``V * I``; an interval whose current changes sign is
split at the zero crossing. Negative measured voltage makes the energy of that half-cycle
unavailable (null) while capacity remains available.

The pure functions `segment_half_cycles`, `classify_steps`, `integrate_cumulatives`,
`elaborate_arrays` and `overview_indices` apply the same rules to in-memory NumPy arrays and are
convenient for small datasets and verification.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.dataset as ds  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from . import __version__
from .contracts import DependencyRef, OperationId, ResultRef
from .identity import hash_dependencies, hash_parameters
from .progress import CancelToken, ProgressEvent, ProgressSink, check_cancelled, report_progress
from .storage import ResultStore

PARQUET_ROW_GROUP_SIZE = 262_144


class _BoundedParquetSink:
    """Parquet writer with a fixed row-group size for bounded input tables."""

    def __init__(
        self,
        path: Path,
        schema: pa.Schema,
        row_limit: int,
        *,
        baseline_row_group_size: bool = True,
    ) -> None:
        """Open a Zstandard-compressed Parquet writer at `path`."""
        self.path = path
        self.schema = schema
        self.row_limit = row_limit
        self.baseline_row_group_size = baseline_row_group_size
        self.writer = pq.ParquetWriter(path, schema, compression="zstd")

    def write_table(self, table: pa.Table) -> None:
        """Write one table using the configured row-group size."""
        if self.baseline_row_group_size:
            self.writer.write_table(table, row_group_size=self.row_limit)
        else:
            self.writer.write_table(table)

    def close(self) -> None:
        """Close the underlying Parquet writer."""
        self.writer.close()


class _BufferedStagingSink(_BoundedParquetSink):
    """`_BoundedParquetSink` for scratch files: collects small tables, writes large row groups.

    The S4/S5 stages emit one small table per classified region, often thousands per
    elaboration; writing each one made a separate Parquet row group and the writes dominated
    the elaboration time. Tables are held until ``row_limit`` rows are pending (about 5 MB for
    65,536 elaborated rows) and then written together. Only scratch files that are read back
    with ``iter_batches`` use it: rows, order and values are unchanged, and published files keep
    their exact layout and hashes (only the scratch sizes in the report shrink). Pending rows
    are written by `close`.
    """

    def __init__(
        self,
        path: Path,
        schema: pa.Schema,
        row_limit: int,
        *,
        baseline_row_group_size: bool = True,
    ) -> None:
        """Open the writer; see `_BoundedParquetSink`."""
        super().__init__(path, schema, row_limit, baseline_row_group_size=baseline_row_group_size)
        self._pending: list[pa.Table] = []
        self._pending_rows = 0

    def write_table(self, table: pa.Table) -> None:
        """Queue one table and write the queue when it holds at least ``row_limit`` rows."""
        self._pending.append(table)
        self._pending_rows += table.num_rows
        if self._pending_rows >= self.row_limit:
            self._flush()

    def _flush(self) -> None:
        """Write every queued table as consecutive rows."""
        if self._pending:
            merged = pa.concat_tables(self._pending)
            self._pending, self._pending_rows = [], 0
            self.writer.write_table(merged, row_group_size=self.row_limit)

    def close(self) -> None:
        """Write the queued rows, then close the writer."""
        try:
            self._flush()
        finally:
            super().close()


OVERVIEW_WIDTHS = (64, 4_096, 262_144)
HALF_INDEX_SCHEMA = pa.schema(
    [
        ("half_cycle", pa.int32()),
        ("cycle", pa.int32()),
        ("first_row", pa.uint64()),
        ("last_row", pa.uint64()),
        ("first_direction", pa.int8()),
        ("charge_capacity_ah", pa.float64()),
        ("discharge_capacity_ah", pa.float64()),
        ("charge_energy_wh", pa.float64()),
        ("discharge_energy_wh", pa.float64()),
    ]
)
SEGMENT_INDEX_SCHEMA = pa.schema(
    [
        ("segment_id", pa.uint64()),
        ("first_row", pa.uint64()),
        ("last_row", pa.uint64()),
        ("first_time_s", pa.float64()),
        ("last_time_s", pa.float64()),
        ("cycle", pa.int32()),
        ("half_cycle", pa.int32()),
        ("step", pa.int8()),
        ("s4_portion", pa.int64()),
        ("polarity", pa.int8()),
    ]
)


class CycleOrder(StrEnum):
    """Order in which half-cycles are paired into a cycle.

    `CHARGE_DISCHARGE` (default) pairs a charge with the following discharge; `DISCHARGE_CHARGE`
    pairs a discharge with the following charge. The order changes cycle numbers and which phase
    is "first" for efficiencies and C-rate, never the measured data.
    """

    CHARGE_DISCHARGE = "charge_discharge"
    DISCHARGE_CHARGE = "discharge_charge"


class Step(IntEnum):
    """Step type stored in the `step` column: 0 rest, 1 CC, 2 CV, 3 other."""

    REST = 0
    CC = 1
    CV = 2
    OTHER = 3


@dataclass(frozen=True, slots=True)
class ElaborationParameters:
    """Settings of the data elaboration.

    Attributes:
        rest_current_threshold_a: Points with ``|I| <= threshold`` (A) are rest. Default 0.
        cycle_order: `CycleOrder` used to pair half-cycles into cycles.
        cc_relative_tolerance: CC criterion ``max(I) - min(I) <= tol * |mean(I)|``. Default 0.01.
        cv_absolute_tolerance_v: CV criterion ``max(V) - min(V) <= tol`` in volts. Default 0.001.

    CC and CV segments need at least three consecutive points. All values must be finite and non-
    negative.
    """

    rest_current_threshold_a: float = 0.0
    cycle_order: CycleOrder = CycleOrder.CHARGE_DISCHARGE
    cc_relative_tolerance: float = 0.01
    cv_absolute_tolerance_v: float = 0.001

    def validate(self) -> None:
        """Raise ValueError for negative or non-finite settings."""
        for name, value in (
            ("rest_current_threshold_a", self.rest_current_threshold_a),
            ("cc_relative_tolerance", self.cc_relative_tolerance),
            ("cv_absolute_tolerance_v", self.cv_absolute_tolerance_v),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    def as_dict(self) -> dict[str, Any]:
        """Return the parameters, including fixed settings, as a dict."""
        return {
            "rest_current_threshold_a": self.rest_current_threshold_a,
            "cycle_order": self.cycle_order.value,
            "cc_relative_tolerance": self.cc_relative_tolerance,
            "cv_absolute_tolerance_v": self.cv_absolute_tolerance_v,
            "step_min_samples": 3,
            "overview_widths": list(OVERVIEW_WIDTHS),
        }


_DEFAULT_ELABORATION_PARAMETERS = ElaborationParameters()


@dataclass(frozen=True, slots=True)
class ElaborationRequest:
    """Parameters of `elaborate`.

    Attributes:
        canonical_parquet: ``points/canonical.parquet`` of an import result.
        canonical_content_hash: `ImportReport.content_hash` of that import.
        caller_root, sample_id, job_id, generation_id: Destination in the result store.
        parameters: `ElaborationParameters`.
        batch_size: Rows per streaming batch.
    """

    canonical_parquet: Path
    canonical_content_hash: str
    caller_root: Path
    sample_id: str
    job_id: str
    generation_id: str
    parameters: ElaborationParameters = ElaborationParameters()
    batch_size: int = 65_536

    def validate(self) -> None:
        """Validate parameters, batch size and the content-hash length."""
        self.parameters.validate()
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len(self.canonical_content_hash) != 64:
            raise ValueError("canonical_content_hash must be SHA-256 hex")


@dataclass(frozen=True, slots=True)
class ElaborationResult:
    """Published elaboration result.

    Attributes:
        ref: Reference to the elaboration generation.
        rows: Number of points.
        half_cycles: Number of active half-cycles (excluding leading rest).
        cycles: Number of positive cycles.
        fingerprint: Result fingerprint; pass it to downstream requests
            (Capacity, DCIR, dQ/dV, cycle correction).
    """

    ref: ResultRef
    rows: int
    half_cycles: int
    cycles: int
    fingerprint: str


@dataclass(frozen=True, slots=True)
class WindowRequest:
    """Parameters of `read_window`.

    Attributes:
        result_path: Elaboration (or corrected elaboration) generation directory.
        start_time_s, end_time_s: Optional inclusive time bounds in seconds.
        cycles: Optional cycles to include; empty means all.
        columns: Columns to return; empty means all.
        point_budget: Optional maximum number of rows to return.
    """

    result_path: Path
    start_time_s: float | None = None
    end_time_s: float | None = None
    cycles: tuple[int, ...] = ()
    columns: tuple[str, ...] = ()
    point_budget: int | None = None


@dataclass(frozen=True, slots=True)
class ElaboratedArrays:
    """In-memory result of `elaborate_arrays`.

    Per-point arrays `cycle`, `half_cycle`, `rest`, `step` and the four cumulative quantities,
    plus `first_direction` (+1 if the first half-cycle is a charge, -1 if a discharge, 0 if the
    data contain only rest). Energy is NaN where it is unavailable.
    """

    cycle: np.ndarray
    half_cycle: np.ndarray
    rest: np.ndarray
    step: np.ndarray
    charge_capacity_ah: np.ndarray
    discharge_capacity_ah: np.ndarray
    charge_energy_wh: np.ndarray
    discharge_energy_wh: np.ndarray
    first_direction: int


def segment_half_cycles(
    current: np.ndarray, threshold: float, order: CycleOrder
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Label half-cycles and cycles from a current array (pure function).

    Args:
        current: Current in A, positive for charge.
        threshold: Rest threshold in A (finite, >= 0).
        order: `CycleOrder` used to number cycles.

    Returns:
        ``(cycle, half_cycle, rest, first_direction)``. Leading rest has cycle and half-cycle 0.
        A direction change is confirmed by three consecutive opposite-sign points; shorter
        excursions keep the previous labels and rest resets the confirmation count.
    """
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("rest threshold must be finite and nonnegative")
    current = np.asarray(current, dtype=np.float64)
    n = len(current)
    rest = np.abs(current) <= threshold
    signs = np.zeros(n, dtype=np.int8)
    signs[current > threshold] = 1
    signs[current < -threshold] = -1
    half = np.zeros(n, dtype=np.int32)
    if n == 0 or np.all(rest):
        return half.copy(), half, rest, 0

    starts = np.r_[0, np.flatnonzero(signs[1:] != signs[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    direction = 0
    half_id = 0
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        sign = int(signs[start])
        if sign == 0:
            if direction:
                half[start:end] = half_id
            continue
        if direction == 0:
            direction = sign
            half_id = 1
        elif sign == -direction and end - start >= 3:
            direction = sign
            half_id += 1
        half[start:end] = half_id

    first_direction = int(signs[np.flatnonzero(signs)[0]])
    active = half > 0
    cycles = np.zeros(n, dtype=np.int32)
    if first_direction == (1 if order is CycleOrder.CHARGE_DISCHARGE else -1):
        cycles[active] = (half[active] + 1) // 2
    else:
        cycles[active] = 1 + half[active] // 2
    return cycles, half, rest, first_direction


def _cc_ok(values: np.ndarray, tolerance: float) -> np.ndarray:
    """Return, for each three-point window, whether it satisfies the CC criterion."""
    if len(values) < 3:
        return np.empty(0, dtype=bool)
    windows = np.lib.stride_tricks.sliding_window_view(values, 3)
    return np.ptp(windows, axis=1) <= tolerance * np.abs(np.mean(windows, axis=1))


def _cv_ok(
    current: np.ndarray,
    voltage: np.ndarray,
    cc_tolerance: float,
    cv_tolerance: float,
    cc_reference: float | None,
) -> np.ndarray:
    """Return, for each three-point window, whether it qualifies as CV onset."""
    if len(current) < 3:
        return np.empty(0, dtype=bool)
    iw = np.lib.stride_tricks.sliding_window_view(current, 3)
    vw = np.lib.stride_tricks.sliding_window_view(voltage, 3)
    stable_v = np.ptp(vw, axis=1) <= cv_tolerance
    if cc_reference is None:
        unstable_i = np.ptp(iw, axis=1) > cc_tolerance * np.abs(np.mean(iw, axis=1))
    else:
        unstable_i = np.all(np.abs(iw - cc_reference) > cc_tolerance * abs(cc_reference), axis=1)
    return stable_v & unstable_i


def _classify_region(
    current: np.ndarray, voltage: np.ndarray, cc_tolerance: float, cv_tolerance: float
) -> np.ndarray:
    """Classify the steps of one uninterrupted non-rest half-cycle region."""
    labels = np.full(len(current), Step.OTHER, dtype=np.int8)
    position = 0
    cc_reference: float | None = None
    while position + 3 <= len(current):
        suffix_i = current[position:]
        suffix_v = voltage[position:]
        cc_candidates = _cc_ok(suffix_i, cc_tolerance)
        cv_candidates = _cv_ok(suffix_i, suffix_v, cc_tolerance, cv_tolerance, cc_reference)
        qualifying = cc_candidates | cv_candidates
        if not np.any(qualifying):
            break
        offset = int(np.flatnonzero(qualifying)[0])
        start = position + offset
        if cc_candidates[offset]:
            segment = current[start:]
            count = np.arange(1, len(segment) + 1, dtype=np.float64)
            means = np.cumsum(segment, dtype=np.float64) / count
            stable = np.maximum.accumulate(segment) - np.minimum.accumulate(
                segment
            ) <= cc_tolerance * np.abs(means)
            violations = np.flatnonzero(~stable[2:])
            end = start + (int(violations[0]) + 2 if len(violations) else len(segment))
            labels[start:end] = Step.CC
            cc_reference = float(np.mean(current[start:end]))
            position = end
        else:
            labels[start:] = Step.CV
            break
    return labels


def classify_steps(
    current: np.ndarray,
    voltage: np.ndarray,
    half_cycle: np.ndarray,
    rest: np.ndarray,
    cc_tolerance: float,
    cv_tolerance: float,
) -> np.ndarray:
    """Classify each point as rest, CC, CV or other (pure function).

    Classification restarts at every rest point and half-cycle change. Within a region, windows
    of three points are searched in time order; CC has priority over CV, a CC segment grows while
    the whole segment stays within tolerance, and confirmed CV lasts until the region ends.
    Returns an int8 array of `Step` values.
    """
    for value in (cc_tolerance, cv_tolerance):
        if not math.isfinite(value) or value < 0:
            raise ValueError("CC/CV tolerances must be finite and nonnegative")
    n = len(current)
    labels = np.full(n, Step.OTHER, dtype=np.int8)
    labels[rest] = Step.REST
    eligible = ~rest
    if not np.any(eligible):
        return labels
    boundaries = np.r_[
        0,
        np.flatnonzero((eligible[1:] != eligible[:-1]) | (half_cycle[1:] != half_cycle[:-1])) + 1,
        n,
    ]
    for start, end in zip(boundaries[:-1].tolist(), boundaries[1:].tolist(), strict=True):
        if eligible[start]:
            labels[start:end] = _classify_region(
                current[start:end], voltage[start:end], cc_tolerance, cv_tolerance
            )
    return labels


def _interval_contributions(
    time: np.ndarray, current: np.ndarray, voltage: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split each interval's capacity/energy into charge and discharge parts."""
    dt = np.diff(time)
    i0, i1 = current[:-1], current[1:]
    v0, v1 = voltage[:-1], voltage[1:]
    cq = np.zeros(len(dt), dtype=np.float64)
    dq = np.zeros(len(dt), dtype=np.float64)
    ce = np.zeros(len(dt), dtype=np.float64)
    de = np.zeros(len(dt), dtype=np.float64)
    crossing = (i0 * i1) < 0
    noncrossing = ~crossing
    q = dt * (i0 + i1) / 7200.0
    e = dt * (v0 * i0 + v1 * i1) / 7200.0
    positive = noncrossing & (q >= 0)
    negative = noncrossing & (q < 0)
    cq[positive], dq[negative] = q[positive], -q[negative]
    ce[positive], de[negative] = e[positive], -e[negative]
    if np.any(crossing):
        alpha = np.abs(i0[crossing]) / (np.abs(i0[crossing]) + np.abs(i1[crossing]))
        left_q = alpha * dt[crossing] * np.abs(i0[crossing]) / 7200.0
        right_q = (1.0 - alpha) * dt[crossing] * np.abs(i1[crossing]) / 7200.0
        left_e = alpha * dt[crossing] * np.abs(v0[crossing] * i0[crossing]) / 7200.0
        right_e = (1.0 - alpha) * dt[crossing] * np.abs(v1[crossing] * i1[crossing]) / 7200.0
        indices = np.flatnonzero(crossing)
        left_positive = i0[crossing] > 0
        cq[indices[left_positive]] = left_q[left_positive]
        ce[indices[left_positive]] = left_e[left_positive]
        dq[indices[~left_positive]] = left_q[~left_positive]
        de[indices[~left_positive]] = left_e[~left_positive]
        right_positive = i1[crossing] > 0
        cq[indices[right_positive]] += right_q[right_positive]
        ce[indices[right_positive]] += right_e[right_positive]
        dq[indices[~right_positive]] += right_q[~right_positive]
        de[indices[~right_positive]] += right_e[~right_positive]
    return cq, dq, ce, de


def _group_cumulative(increments: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Cumulative sum of interval increments, restarted for each group."""
    output = np.zeros(len(groups), dtype=np.float64)
    if len(groups) < 2:
        return output
    point_increment = np.r_[0.0, increments]
    cumulative = np.cumsum(point_increment)
    starts = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1]
    baselines = cumulative[starts] - point_increment[starts]
    offsets = np.repeat(baselines, np.diff(np.r_[starts, len(groups)]))
    output[:] = cumulative - offsets
    return output


def _group_point_cumulative(increments: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Cumulative sum of per-point increments, restarted for each group."""
    if len(groups) == 0:
        return np.empty(0, dtype=np.float64)
    cumulative = np.cumsum(increments, dtype=np.longdouble)
    starts = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1]
    baselines = cumulative[starts] - increments[starts]
    offsets = np.repeat(baselines, np.diff(np.r_[starts, len(groups)]))
    return np.asarray(cumulative - offsets, dtype=np.float64)


def integrate_cumulatives(
    time: np.ndarray,
    current: np.ndarray,
    voltage: np.ndarray,
    half_cycle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, tuple[float, ...]]]:
    """Integrate capacity and energy per half-cycle (pure function).

    Returns the per-point cumulative charge/discharge capacity (Ah) and energy (Wh), reset at
    each half-cycle, and a dict mapping each half-cycle to its final totals including any zero-
    crossing contribution at its end. Energy is NaN for half-cycles with negative voltage.
    """
    n = len(time)
    if not (len(current) == len(voltage) == len(half_cycle) == n):
        raise ValueError("Integration arrays must have equal length")
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty.copy(), empty.copy(), empty.copy(), {}
    cq_i, dq_i, ce_i, de_i = _interval_contributions(time, current, voltage)
    boundary = half_cycle[1:] != half_cycle[:-1]
    crossing = current[:-1] * current[1:] < 0

    # At a crossing boundary, only the side ending at the new real point contributes
    # to the new point cumulative. The prior side is retained in compact totals.
    cq_point, dq_point, ce_point, de_point = (value.copy() for value in (cq_i, dq_i, ce_i, de_i))
    boundary_cross = boundary & crossing
    if np.any(boundary_cross):
        indices = np.flatnonzero(boundary_cross)
        alpha = np.abs(current[:-1][boundary_cross]) / (
            np.abs(current[:-1][boundary_cross]) + np.abs(current[1:][boundary_cross])
        )
        right_dt = (1.0 - alpha) * np.diff(time)[boundary_cross]
        right_q = right_dt * np.abs(current[1:][boundary_cross]) / 7200.0
        right_e = (
            right_dt * np.abs(voltage[1:][boundary_cross] * current[1:][boundary_cross]) / 7200.0
        )
        cq_point[indices] = np.where(current[1:][boundary_cross] > 0, right_q, 0.0)
        dq_point[indices] = np.where(current[1:][boundary_cross] < 0, right_q, 0.0)
        ce_point[indices] = np.where(current[1:][boundary_cross] > 0, right_e, 0.0)
        de_point[indices] = np.where(current[1:][boundary_cross] < 0, right_e, 0.0)

    cq = _group_cumulative(cq_point, half_cycle)
    dq = _group_cumulative(dq_point, half_cycle)
    ce = _group_cumulative(ce_point, half_cycle)
    de = _group_cumulative(de_point, half_cycle)

    negative_groups = np.unique(half_cycle[voltage < 0])
    invalid_energy = np.isin(half_cycle, negative_groups)
    ce[invalid_energy] = np.nan
    de[invalid_energy] = np.nan

    totals: dict[int, tuple[float, ...]] = {}
    groups = np.unique(half_cycle)
    for group in groups.tolist():
        rows = np.flatnonzero(half_cycle == group)
        total_cq = float(cq[rows[-1]])
        total_dq = float(dq[rows[-1]])
        total_ce = float(ce[rows[-1]])
        total_de = float(de[rows[-1]])
        outgoing = (half_cycle[:-1] == group) & boundary_cross
        if np.any(outgoing):
            alpha = np.abs(current[:-1][outgoing]) / (
                np.abs(current[:-1][outgoing]) + np.abs(current[1:][outgoing])
            )
            left_dt = alpha * np.diff(time)[outgoing]
            left_q = left_dt * np.abs(current[:-1][outgoing]) / 7200.0
            left_e = left_dt * np.abs(voltage[:-1][outgoing] * current[:-1][outgoing]) / 7200.0
            total_cq += float(np.sum(left_q[current[:-1][outgoing] > 0]))
            total_dq += float(np.sum(left_q[current[:-1][outgoing] < 0]))
            total_ce += float(np.sum(left_e[current[:-1][outgoing] > 0]))
            total_de += float(np.sum(left_e[current[:-1][outgoing] < 0]))
        if group in negative_groups:
            total_ce = total_de = math.nan
        totals[int(group)] = (total_cq, total_dq, total_ce, total_de)
    return cq, dq, ce, de, totals


def elaborate_arrays(
    time: np.ndarray,
    current: np.ndarray,
    voltage: np.ndarray,
    parameters: ElaborationParameters = _DEFAULT_ELABORATION_PARAMETERS,
) -> ElaboratedArrays:
    """Apply segmentation, step classification and integration in memory.

    Convenience wrapper around `segment_half_cycles`, `classify_steps` and
    `integrate_cumulatives` for arrays that fit in memory. Use `elaborate` for files.
    """
    parameters.validate()
    cycle, half, rest, first_direction = segment_half_cycles(
        current, parameters.rest_current_threshold_a, parameters.cycle_order
    )
    step = classify_steps(
        current,
        voltage,
        half,
        rest,
        parameters.cc_relative_tolerance,
        parameters.cv_absolute_tolerance_v,
    )
    cq, dq, ce, de, _ = integrate_cumulatives(time, current, voltage, half)
    return ElaboratedArrays(cycle, half, rest, step, cq, dq, ce, de, first_direction)


def overview_indices(columns: tuple[np.ndarray, ...], width: int) -> np.ndarray:
    """Return indices of the first, last, minimum and maximum rows per bucket.

    Each bucket covers `width` consecutive rows; for every array in `columns` the earliest
    minimum and maximum are kept. Returned indices are sorted and unique, so a plot built from
    them shows only real measured points while preserving visible extremes.
    """
    if width < 1:
        raise ValueError("overview width must be positive")
    n = len(columns[0]) if columns else 0
    if any(len(column) != n for column in columns):
        raise ValueError("overview columns must align")
    if n == 0:
        return np.empty(0, dtype=np.uint64)
    complete = n // width
    starts = np.arange(complete, dtype=np.uint64) * width
    ends = starts + width - 1
    selected = [starts, ends]
    for column in columns:
        if complete:
            blocks = column[: complete * width].reshape(complete, width)
            selected.append(starts + np.argmin(blocks, axis=1).astype(np.uint64))
            selected.append(starts + np.argmax(blocks, axis=1).astype(np.uint64))
    partial_start = complete * width
    if partial_start < n:
        selected.extend(
            (
                np.array([partial_start], dtype=np.uint64),
                np.array([n - 1], dtype=np.uint64),
            )
        )
        for column in columns:
            partial = column[partial_start:]
            selected.append(np.array([partial_start + int(np.argmin(partial))], dtype=np.uint64))
            selected.append(np.array([partial_start + int(np.argmax(partial))], dtype=np.uint64))
    return np.unique(np.concatenate(selected))


def _table(
    time: np.ndarray,
    current: np.ndarray,
    voltage: np.ndarray,
    values: ElaboratedArrays,
) -> pa.Table:
    """Build the elaborated point table from arrays (in-memory helper)."""
    return pa.table(
        {
            "time_s": pa.array(time, type=pa.float64()),
            "current_a": pa.array(current, type=pa.float64()),
            "voltage_v": pa.array(voltage, type=pa.float64()),
            "charge_capacity_ah": pa.array(values.charge_capacity_ah, type=pa.float64()),
            "discharge_capacity_ah": pa.array(values.discharge_capacity_ah, type=pa.float64()),
            "charge_energy_wh": pa.array(
                values.charge_energy_wh, type=pa.float64(), from_pandas=True
            ),
            "discharge_energy_wh": pa.array(
                values.discharge_energy_wh, type=pa.float64(), from_pandas=True
            ),
            "cycle": pa.array(values.cycle, type=pa.int32()),
            "half_cycle": pa.array(values.half_cycle, type=pa.int32()),
            "rest": pa.array(values.rest, type=pa.bool_()),
            "step": pa.array(values.step, type=pa.int8()),
            "source_row": pa.array(np.arange(len(time), dtype=np.uint64), type=pa.uint64()),
        }
    )


def _fingerprint(table: pa.Table, parameters: ElaborationParameters) -> str:
    """Hash a table's Arrow buffers together with the elaboration parameters."""
    digest = hashlib.sha256(b"gregorithm-elaboration-v1\0")
    digest.update(hash_parameters(parameters.as_dict()).encode("ascii"))
    for batch in table.to_batches(max_chunksize=65_536):
        for column in batch.columns:
            for buffer in column.buffers():
                if buffer is not None:
                    digest.update(buffer.to_pybytes())
    return digest.hexdigest()


def _result_fingerprint(request: ElaborationRequest) -> str:
    """Return the elaboration fingerprint from version, parameters and input."""
    dependency = DependencyRef("canonical", request.canonical_content_hash)
    digest = hashlib.sha256(b"gregorithm-elaboration-result-v2\0")
    digest.update(__version__.encode("utf-8"))
    digest.update(b"\0")
    digest.update(hash_parameters(request.parameters.as_dict()).encode("ascii"))
    digest.update(b"\0")
    digest.update(hash_dependencies((dependency,)).encode("ascii"))
    return digest.hexdigest()


def _cycle_values(half: np.ndarray, first_direction: int, order: CycleOrder) -> np.ndarray:
    """Convert half-cycle numbers to cycle numbers for a first direction and order."""
    cycles = np.zeros(len(half), dtype=np.int32)
    active = half > 0
    if first_direction == (1 if order is CycleOrder.CHARGE_DISCHARGE else -1):
        cycles[active] = (half[active] + 1) // 2
    else:
        cycles[active] = 1 + half[active] // 2
    return cycles


class _S3Stream:
    """Streaming half-cycle segmentation holding at most two undecided rows."""

    def __init__(self, threshold: float, order: CycleOrder) -> None:
        """Create the segmentation state for a threshold and cycle order."""
        self.threshold = threshold
        self.order = order
        self.direction = 0
        self.first_direction = 0
        self.half_id = 0
        self.carry: pa.Table | None = None
        self.run_count = 0

    def feed(self, table: pa.Table, *, final: bool = False) -> pa.Table | None:
        """Label a batch; rows of an undecided short sign run are held back."""
        if self.carry is not None:
            table = pa.concat_tables((self.carry, table))
            self.carry = None
        if table.num_rows == 0:
            return None
        current = table["current_a"].combine_chunks().to_numpy(zero_copy_only=False)
        signs = np.zeros(len(current), dtype=np.int8)
        signs[current > self.threshold] = 1
        signs[current < -self.threshold] = -1
        starts = np.r_[0, np.flatnonzero(signs[1:] != signs[:-1]) + 1]
        ends = np.r_[starts[1:], len(signs)]
        run_signs = signs[starts]
        lengths = ends - starts
        self.run_count += len(starts)

        qualifier = (run_signs != 0) & (lengths >= 3)
        if self.direction == 0:
            active_runs = np.flatnonzero(run_signs != 0)
            if len(active_runs):
                qualifier[active_runs[0]] = True
        qrows = np.flatnonzero(qualifier)
        qsigns = run_signs[qrows]
        previous = np.r_[self.direction, qsigns[:-1]] if len(qsigns) else np.empty(0)
        changes = qsigns != previous
        events = np.zeros(len(signs), dtype=np.int32)
        events[starts[qrows[changes]]] = 1
        half = self.half_id + np.cumsum(events, dtype=np.int32)

        if len(qsigns):
            changed_signs = qsigns[changes]
            if len(changed_signs):
                self.direction = int(changed_signs[-1])
                if self.first_direction == 0:
                    self.first_direction = int(changed_signs[0])

        hold = 0
        if not final and self.direction and run_signs[-1] == -self.direction and lengths[-1] < 3:
            hold = int(lengths[-1])
            self.carry = table.slice(len(signs) - hold, hold)
        emit = len(signs) - hold
        if emit == 0:
            return None
        self.half_id = int(half[emit - 1])
        rest = signs[:emit] == 0
        cycles = _cycle_values(half[:emit], self.first_direction, self.order)
        base = table.slice(0, emit)
        return (
            base.append_column("cycle", pa.array(cycles))
            .append_column("half_cycle", pa.array(half[:emit]))
            .append_column("rest", pa.array(rest))
        )

    def finish(self) -> pa.Table | None:
        """Label any held-back rows at the end of the data."""
        if self.carry is None:
            return None
        carry, self.carry = self.carry, None
        return self.feed(carry, final=True)


class _S4Stream:
    """Streaming step classification holding at most two undecided rows."""

    def __init__(self, cc_tolerance: float, cv_tolerance: float) -> None:
        """Create the classification state for the CC and CV tolerances."""
        self.cc_tolerance = cc_tolerance
        self.cv_tolerance = cv_tolerance
        self.mode = "search"
        self.cc_reference: float | None = None
        self.cc_count = 0
        self.cc_sum = 0.0
        self.cc_min = math.inf
        self.cc_max = -math.inf
        self.pending: pa.Table | None = None
        self.region_key: tuple[int, bool] | None = None
        self.transition_count = 0
        self.portion_id = -1

    def _label(self, table: pa.Table, value: Step, *, new_portion: bool = False) -> pa.Table:
        """Append `step` and CC-portion columns to a table."""
        if value is Step.CC:
            if new_portion:
                self.portion_id += 1
            portions = np.full(table.num_rows, self.portion_id, np.int64)
        else:
            portions = np.full(table.num_rows, -1, np.int64)
        return table.append_column(
            "step", pa.array(np.full(table.num_rows, value, np.int8))
        ).append_column("s4_portion", pa.array(portions))

    def _reset_region(self) -> list[pa.Table]:
        """Flush pending rows as Other and reset state for a new region."""
        output: list[pa.Table] = []
        if self.pending is not None:
            output.append(self._label(self.pending, Step.OTHER))
            self.pending = None
        self.mode = "search"
        self.cc_reference = None
        self.cc_count = 0
        self.cc_sum = 0.0
        self.cc_min = math.inf
        self.cc_max = -math.inf
        return output

    def _active(self, table: pa.Table, final: bool = False) -> list[pa.Table]:
        """Classify rows of the current non-rest region."""
        output: list[pa.Table] = []
        if self.pending is not None:
            table = pa.concat_tables((self.pending, table))
            self.pending = None
        while table.num_rows:
            current = table["current_a"].combine_chunks().to_numpy(zero_copy_only=False)
            voltage = table["voltage_v"].combine_chunks().to_numpy(zero_copy_only=False)
            if self.mode == "cv":
                output.append(self._label(table, Step.CV))
                break
            if self.mode == "cc":
                count = self.cc_count + np.arange(1, len(current) + 1, dtype=np.float64)
                sums = self.cc_sum + np.cumsum(current, dtype=np.longdouble)
                means = sums / count
                minima = np.minimum.accumulate(np.r_[self.cc_min, current])[1:]
                maxima = np.maximum.accumulate(np.r_[self.cc_max, current])[1:]
                stable = maxima - minima <= self.cc_tolerance * np.abs(means)
                failed = np.flatnonzero(~stable)
                accepted = int(failed[0]) if len(failed) else len(current)
                if accepted:
                    part = table.slice(0, accepted)
                    output.append(self._label(part, Step.CC))
                    accepted_i = current[:accepted]
                    self.cc_count += accepted
                    self.cc_sum += float(np.sum(accepted_i, dtype=np.longdouble))
                    self.cc_min = min(self.cc_min, float(np.min(accepted_i)))
                    self.cc_max = max(self.cc_max, float(np.max(accepted_i)))
                if not len(failed):
                    break
                self.cc_reference = self.cc_sum / self.cc_count
                self.mode = "search"
                self.transition_count += 1
                table = table.slice(accepted)
                continue

            if len(current) < 3:
                if final:
                    output.append(self._label(table, Step.OTHER))
                else:
                    self.pending = table
                break
            cc = _cc_ok(current, self.cc_tolerance)
            cv = _cv_ok(
                current,
                voltage,
                self.cc_tolerance,
                self.cv_tolerance,
                self.cc_reference,
            )
            qualifying = cc | cv
            if not np.any(qualifying):
                keep = 0 if final else 2
                emit = len(current) - keep
                if emit:
                    output.append(self._label(table.slice(0, emit), Step.OTHER))
                if keep:
                    self.pending = table.slice(emit)
                break
            start = int(np.flatnonzero(qualifying)[0])
            if start:
                output.append(self._label(table.slice(0, start), Step.OTHER))
                table = table.slice(start)
            if cc[start]:
                seed = table.slice(0, 3)
                seed_i = seed["current_a"].combine_chunks().to_numpy(zero_copy_only=False)
                output.append(self._label(seed, Step.CC, new_portion=True))
                self.mode = "cc"
                self.cc_count = 3
                self.cc_sum = float(np.sum(seed_i, dtype=np.longdouble))
                self.cc_min = float(np.min(seed_i))
                self.cc_max = float(np.max(seed_i))
                self.transition_count += 1
                table = table.slice(3)
            else:
                self.mode = "cv"
                self.transition_count += 1
        return output

    def feed(self, table: pa.Table, *, final: bool = False) -> list[pa.Table]:
        """Classify a batch, splitting it at rest and half-cycle boundaries."""
        output: list[pa.Table] = []
        if table.num_rows == 0:
            return output
        half = table["half_cycle"].combine_chunks().to_numpy(zero_copy_only=False)
        rest = table["rest"].combine_chunks().to_numpy(zero_copy_only=False)
        boundaries = np.r_[0, np.flatnonzero((half[1:] != half[:-1]) | (rest[1:] != rest[:-1])) + 1]
        ends = np.r_[boundaries[1:], len(half)]
        lengths = ends - boundaries
        if len(boundaries) > 128 and np.all(lengths[1:-1] <= 3):
            first_end = int(ends[0])
            output.extend(
                self._feed_region(
                    table.slice(0, first_end),
                    (int(half[0]), bool(rest[0])),
                    final=True,
                )
            )
            middle_start = int(boundaries[1])
            middle_end = int(ends[-2])
            middle = table.slice(middle_start, middle_end - middle_start)
            labels = np.full(middle.num_rows, Step.OTHER, dtype=np.int8)
            local_starts = boundaries[1:-1] - middle_start
            local_ends = ends[1:-1] - middle_start
            middle_rest = rest[boundaries[1:-1]]
            for start, end in zip(
                local_starts[middle_rest].tolist(),
                local_ends[middle_rest].tolist(),
                strict=True,
            ):
                labels[start:end] = Step.REST
            triples = local_starts[(~middle_rest) & ((local_ends - local_starts) == 3)]
            if len(triples):
                current = middle["current_a"].to_numpy(zero_copy_only=False)
                voltage = middle["voltage_v"].to_numpy(zero_copy_only=False)
                indices = triples[:, None] + np.arange(3)
                iw = current[indices]
                vw = voltage[indices]
                cc = np.ptp(iw, axis=1) <= self.cc_tolerance * np.abs(np.mean(iw, axis=1))
                cv = (np.ptp(vw, axis=1) <= self.cv_tolerance) & (
                    np.ptp(iw, axis=1) > self.cc_tolerance * np.abs(np.mean(iw, axis=1))
                )
                labels[indices[cc].ravel()] = Step.CC
                labels[indices[(~cc) & cv].ravel()] = Step.CV
                self.transition_count += int(np.count_nonzero(cc | cv))
            portions = np.full(middle.num_rows, -1, dtype=np.int64)
            for start in triples[cc].tolist():
                self.portion_id += 1
                portions[start : start + 3] = self.portion_id
            output.extend(self._reset_region())
            self.region_key = None
            output.append(
                middle.append_column("step", pa.array(labels)).append_column(
                    "s4_portion", pa.array(portions)
                )
            )
            last_start = int(boundaries[-1])
            output.extend(
                self._feed_region(
                    table.slice(last_start),
                    (int(half[last_start]), bool(rest[last_start])),
                    final=final,
                )
            )
            return output
        for start, end in zip(boundaries.tolist(), ends.tolist(), strict=True):
            key = (int(half[start]), bool(rest[start]))
            output.extend(self._feed_region(table.slice(start, end - start), key, final=final))
        return output

    def _feed_region(self, part: pa.Table, key: tuple[int, bool], *, final: bool) -> list[pa.Table]:
        """Classify one part of a batch that lies in a single region."""
        output: list[pa.Table] = []
        if self.region_key != key:
            output.extend(self._reset_region())
            self.region_key = key
        if key[1]:
            output.append(self._label(part, Step.REST))
        else:
            output.extend(self._active(part, final=final))
        return output

    def finish(self) -> list[pa.Table]:
        """Classify pending rows at the end of the data."""
        output: list[pa.Table] = []
        if self.pending is not None:
            pending, self.pending = self.pending, None
            output.extend(self._active(pending, final=True))
        return output


class _S5Stream:
    """Streaming integration of cumulative capacity and energy."""

    def __init__(self) -> None:
        """Create the integration state."""
        self.previous: tuple[float, float, float, int] | None = None
        self.half = -1
        self.carry = np.zeros(4, dtype=np.longdouble)
        self.invalid_current = False
        self.closed: pa.Table | None = None

    def feed(self, table: pa.Table) -> pa.Table:
        """Integrate a classified batch and return it with cumulative columns."""
        time = table["time_s"].combine_chunks().to_numpy(zero_copy_only=False)
        current = table["current_a"].combine_chunks().to_numpy(zero_copy_only=False)
        voltage = table["voltage_v"].combine_chunks().to_numpy(zero_copy_only=False)
        half = table["half_cycle"].combine_chunks().to_numpy(zero_copy_only=False)
        had_previous = self.previous is not None
        if not had_previous:
            ext_t, ext_i, ext_v, ext_h = time, current, voltage, half
        else:
            assert self.previous is not None
            pt, pi, pv, ph = self.previous
            ext_t = np.r_[pt, time]
            ext_i = np.r_[pi, current]
            ext_v = np.r_[pv, voltage]
            ext_h = np.r_[ph, half]
        cq_i, dq_i, ce_i, de_i = _interval_contributions(ext_t, ext_i, ext_v)
        increments = np.column_stack((cq_i, dq_i, ce_i, de_i)).astype(np.longdouble)
        if not had_previous:
            increments = np.vstack((np.zeros((1, 4), dtype=np.longdouble), increments))
        boundary = ext_h[1:] != ext_h[:-1]
        crossing = ext_i[:-1] * ext_i[1:] < 0
        bc = boundary & crossing
        left_all = np.zeros((len(boundary), 4), dtype=np.longdouble)
        if np.any(bc):
            indices = np.flatnonzero(bc)
            alpha = np.abs(ext_i[indices]) / (np.abs(ext_i[indices]) + np.abs(ext_i[indices + 1]))
            dt = ext_t[indices + 1] - ext_t[indices]
            left_dt = alpha * dt
            right_dt = (1.0 - alpha) * dt
            left_positive = ext_i[indices] > 0
            right_positive = ext_i[indices + 1] > 0
            left_all[indices, 0] = np.where(
                left_positive, left_dt * np.abs(ext_i[indices]) / 7200.0, 0.0
            )
            left_all[indices, 1] = np.where(
                ~left_positive, left_dt * np.abs(ext_i[indices]) / 7200.0, 0.0
            )
            left_all[indices, 2] = np.where(
                left_positive,
                left_dt * np.abs(ext_v[indices] * ext_i[indices]) / 7200.0,
                0.0,
            )
            left_all[indices, 3] = np.where(
                ~left_positive,
                left_dt * np.abs(ext_v[indices] * ext_i[indices]) / 7200.0,
                0.0,
            )
            right = np.zeros((len(indices), 4), dtype=np.longdouble)
            right[:, 0] = np.where(
                right_positive, right_dt * np.abs(ext_i[indices + 1]) / 7200.0, 0.0
            )
            right[:, 1] = np.where(
                ~right_positive, right_dt * np.abs(ext_i[indices + 1]) / 7200.0, 0.0
            )
            right[:, 2] = np.where(
                right_positive,
                right_dt * np.abs(ext_v[indices + 1] * ext_i[indices + 1]) / 7200.0,
                0.0,
            )
            right[:, 3] = np.where(
                ~right_positive,
                right_dt * np.abs(ext_v[indices + 1] * ext_i[indices + 1]) / 7200.0,
                0.0,
            )
            increments[indices if had_previous else indices + 1] = right

        starts = np.r_[0, np.flatnonzero(half[1:] != half[:-1]) + 1]
        ends = np.r_[starts[1:], len(half)]
        output = np.column_stack(
            tuple(_group_point_cumulative(increments[:, column], half) for column in range(4))
        )
        if self.half == int(half[0]):
            first_end = int(ends[0])
            output[:first_end] += self.carry.astype(np.float64)

        invalid = np.logical_or.reduceat(voltage < 0, starts)
        if self.half == int(half[0]):
            invalid[0] |= self.invalid_current
        closed_half: list[np.ndarray] = []
        closed_invalid: list[np.ndarray] = []
        closed_outgoing: list[np.ndarray] = []
        offset = 1 if had_previous else 0
        if had_previous and int(ext_h[0]) != int(half[0]):
            closed_half.append(np.array([int(ext_h[0])], dtype=np.int32))
            closed_invalid.append(np.array([self.invalid_current]))
            closed_outgoing.append(left_all[[0]].astype(np.float64))
        if len(starts) > 1:
            closed_half.append(half[starts[:-1]].astype(np.int32))
            closed_invalid.append(invalid[:-1])
            boundary_indices = offset + ends[:-1] - 1
            closed_outgoing.append(left_all[boundary_indices].astype(np.float64))
        if closed_half:
            ids = np.concatenate(closed_half)
            flags = np.concatenate(closed_invalid)
            outgoing = np.vstack(closed_outgoing)
            self.closed = pa.table(
                {
                    "half_cycle": ids,
                    "invalid_energy": flags,
                    "outgoing_charge_capacity_ah": outgoing[:, 0],
                    "outgoing_discharge_capacity_ah": outgoing[:, 1],
                    "outgoing_charge_energy_wh": outgoing[:, 2],
                    "outgoing_discharge_energy_wh": outgoing[:, 3],
                }
            )
        else:
            self.closed = None

        self.carry = output[-1].astype(np.longdouble)
        self.half = int(half[-1])
        self.invalid_current = bool(invalid[-1])
        self.previous = (float(time[-1]), float(current[-1]), float(voltage[-1]), int(half[-1]))
        names = (
            "charge_capacity_ah",
            "discharge_capacity_ah",
            "charge_energy_wh",
            "discharge_energy_wh",
        )
        for index, name in enumerate(names):
            table = table.append_column(name, pa.array(output[:, index]))
        source_start = 0 if "source_row" not in table.column_names else None
        if source_start is not None:
            raise RuntimeError("source_row must be assigned before integration")
        return table

    def finish(self) -> pa.Table:
        """Return the metadata of the final half-cycle."""
        return pa.table(
            {
                "half_cycle": pa.array([self.half], type=pa.int32()),
                "invalid_energy": [self.invalid_current],
                "outgoing_charge_capacity_ah": [0.0],
                "outgoing_discharge_capacity_ah": [0.0],
                "outgoing_charge_energy_wh": [0.0],
                "outgoing_discharge_energy_wh": [0.0],
            }
        )


def _write_overview(
    points_path: Path,
    output_path: Path,
    width: int,
    batch_size: int,
    *,
    cancel: CancelToken | None = None,
    on_batch: Any | None = None,
) -> None:
    """Write one overview level of real points for bucket `width`."""
    if width > batch_size:
        _write_large_bucket_overview(
            points_path, output_path, width, batch_size, cancel=cancel, on_batch=on_batch
        )
        return
    writer: pq.ParquetWriter | None = None
    carry: pa.Table | None = None
    parquet = pq.ParquetFile(points_path)
    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            check_cancelled(cancel, OperationId.INGEST_AND_ELABORATE.value)
            table = pa.Table.from_batches((batch,))
            if carry is not None:
                table = pa.concat_tables((carry, table))
                carry = None
            complete = (table.num_rows // width) * width
            if complete:
                part = table.slice(0, complete)
                numeric = tuple(
                    part[name].combine_chunks().to_numpy(zero_copy_only=False)
                    for name in (
                        "time_s",
                        "current_a",
                        "voltage_v",
                        "charge_capacity_ah",
                        "discharge_capacity_ah",
                    )
                )
                selected = overview_indices(numeric, width)
                reduced = part.take(pa.array(selected, type=pa.uint64()))
                if writer is None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(output_path, reduced.schema, compression="zstd")
                writer.write_table(reduced)
            if complete < table.num_rows:
                carry = table.slice(complete)
            if on_batch is not None:
                on_batch(batch.num_rows)
        if carry is not None:
            numeric = tuple(
                carry[name].combine_chunks().to_numpy(zero_copy_only=False)
                for name in (
                    "time_s",
                    "current_a",
                    "voltage_v",
                    "charge_capacity_ah",
                    "discharge_capacity_ah",
                )
            )
            reduced = carry.take(pa.array(overview_indices(numeric, width), type=pa.uint64()))
            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(output_path, reduced.schema, compression="zstd")
            writer.write_table(reduced)
    finally:
        parquet.close()
        if writer is not None:
            writer.close()


def _overview_candidates(table: pa.Table) -> pa.Table:
    """Keep the first, last and extreme rows of a table."""
    numeric = tuple(
        table[name].combine_chunks().to_numpy(zero_copy_only=False)
        for name in (
            "time_s",
            "current_a",
            "voltage_v",
            "charge_capacity_ah",
            "discharge_capacity_ah",
        )
    )
    selected = overview_indices(numeric, table.num_rows)
    return table.take(pa.array(selected, type=pa.uint64()))


def _write_large_bucket_overview(
    points_path: Path,
    output_path: Path,
    width: int,
    batch_size: int,
    *,
    cancel: CancelToken | None = None,
    on_batch: Any | None = None,
) -> None:
    """Write an overview level whose buckets exceed one batch."""
    parquet = pq.ParquetFile(points_path)
    writer: pq.ParquetWriter | None = None
    candidates: pa.Table | None = None
    seen = 0
    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            check_cancelled(cancel, OperationId.INGEST_AND_ELABORATE.value)
            table = pa.Table.from_batches((batch,))
            offset = 0
            while offset < table.num_rows:
                take = min(width - seen, table.num_rows - offset)
                part = _overview_candidates(table.slice(offset, take))
                candidates = (
                    part
                    if candidates is None
                    else _overview_candidates(pa.concat_tables((candidates, part)))
                )
                seen += take
                offset += take
                if seen == width:
                    candidates = candidates.sort_by([("source_row", "ascending")])
                    if writer is None:
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        writer = pq.ParquetWriter(
                            output_path, candidates.schema, compression="zstd"
                        )
                    writer.write_table(candidates)
                    candidates = None
                    seen = 0
            if on_batch is not None:
                on_batch(batch.num_rows)
        if candidates is not None:
            candidates = candidates.sort_by([("source_row", "ascending")])
            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(output_path, candidates.schema, compression="zstd")
            writer.write_table(candidates)
    finally:
        parquet.close()
        if writer is not None:
            writer.close()


def _release_batch_memory() -> None:
    """Return memory from completed passes before the next pass."""
    gc.collect()
    pa.default_memory_pool().release_unused()


class _HalfMetaCursor:
    """Ordered reader of staged half-cycle metadata, merged batch by batch."""

    def __init__(self, path: Path, batch_size: int) -> None:
        """Open the staged metadata file for batched reading."""
        self.file = pq.ParquetFile(path)
        self.batches = iter(self.file.iter_batches(batch_size=batch_size))
        self.ids = np.empty(0, dtype=np.int32)
        self.invalid = np.empty(0, dtype=bool)
        self.outgoing = np.empty((0, 4), dtype=np.float64)
        self.position = 0
        self.last_id: int | None = None
        self.last_invalid = False
        self.last_outgoing = np.zeros(4, dtype=np.float64)

    def _load(self) -> None:
        """Load the next metadata batch."""
        batch = next(self.batches)
        table = pa.Table.from_batches((batch,))
        self.ids = table["half_cycle"].to_numpy(zero_copy_only=False)
        self.invalid = table["invalid_energy"].to_numpy(zero_copy_only=False)
        self.outgoing = np.column_stack(
            tuple(
                table[name].to_numpy(zero_copy_only=False)
                for name in (
                    "outgoing_charge_capacity_ah",
                    "outgoing_discharge_capacity_ah",
                    "outgoing_charge_energy_wh",
                    "outgoing_discharge_energy_wh",
                )
            )
        )
        self.position = 0

    def lookup(self, requested: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return energy validity and outgoing contributions for half-cycles."""
        flags = np.empty(len(requested), dtype=bool)
        outgoing = np.empty((len(requested), 4), dtype=np.float64)
        result = 0
        if len(requested) and self.last_id == int(requested[0]):
            flags[0] = self.last_invalid
            outgoing[0] = self.last_outgoing
            result = 1
        while result < len(requested):
            if self.position == len(self.ids):
                self._load()
            available = len(self.ids) - self.position
            take = min(available, len(requested) - result)
            actual = self.ids[self.position : self.position + take]
            expected = requested[result : result + take]
            if not np.array_equal(actual, expected):
                raise RuntimeError(
                    "Cumulative metadata stream is not aligned with point half-cycles"
                )
            flags[result : result + take] = self.invalid[self.position : self.position + take]
            outgoing[result : result + take] = self.outgoing[self.position : self.position + take]
            self.position += take
            result += take
        if len(requested):
            self.last_id = int(requested[-1])
            self.last_invalid = bool(flags[-1])
            self.last_outgoing = outgoing[-1].copy()
        return flags, outgoing

    def close(self) -> None:
        """Close the metadata file."""
        self.file.close()


def elaborate(
    request: ElaborationRequest,
    *,
    progress: ProgressSink | None = None,
    cancel: CancelToken | None = None,
    fault_hook: Any | None = None,
) -> ElaborationResult:
    """Elaborate a canonical import and publish the result.

    Args:
        request: `ElaborationRequest`.
        progress: Optional progress sink (unit ``work_rows``).
        cancel: Optional cancellation token.
        fault_hook: Testing hook; leave it None.

    Returns:
        `ElaborationResult`; its `fingerprint` and ``ref.path`` are the inputs of every later
        analysis.

    Raises:
        ValueError: invalid parameters or an empty canonical source.
        StorageError, OperationCancelled: storage failures or cancellation.

    The data are processed in several bounded streaming passes, so memory use does not grow with
    the file size; temporary scratch files are written inside the job's staging directory.
    """
    request.validate()
    source = request.canonical_parquet.resolve(strict=True)
    check_cancelled(cancel, OperationId.INGEST_AND_ELABORATE.value)
    source_file = pq.ParquetFile(source)
    source_rows = source_file.metadata.num_rows
    source_file.close()
    total_work = source_rows * (3 + len(OVERVIEW_WIDTHS))
    report_progress(
        progress,
        ProgressEvent(OperationId.INGEST_AND_ELABORATE, "starting", 0, total_work, "work_rows"),
    )
    store = ResultStore(request.caller_root)
    staging = store.begin_job(request.sample_id, request.job_id, request.generation_id)
    try:
        scratch = staging.path / ".m2-scratch"
        scratch.mkdir()
        s3_path = scratch / "s3.parquet"
        provisional_path = scratch / "provisional.parquet"
        half_meta_path = scratch / "half-meta.parquet"
        rows = 0

        s3 = _S3Stream(request.parameters.rest_current_threshold_a, request.parameters.cycle_order)
        s3_writer: pq.ParquetWriter | None = None
        try:
            parquet = pq.ParquetFile(source)
            for batch in parquet.iter_batches(
                batch_size=request.batch_size,
                columns=["time_s", "current_a", "voltage_v"],
            ):
                check_cancelled(cancel, OperationId.INGEST_AND_ELABORATE.value)
                table = pa.Table.from_batches((batch,)).append_column(
                    "source_row",
                    pa.array(np.arange(rows, rows + batch.num_rows, dtype=np.uint64)),
                )
                rows += batch.num_rows
                report_progress(
                    progress,
                    ProgressEvent(
                        OperationId.INGEST_AND_ELABORATE,
                        "s3-segmentation",
                        rows,
                        total_work,
                        "work_rows",
                    ),
                )
                emitted = s3.feed(table)
                if emitted is not None:
                    if s3_writer is None:
                        s3_writer = pq.ParquetWriter(s3_path, emitted.schema, compression="zstd")
                    s3_writer.write_table(emitted, row_group_size=request.batch_size)
            emitted = s3.finish()
            if emitted is not None:
                if s3_writer is None:
                    s3_writer = pq.ParquetWriter(s3_path, emitted.schema, compression="zstd")
                s3_writer.write_table(emitted, row_group_size=request.batch_size)
            parquet.close()
        finally:
            if s3_writer is not None:
                s3_writer.close()
        del parquet, table, batch, emitted
        _release_batch_memory()
        if rows == 0 or not s3_path.exists():
            raise ValueError("canonical source must contain at least one row")

        s4 = _S4Stream(
            request.parameters.cc_relative_tolerance,
            request.parameters.cv_absolute_tolerance_v,
        )
        s5 = _S5Stream()
        provisional_writer: _BoundedParquetSink | None = None
        meta_writer: _BoundedParquetSink | None = None
        s4_scanned = 0
        try:
            s3_file = pq.ParquetFile(s3_path)
            for batch in s3_file.iter_batches(batch_size=request.batch_size):
                check_cancelled(cancel, OperationId.INGEST_AND_ELABORATE.value)
                s4_scanned += batch.num_rows
                for classified in s4.feed(pa.Table.from_batches((batch,))):
                    integrated = s5.feed(classified)
                    if provisional_writer is None:
                        provisional_writer = _BufferedStagingSink(
                            provisional_path, integrated.schema, request.batch_size
                        )
                    provisional_writer.write_table(integrated)
                    if s5.closed is not None:
                        if meta_writer is None:
                            meta_writer = _BufferedStagingSink(
                                half_meta_path,
                                s5.closed.schema,
                                request.batch_size,
                                baseline_row_group_size=False,
                            )
                        meta_writer.write_table(s5.closed)
                report_progress(
                    progress,
                    ProgressEvent(
                        OperationId.INGEST_AND_ELABORATE,
                        "s4-s5-elaboration",
                        rows + s4_scanned,
                        total_work,
                        "work_rows",
                    ),
                )
            for classified in s4.finish():
                integrated = s5.feed(classified)
                if provisional_writer is None:
                    provisional_writer = _BufferedStagingSink(
                        provisional_path, integrated.schema, request.batch_size
                    )
                provisional_writer.write_table(integrated)
                if s5.closed is not None:
                    if meta_writer is None:
                        meta_writer = _BufferedStagingSink(
                            half_meta_path,
                            s5.closed.schema,
                            request.batch_size,
                            baseline_row_group_size=False,
                        )
                    meta_writer.write_table(s5.closed)
            final_meta = s5.finish()
            if meta_writer is None:
                meta_writer = _BufferedStagingSink(
                    half_meta_path,
                    final_meta.schema,
                    request.batch_size,
                    baseline_row_group_size=False,
                )
            meta_writer.write_table(final_meta)
            s3_file.close()
        finally:
            if provisional_writer is not None:
                provisional_writer.close()
            if meta_writer is not None:
                meta_writer.close()
        del s3_file, batch, classified, integrated
        _release_batch_memory()

        points_path = staging.path / "points" / "elaborated.parquet"
        points_path.parent.mkdir(parents=True)
        point_writer: pq.ParquetWriter | None = None
        index_path = staging.path / "indexes" / "half_cycles.parquet"
        segments_path = staging.path / "indexes" / "segments.parquet"
        index_path.parent.mkdir(parents=True)
        half_writer = _BoundedParquetSink(
            index_path,
            HALF_INDEX_SCHEMA,
            request.batch_size,
            baseline_row_group_size=False,
        )
        segment_writer = _BoundedParquetSink(
            segments_path,
            SEGMENT_INDEX_SCHEMA,
            request.batch_size,
            baseline_row_group_size=False,
        )
        half_cursor = _HalfMetaCursor(half_meta_path, request.batch_size)
        open_half: dict[str, Any] | None = None
        open_segment: dict[str, Any] | None = None
        half_count = 0
        cycle_count = 0
        segment_count = 0
        final_scanned = 0
        try:
            provisional_file = pq.ParquetFile(provisional_path)
            for batch in provisional_file.iter_batches(batch_size=request.batch_size):
                check_cancelled(cancel, OperationId.INGEST_AND_ELABORATE.value)
                final_scanned += batch.num_rows
                table = pa.Table.from_batches((batch,))
                half = table["half_cycle"].to_numpy(zero_copy_only=False)
                half_starts = np.r_[0, np.flatnonzero(half[1:] != half[:-1]) + 1]
                half_ends = np.r_[half_starts[1:], len(half)]
                half_ids = half[half_starts].astype(np.int32)
                invalid_groups, outgoing = half_cursor.lookup(half_ids)
                invalid = pa.array(np.repeat(invalid_groups, half_ends - half_starts))
                for name in ("charge_energy_wh", "discharge_energy_wh"):
                    index = table.schema.get_field_index(name)
                    table = table.set_column(index, name, pc.if_else(invalid, None, table[name]))
                order = [
                    "time_s",
                    "current_a",
                    "voltage_v",
                    "charge_capacity_ah",
                    "discharge_capacity_ah",
                    "charge_energy_wh",
                    "discharge_energy_wh",
                    "cycle",
                    "half_cycle",
                    "rest",
                    "step",
                    "source_row",
                ]
                s4_portion = table["s4_portion"].to_numpy(zero_copy_only=False)
                table = table.select(order)
                if point_writer is None:
                    point_writer = pq.ParquetWriter(
                        points_path,
                        table.schema,
                        compression="zstd",
                        write_statistics=["time_s", "cycle", "half_cycle"],
                    )
                point_writer.write_table(table, row_group_size=PARQUET_ROW_GROUP_SIZE)

                source_rows = table["source_row"].to_numpy(zero_copy_only=False)
                current = table["current_a"].to_numpy(zero_copy_only=False)
                cycle = table["cycle"].to_numpy(zero_copy_only=False)
                step = table["step"].to_numpy(zero_copy_only=False)
                cq = table["charge_capacity_ah"].to_numpy(zero_copy_only=False)
                dq = table["discharge_capacity_ah"].to_numpy(zero_copy_only=False)
                ce = table["charge_energy_wh"].to_numpy(zero_copy_only=False)
                de = table["discharge_energy_wh"].to_numpy(zero_copy_only=False)
                last_indices = half_ends - 1
                first_rows = source_rows[half_starts].astype(np.uint64)
                if open_half is not None and int(half_ids[0]) != int(open_half["half_cycle"]):
                    half_writer.write_table(
                        pa.Table.from_pylist([open_half], schema=HALF_INDEX_SCHEMA)
                    )
                    half_count += 1
                    cycle_count = max(cycle_count, int(open_half["cycle"]))
                    open_half = None
                if open_half is not None:
                    first_rows[0] = int(open_half["first_row"])
                summary = pa.table(
                    {
                        "half_cycle": half_ids,
                        "cycle": cycle[half_starts].astype(np.int32),
                        "first_row": first_rows,
                        "last_row": source_rows[last_indices].astype(np.uint64),
                        "first_direction": np.full(
                            len(half_ids), s3.first_direction, dtype=np.int8
                        ),
                        "charge_capacity_ah": cq[last_indices] + outgoing[:, 0],
                        "discharge_capacity_ah": dq[last_indices] + outgoing[:, 1],
                        "charge_energy_wh": pa.array(
                            ce[last_indices] + outgoing[:, 2], mask=invalid_groups
                        ),
                        "discharge_energy_wh": pa.array(
                            de[last_indices] + outgoing[:, 3], mask=invalid_groups
                        ),
                    }
                )
                if summary.num_rows > 1:
                    closed_summary = summary.slice(0, summary.num_rows - 1)
                    half_writer.write_table(closed_summary)
                    half_count += closed_summary.num_rows
                    cycle_count = max(
                        cycle_count,
                        int(np.max(closed_summary["cycle"].to_numpy(zero_copy_only=False))),
                    )
                open_half = summary.slice(summary.num_rows - 1, 1).to_pylist()[0]

                boundaries = np.r_[
                    0,
                    np.flatnonzero(
                        (half[1:] != half[:-1])
                        | (step[1:] != step[:-1])
                        | ((step[1:] == Step.CC) & (s4_portion[1:] != s4_portion[:-1]))
                    )
                    + 1,
                ]
                ends = np.r_[boundaries[1:], len(half)]
                segment_first = source_rows[boundaries].astype(np.uint64)
                point_time = np.asarray(
                    table["time_s"].to_numpy(zero_copy_only=False), dtype=np.float64
                )
                segment_first_time = point_time[boundaries].copy()
                segment_polarity = np.sign(current[boundaries]).astype(np.int8)
                if open_segment is not None and (
                    int(half[boundaries[0]]) != int(open_segment["half_cycle"])
                    or int(step[boundaries[0]]) != int(open_segment["step"])
                    or int(s4_portion[boundaries[0]]) != int(open_segment["s4_portion"])
                ):
                    open_segment["segment_id"] = segment_count
                    segment_writer.write_table(
                        pa.Table.from_pylist([open_segment], schema=SEGMENT_INDEX_SCHEMA)
                    )
                    segment_count += 1
                    open_segment = None
                if open_segment is not None:
                    segment_first[0] = int(open_segment["first_row"])
                    segment_first_time[0] = float(open_segment["first_time_s"])
                    segment_polarity[0] = int(open_segment["polarity"])
                segments = pa.table(
                    {
                        "segment_id": np.arange(
                            segment_count,
                            segment_count + len(boundaries),
                            dtype=np.uint64,
                        ),
                        "first_row": segment_first,
                        "last_row": source_rows[ends - 1].astype(np.uint64),
                        "first_time_s": segment_first_time,
                        "last_time_s": point_time[ends - 1],
                        "cycle": cycle[boundaries].astype(np.int32),
                        "half_cycle": half[boundaries].astype(np.int32),
                        "step": step[boundaries].astype(np.int8),
                        "s4_portion": s4_portion[boundaries].astype(np.int64),
                        "polarity": segment_polarity,
                    }
                )
                if segments.num_rows > 1:
                    closed_segments = segments.slice(0, segments.num_rows - 1)
                    segment_writer.write_table(closed_segments)
                    segment_count += closed_segments.num_rows
                open_segment = segments.slice(segments.num_rows - 1, 1).to_pylist()[0]
                report_progress(
                    progress,
                    ProgressEvent(
                        OperationId.INGEST_AND_ELABORATE,
                        "finalize-points-indexes",
                        2 * rows + final_scanned,
                        total_work,
                        "work_rows",
                    ),
                )
            provisional_file.close()
        except Exception:
            half_writer.close()
            segment_writer.close()
            raise
        finally:
            if point_writer is not None:
                point_writer.close()
            half_cursor.close()
        if open_half is None or open_segment is None:
            raise RuntimeError("streaming indexes have no final record")
        final_summary = pa.Table.from_pylist([open_half], schema=HALF_INDEX_SCHEMA)
        half_writer.write_table(final_summary)
        half_writer.close()
        half_count += 1
        cycle_count = max(cycle_count, int(open_half["cycle"]))
        open_segment["segment_id"] = segment_count
        final_segment = pa.Table.from_pylist([open_segment], schema=SEGMENT_INDEX_SCHEMA)
        segment_writer.write_table(final_segment)
        segment_writer.close()
        segment_count += 1
        # Drop the last batch's Arrow and NumPy views before the overview
        # scans.  PyArrow otherwise keeps those buffers alive long enough to
        # inflate the process high-water mark on large inputs.
        del (
            provisional_file,
            batch,
            table,
            half,
            half_starts,
            half_ends,
            half_ids,
            invalid_groups,
            invalid,
            outgoing,
            source_rows,
            cycle,
            step,
            current,
            s4_portion,
            point_time,
            segment_first_time,
            segment_polarity,
            cq,
            dq,
            ce,
            de,
            summary,
            segments,
            final_summary,
            final_segment,
            half_cursor,
            half_writer,
            segment_writer,
            open_half,
            open_segment,
        )
        _release_batch_memory()
        check_cancelled(cancel, OperationId.INGEST_AND_ELABORATE.value)
        if fault_hook is not None:
            fault_hook("after_points")
        for overview_index, width in enumerate(OVERVIEW_WIDTHS):
            check_cancelled(cancel, OperationId.INGEST_AND_ELABORATE.value)
            path = staging.path / "overview" / f"level-{width}.parquet"
            overview_scanned = [0]

            def overview_progress(
                batch_rows: int,
                *,
                stage: int = overview_index,
                tracker: list[int] = overview_scanned,
                level: int = width,
            ) -> None:
                tracker[0] += batch_rows
                report_progress(
                    progress,
                    ProgressEvent(
                        OperationId.INGEST_AND_ELABORATE,
                        f"overview-{level}",
                        (3 + stage) * rows + tracker[0],
                        total_work,
                        "work_rows",
                    ),
                )

            _write_overview(
                points_path,
                path,
                width,
                request.batch_size,
                cancel=cancel,
                on_batch=overview_progress,
            )
            _release_batch_memory()
        reports = staging.path / "reports"
        reports.mkdir()
        scratch_bytes = {
            "s3_parquet": s3_path.stat().st_size,
            "provisional_parquet": provisional_path.stat().st_size,
            "half_metadata_parquet": half_meta_path.stat().st_size,
        }
        (reports / "elaboration.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "cycle_order": request.parameters.cycle_order.value,
                    "canonical_scans": 1,
                    "scratch_scans": {"s3_parquet": 1, "provisional_parquet": 1},
                    "half_metadata_merge_scans": 1,
                    "final_point_overview_scans": len(OVERVIEW_WIDTHS),
                    "scratch_bytes": scratch_bytes,
                    "scratch_total_bytes": sum(scratch_bytes.values()),
                    "half_cycle_records": half_count,
                    "step_segment_records": segment_count,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        shutil.rmtree(scratch)
        fingerprint = _result_fingerprint(request)
        report_progress(
            progress,
            ProgressEvent(
                OperationId.INGEST_AND_ELABORATE,
                "elaborated",
                total_work,
                total_work,
                "work_rows",
            ),
        )
        dependency = DependencyRef("canonical", request.canonical_content_hash)
        manifest = staging.build_manifest(
            operation=OperationId.INGEST_AND_ELABORATE,
            parameter_hash=hash_parameters(request.parameters.as_dict()),
            dependency_hash=hash_dependencies((dependency,)),
            dependencies=(dependency,),
            content_hash=fingerprint,
            diagnostics=(),
            row_counts={
                "points": rows,
                "half_cycle_records": half_count,
                "half_cycles": s5.half,
                "cycles": cycle_count,
                "step_segment_records": segment_count,
                "s3_sign_runs": s3.run_count,
                "s4_state_transitions": s4.transition_count,
            },
            logical_schemas={
                "points/elaborated.parquet": "gregorithm.elaborated_points.v1",
                "indexes/half_cycles.parquet": "gregorithm.half_cycle_index.v1",
                "indexes/segments.parquet": "gregorithm.step_segment_index.v1",
                "reports/elaboration.json": "gregorithm.elaboration_report.v1",
                **{
                    f"overview/level-{width}.parquet": "gregorithm.real_point_overview.v1"
                    for width in OVERVIEW_WIDTHS
                },
            },
        )
        ref = store.publish(staging, manifest, cancel=cancel, fault_hook=fault_hook)
        return ElaborationResult(
            ref,
            rows,
            s5.half,
            cycle_count,
            fingerprint,
        )
    except Exception:
        staging.close()
        raise


def read_window(request: WindowRequest) -> pa.Table:
    """Read elaborated points, optionally filtered by time and cycle.

    Without `point_budget` all matching rows are returned. With a budget, at most that many real
    rows are returned: the first and last rows and the minimum/maximum of each requested numeric
    column are always kept, the rest are evenly spaced. Raises ValueError if the budget cannot
    hold those mandatory rows. Returns a `pyarrow.Table` in time order.
    """
    root = request.result_path.resolve(strict=True)
    points = root / "points" / "elaborated.parquet"
    expression: ds.Expression | None = None

    def add_filter(value: ds.Expression) -> None:
        """Combine a filter expression with the existing ones (logical AND)."""
        nonlocal expression
        expression = value if expression is None else expression & value

    if request.start_time_s is not None:
        add_filter(ds.field("time_s") >= request.start_time_s)
    if request.end_time_s is not None:
        add_filter(ds.field("time_s") <= request.end_time_s)
    if request.cycles:
        add_filter(ds.field("cycle").isin(list(request.cycles)))
    columns = list(request.columns) if request.columns else None
    dataset = ds.dataset(points, format="parquet")
    if request.point_budget is None:
        return dataset.to_table(columns=columns, filter=expression)
    if request.point_budget < 1:
        raise ValueError("point_budget must be positive")

    output_names = columns or list(dataset.schema.names)
    extrema_names: list[str] = [
        name
        for name in (
            "time_s",
            "current_a",
            "voltage_v",
            "charge_capacity_ah",
            "discharge_capacity_ah",
            "charge_energy_wh",
            "discharge_energy_wh",
        )
        if name in output_names
    ]
    scan_names = list(dict.fromkeys(output_names + extrema_names))
    count = 0
    minima: dict[str, tuple[float, int]] = {}
    maxima: dict[str, tuple[float, int]] = {}
    scanner = dataset.scanner(columns=scan_names, filter=expression, batch_size=65_536)
    for batch in scanner.to_batches():
        for name in extrema_names:
            values = batch.column(batch.schema.get_field_index(name)).to_numpy(zero_copy_only=False)
            finite = np.flatnonzero(np.isfinite(values))
            if not len(finite):
                continue
            valid = values[finite]
            low_local = int(finite[int(np.argmin(valid))])
            high_local = int(finite[int(np.argmax(valid))])
            low = float(values[low_local])
            high = float(values[high_local])
            if name not in minima or low < minima[name][0]:
                minima[name] = (low, count + low_local)
            if name not in maxima or high > maxima[name][0]:
                maxima[name] = (high, count + high_local)
        count += batch.num_rows
    if count == 0:
        return dataset.to_table(columns=output_names, filter=expression)
    if count <= request.point_budget:
        return dataset.to_table(columns=output_names, filter=expression)

    mandatory = {0, count - 1}
    mandatory.update(rank for _, rank in minima.values())
    mandatory.update(rank for _, rank in maxima.values())
    if len(mandatory) > request.point_budget:
        raise ValueError(
            "point_budget is too small for first/last and all requested-column extrema: "
            f"required={len(mandatory)}, budget={request.point_budget}"
        )
    targets = set(mandatory)
    if len(targets) < request.point_budget:
        for rank in np.linspace(0, count - 1, request.point_budget, dtype=np.int64).tolist():
            targets.add(int(rank))
            if len(targets) == request.point_budget:
                break
    ordered = sorted(targets)
    pieces: list[pa.Table] = []
    offset = 0
    target_index = 0
    for batch in dataset.scanner(
        columns=output_names, filter=expression, batch_size=65_536
    ).to_batches():
        local: list[int] = []
        while target_index < len(ordered) and ordered[target_index] < offset + batch.num_rows:
            if ordered[target_index] >= offset:
                local.append(ordered[target_index] - offset)
            target_index += 1
        if local:
            pieces.append(pa.Table.from_batches((batch,)).take(pa.array(local, type=pa.int64())))
        offset += batch.num_rows
    return pa.concat_tables(pieces) if pieces else pa.table({name: [] for name in output_names})
