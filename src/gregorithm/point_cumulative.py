"""Streaming cumulative capacity and energy coordinates on original measurements."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ._result_tables import bounded_frame
from .diagnostics import BudgetExceeded, SourceChangedError
from .execution import CancellationToken, MemoryBudget
from .io import atomic_parquet, fingerprint
from .point_calculations import _effective_batches, directional_integrals

_DERIVED = (
    "cumulative_charge_mAh",
    "cumulative_discharge_mAh",
    "cumulative_net_mAh",
    "cumulative_capacity_quality",
    "cumulative_charge_Wh",
    "cumulative_discharge_Wh",
    "cumulative_net_Wh",
    "cumulative_energy_quality",
)


def _kahan(total, correction, value):
    """Add one interval contribution reproducibly without changing its sign."""
    adjusted = value - correction
    combined = total + adjusted
    return combined, (combined - total) - adjusted


@dataclass(frozen=True)
class CumulativeView:
    """Lazy original measurements plus explicitly requested cumulative coordinates."""

    source: object = field(repr=False)
    report: dict
    _budget: object = field(repr=False)
    _include_energy: bool = True

    def iter_batches(self, columns=None, cancellation=None):
        """Yield preserved source batches with signed cumulative Q/E columns appended."""
        token = cancellation or CancellationToken()
        requested = None if columns is None else tuple(dict.fromkeys(columns))
        if requested == ():
            raise ValueError("Projection must contain at least one column")
        original_names = pq.read_schema(self.source.path).names
        if requested is not None:
            unknown = set(requested) - set(original_names) - set(_DERIVED)
            if unknown:
                raise ValueError(f"Unknown cumulative projection columns: {sorted(unknown)}")
            original_projection = [name for name in requested if name in original_names]
        else:
            original_projection = list(original_names)
        required = [self.source.report["time_column"], "current_mA"]
        if self._include_energy:
            required.append("voltage_V")
        extra = tuple(name for name in original_projection if name not in required)
        reader_bytes = min(
            self.source.read_options.max_batch_bytes,
            (self._budget.limit - self._budget.used - 2 * 1024**2) // 20,
        )
        if reader_bytes < 64:
            raise BudgetExceeded("Insufficient cumulative-coordinate budget")
        reservation = reader_bytes * 20 + 2 * 1024**2
        self._budget.reserve(reservation)
        view = replace(
            self.source,
            read_options=replace(self.source.read_options, max_batch_bytes=reader_bytes),
        )
        initial = self.report["prepared_source_metadata"]
        reader = _effective_batches(
            view,
            required,
            self.report["missing_span_s"],
            token,
            extra_columns=extra,
            include_original=True,
        )
        totals = np.zeros(4, dtype=np.float64)
        corrections = np.zeros(4, dtype=np.float64)
        partial = np.zeros(2, dtype=bool)
        estimated_seen = np.zeros(2, dtype=bool)
        previous = None
        try:
            for times, _, current, voltage, estimate_i, estimate_v, original, _ in reader:
                token.check()
                n = len(times)
                values = np.empty((n, 4), dtype=np.float64)
                values[:] = np.nan
                q_quality, e_quality = [], []
                for position in range(n):
                    if previous is None and position == 0:
                        values[position] = totals
                        q_quality.append("complete")
                        e_quality.append("complete" if self._include_energy else "unavailable")
                        continue
                    if position == 0:
                        pair_t = np.array([previous[0], times[position]])
                        pair_i = np.array([previous[1], current[position]])
                        pair_v = (
                            np.array([previous[2], voltage[position]])
                            if self._include_energy
                            else None
                        )
                        left_ei, left_ev = previous[3], previous[4]
                    else:
                        pair_t = times[position - 1 : position + 1]
                        pair_i = current[position - 1 : position + 1]
                        pair_v = (
                            voltage[position - 1 : position + 1] if self._include_energy else None
                        )
                        left_ei = estimate_i[position - 1]
                        left_ev = False if estimate_v is None else estimate_v[position - 1]
                    increment = directional_integrals(pair_t, pair_i, pair_v)[0]
                    q_valid = np.all(np.isfinite(increment[:2]))
                    if q_valid:
                        for k in range(2):
                            totals[k], corrections[k] = _kahan(
                                totals[k], corrections[k], increment[k]
                            )
                        values[position, :2] = totals[:2]
                        estimated_seen[0] |= bool(left_ei or estimate_i[position])
                        q_quality.append(
                            "partial"
                            if partial[0]
                            else "estimated"
                            if estimated_seen[0]
                            else "complete"
                        )
                    else:
                        partial[0] = True
                        q_quality.append("unavailable_interval")
                    if self._include_energy:
                        e_valid = np.all(np.isfinite(increment[2:]))
                        if e_valid:
                            for k in range(2, 4):
                                totals[k], corrections[k] = _kahan(
                                    totals[k], corrections[k], increment[k]
                                )
                            values[position, 2:] = totals[2:]
                            estimated_seen[1] |= bool(
                                left_ei or estimate_i[position] or left_ev or estimate_v[position]
                            )
                            e_quality.append(
                                "partial"
                                if partial[1]
                                else "estimated"
                                if estimated_seen[1]
                                else "complete"
                            )
                        else:
                            partial[1] = True
                            e_quality.append("unavailable_interval")
                    else:
                        e_quality.append("unavailable")
                previous = (
                    times[-1],
                    current[-1],
                    None if voltage is None else voltage[-1],
                    estimate_i[-1],
                    False if estimate_v is None else estimate_v[-1],
                )
                appended = {
                    "cumulative_charge_mAh": pa.array(
                        values[:, 0], mask=~np.isfinite(values[:, 0])
                    ),
                    "cumulative_discharge_mAh": pa.array(
                        values[:, 1], mask=~np.isfinite(values[:, 1])
                    ),
                    "cumulative_net_mAh": pa.array(
                        values[:, 0] - values[:, 1],
                        mask=~np.isfinite(values[:, 0]) | ~np.isfinite(values[:, 1]),
                    ),
                    "cumulative_capacity_quality": pa.array(q_quality).dictionary_encode(),
                    "cumulative_charge_Wh": pa.array(values[:, 2], mask=~np.isfinite(values[:, 2])),
                    "cumulative_discharge_Wh": pa.array(
                        values[:, 3], mask=~np.isfinite(values[:, 3])
                    ),
                    "cumulative_net_Wh": pa.array(
                        values[:, 2] - values[:, 3],
                        mask=~np.isfinite(values[:, 2]) | ~np.isfinite(values[:, 3]),
                    ),
                    "cumulative_energy_quality": pa.array(e_quality).dictionary_encode(),
                }
                output = original.select(original_projection)
                names = (
                    _DERIVED
                    if requested is None
                    else tuple(name for name in requested if name in _DERIVED)
                )
                for name in names:
                    output = output.append_column(name, appended[name])
                if requested is not None:
                    output = output.select(requested)
                yield output
            token.check()
            if fingerprint(self.source.path) != initial:
                raise SourceChangedError("Prepared source changed during cumulative calculation")
        finally:
            reader.close()
            self._budget.release(reservation)

    def preview(self, rows=5):
        """Return a small explicitly truncated pandas preview."""
        return bounded_frame(self, rows, preview=True)

    def table(self, max_rows=100000):
        """Materialize only results inside the explicit row and memory guards."""
        return bounded_frame(self, max_rows)

    def write_parquet(self, path, *, columns=None, cancellation=None):
        """Stream originals and cumulative coordinates to a new Parquet file."""
        if Path(path).exists():
            raise FileExistsError("Choose a new cumulative export path")
        return atomic_parquet(
            self.iter_batches(columns=columns, cancellation=cancellation),
            path,
            manifest={**self.report, "projection": columns},
        )


def cumulative_coordinates(
    source,
    *,
    include_energy=True,
    missing_span_s=10.0,
    memory_budget=None,
):
    """Return a lazy cumulative-coordinate view without changing the prepared source."""
    if not isinstance(include_energy, (bool, np.bool_)):
        raise ValueError("include_energy must be boolean")
    if (
        isinstance(missing_span_s, (bool, np.bool_))
        or not math.isfinite(missing_span_s)
        or missing_span_s < 0
    ):
        raise ValueError("missing_span_s must be finite and nonnegative")
    budget = memory_budget or MemoryBudget(2 * 1024**3)
    collisions = set(pq.read_schema(source.path).names) & set(_DERIVED)
    if collisions:
        raise ValueError(f"Cumulative output columns already exist: {sorted(collisions)}")
    report = {
        "source_file": source.report["source_file"],
        "prepared_source_metadata": fingerprint(source.path),
        "operation": "cumulative_capacity_energy",
        "include_energy": bool(include_energy),
        "missing_span_s": float(missing_span_s),
        "coordinate_origin": "original_source_start",
        "generated_measurement_identifiers": False,
    }
    return CumulativeView(source, report, budget, bool(include_energy))
