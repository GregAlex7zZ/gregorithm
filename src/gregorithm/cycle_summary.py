"""Streaming cycle summaries and additive manual-basis normalization.

These operations consume compact half-cycle summaries, never original measurement
arrays. Efficiencies use paired net quantities and independent support quality.
"""

import math
from dataclasses import dataclass, field

import pyarrow as pa
import pyarrow.compute as pc

from .diagnostics import BudgetExceeded, MaterializationLimitError
from .io import atomic_parquet


def _efficiency(charge, discharge, quantity, quality):
    """Return percent, propagated quality and reason without clamping or fallback."""
    if charge is None or discharge is None:
        return None, "unavailable", "missing_counterpart"
    if any(row[quality] not in {"complete", "estimated"} for row in (charge, discharge)):
        return None, "unavailable", "incomplete_support"
    denominator, numerator = charge[quantity], discharge[quantity]
    if denominator is None or numerator is None:
        return None, "unavailable", "unavailable_quantity"
    if denominator <= 0:
        return None, "unavailable", "nonpositive_charge"
    status = (
        "estimated"
        if any(row[quality] == "estimated" for row in (charge, discharge))
        else "complete"
    )
    return 100 * numerator / denominator, status, None


def _cycle_row(rows, energy):
    """Combine at most two directional halves; protocol completion is not inferred."""
    charge = next((row for row in rows if row["direction"] == "charge"), None)
    discharge = next((row for row in rows if row["direction"] == "discharge"), None)
    result = {
        "cycle": rows[0]["cycle"],
        "cycle_pair_complete": charge is not None and discharge is not None,
    }
    for prefix, row in (("charge", charge), ("discharge", discharge)):
        result[f"{prefix}_capacity_mAh"] = None if row is None else row["capacity_mAh"]
        result[f"{prefix}_capacity_quality"] = (
            "unavailable" if row is None else row["capacity_quality"]
        )
        if energy:
            result[f"{prefix}_energy_Wh"] = None if row is None else row["energy_Wh"]
            result[f"{prefix}_energy_quality"] = (
                "unavailable" if row is None else row["energy_quality"]
            )
    for name, quantity, quality in [("coulombic", "capacity_mAh", "capacity_quality")] + (
        [("energy", "energy_Wh", "energy_quality")] if energy else []
    ):
        value, status, reason = _efficiency(charge, discharge, quantity, quality)
        result[f"{name}_efficiency_percent"] = value
        result[f"{name}_efficiency_quality"] = status
        result[f"{name}_efficiency_reason"] = reason
    return result


def _cycle_schema(energy):
    """Provide stable types even for empty selections or entirely unavailable values."""
    fields = [("cycle", pa.int64()), ("cycle_pair_complete", pa.bool_())]
    for prefix in ("charge", "discharge"):
        fields += [
            (f"{prefix}_capacity_mAh", pa.float64()),
            (f"{prefix}_capacity_quality", pa.string()),
        ]
        if energy:
            fields += [
                (f"{prefix}_energy_Wh", pa.float64()),
                (f"{prefix}_energy_quality", pa.string()),
            ]
    for name in ("coulombic", "energy") if energy else ("coulombic",):
        fields += [
            (f"{name}_efficiency_percent", pa.float64()),
            (f"{name}_efficiency_quality", pa.string()),
            (f"{name}_efficiency_reason", pa.string()),
        ]
    return pa.schema(fields)


@dataclass(frozen=True)
class SummaryView:
    """Lazy bounded transformation of existing results; shares their storage lifetime.

    Use capacity.summary().table() or capacity.normalize(mass_g=0.005).table().
    Both mass and area may be supplied independently. No original column is removed.
    """

    source: object = field(repr=False)
    operation: str
    settings: dict = field(default_factory=dict)

    @property
    def report(self):
        """Return readable inherited provenance and operation settings once per result."""
        original = self.source.report
        return {
            **original,
            "selected_cycles": getattr(self.source, "_cycles", original.get("selected_cycles")),
            "view_operations": original.get("view_operations", [])
            + [{"operation": self.operation, "settings": self.settings}],
        }

    @property
    def _workspace(self):
        """Share the storage reader guard even after the upstream iterator is exhausted."""
        return self.source._workspace

    @property
    def _budget(self):
        """Share the original calculation budget across nested views and iterators."""
        return self.source._budget

    @property
    def _materialize_bytes(self):
        """Reuse the original result's conservative materialization allowance."""
        return self.source._materialize_bytes

    def cycles(self, numbers):
        """Select original cycle labels before applying the same transformation."""
        return SummaryView(self.source.cycles(numbers), self.operation, self.settings)

    def normalize(self, **settings):
        """Add per-mass and/or per-area columns lazily without replacing original columns or null masks. Example: normalize(view, mass_g=0.005, area_cm2=1.13). Output-name collisions and nonpositive denominators raise ValueError."""
        return normalize_summary(self, **settings)

    def iter_batches(self, batch_rows=512):
        """Transform bounded batches; cycle pairing carries one group across input batches."""
        if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows < 1:
            raise ValueError("Positive batch_rows required")
        available = self._budget.limit - self._budget.used - 2 * 1024**2
        batch_rows = min(batch_rows, available // 16384, max(1, self._materialize_bytes // 16384))
        if batch_rows < 1:
            raise BudgetExceeded("Insufficient shared budget for derived summaries")
        reservation = batch_rows * 4096
        self._budget.reserve(reservation)
        self._workspace.active_readers = getattr(self._workspace, "active_readers", 0) + 1
        reader = self.source.iter_batches(batch_rows=batch_rows)
        try:
            if self.operation == "normalization":
                for block in reader:
                    yield _normalize_block(block, self.settings)
                return
            schema = _cycle_schema(self.report["include_energy"])
            pending, output = [], []
            for block in reader:
                for row in block.to_pylist():
                    if pending and row["cycle"] != pending[0]["cycle"]:
                        output.append(_cycle_row(pending, self.report["include_energy"]))
                        pending = []
                    pending.append(row)
                    if len(pending) > 2:
                        raise ValueError("Cycle summary expects at most two confirmed half-cycles")
                    if len(output) == batch_rows:
                        yield pa.RecordBatch.from_pylist(output, schema=schema)
                        output = []
            if pending:
                output.append(_cycle_row(pending, self.report["include_energy"]))
            if output:
                yield pa.RecordBatch.from_pylist(output, schema=schema)
            else:
                yield pa.RecordBatch.from_pylist([], schema=schema)
        finally:
            reader.close()
            self._workspace.active_readers -= 1
            self._budget.release(reservation)

    def table(self, max_rows=100000):
        """Materialize a guarded pandas table; returned frames become caller-owned."""
        if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 0:
            raise ValueError("Nonnegative max_rows required")
        self._budget.reserve(self._materialize_bytes)
        reader, blocks, count, size = self.iter_batches(), [], 0, 0
        try:
            for block in reader:
                count += block.num_rows
                size += block.nbytes
                if count > max_rows or size * 3 > self._materialize_bytes:
                    raise MaterializationLimitError(
                        "Derived table exceeds limits; use iter_batches()"
                    )
                blocks.append(block)
            return pa.Table.from_batches(blocks).to_pandas()
        finally:
            reader.close()
            self._budget.release(self._materialize_bytes)

    def write_parquet(self, path):
        """Stream derived columns to Parquet with human-readable sidecar metadata."""
        return atomic_parquet(self.iter_batches(), path, manifest=self.report)

    def close(self):
        """Close shared source storage; all sibling views become unavailable."""
        self.source.close()


def _normalize_block(block, settings):
    """Append independent unit-labelled divisions without mutating existing arrays."""
    names = tuple(block.schema.names)
    for name in names:
        if not (name.endswith("_mAh") or name.endswith("_Wh")):
            continue
        for basis, suffix in (("mass_g", "g"), ("area_cm2", "cm2")):
            if settings[basis] is None:
                continue
            target = f"{name}_{suffix}"
            if target in block.schema.names:
                raise ValueError(f"Normalization would overwrite existing column {target}")
            block = block.append_column(target, pc.divide(block.column(name), settings[basis]))
    return block


def normalize_summary(
    source, *, mass_g=None, area_cm2=None, mass_basis="user-supplied", area_basis="user-supplied"
):
    """Add mAh/g, Wh/g, mAh/cm2 and Wh/cm2 using explicit positive finite bases.

    Both denominators may be supplied. Basis descriptions document the physical
    interpretation; no active-material or geometric-area assumption is inferred.
    Quality columns and every preexisting value remain unchanged.
    """
    if mass_g is None and area_cm2 is None:
        raise ValueError("Supply mass_g and/or area_cm2")
    for name, value in (("mass_g", mass_g), ("area_cm2", area_cm2)):
        if value is not None and (
            isinstance(value, bool) or not math.isfinite(value) or value <= 0
        ):
            raise ValueError(f"{name} must be positive and finite")
    for description in (mass_basis, area_basis):
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Physical basis descriptions must be nonempty strings")
    return SummaryView(
        source,
        "normalization",
        dict(mass_g=mass_g, area_cm2=area_cm2, mass_basis=mass_basis, area_basis=area_basis),
    )
