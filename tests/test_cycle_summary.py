"""Synthetic net-pair and additive normalization checks, not experimental validation."""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from test_half_cycles import prepared

from gregorithm import HalfCycleConfig, MemoryBudget


@pytest.mark.parametrize("order", ["charge_then_discharge", "discharge_then_charge"])
def test_efficiency_uses_net_quantities_and_fixed_discharge_over_charge(tmp_path, order):
    """Both pairing orders use discharge/charge, not direction-order ratios or gross Q."""
    current = [2, 2, -1, -1] if order == "charge_then_discharge" else [-1, -1, 2, 2]
    source = prepared(tmp_path, [0, 2, 5, 7], current)
    result = source.capacity(include_energy=True, config=HalfCycleConfig(cycle_order=order))
    try:
        half = result.table().set_index("direction")
        summary = result.summary().table().iloc[0]
        expected = 100 * half.loc["discharge", "capacity_mAh"] / half.loc["charge", "capacity_mAh"]
        assert summary.coulombic_efficiency_percent == pytest.approx(expected)
        assert summary.energy_efficiency_percent == pytest.approx(expected)
        assert summary.coulombic_efficiency_quality == "complete"
        assert summary.cycle_pair_complete
        assert result.summary().cycles([]).table().empty
    finally:
        result.close()


def test_initial_unpaired_cycle_has_no_fabricated_efficiency(tmp_path):
    """Incomplete prefix remains visible while later clean pairs remain eligible."""
    result = prepared(tmp_path, range(6), [-1, -1, 1, 1, -1, -1]).capacity()
    try:
        frame = result.summary().table()
        assert frame.cycle.tolist() == [1, 2]
        assert np.isnan(frame.coulombic_efficiency_percent.iloc[0])
        assert frame.coulombic_efficiency_reason.iloc[0] == "missing_counterpart"
        assert frame.coulombic_efficiency_quality.iloc[1] == "complete"
        assert "energy_efficiency_percent" not in frame
        batches = list(result.summary().iter_batches(batch_rows=1))
        assert sum(block.num_rows for block in batches) == 2
    finally:
        result.close()


def test_voltage_gap_blocks_only_energy_efficiency(tmp_path):
    """Unsupported voltage retains Q and known E while suppressing only full energy ratio."""
    with pytest.warns(UserWarning):
        result = prepared(
            tmp_path, range(8), [1] * 4 + [-1] * 4, [4, np.nan, np.nan, 4, 4, 4, 4, 4]
        ).capacity(include_energy=True)
    try:
        row = result.summary().table().iloc[0]
        assert row.coulombic_efficiency_quality == "complete"
        assert row.energy_efficiency_quality == "unavailable"
        assert row.energy_efficiency_reason == "incomplete_support"
        assert np.isnan(row.energy_efficiency_percent)
        assert row.charge_energy_Wh > 0
    finally:
        result.close()


def test_estimates_propagate_to_efficiency(tmp_path):
    """Eligible isolated-point estimation yields explicitly estimated paired efficiency."""
    with pytest.warns(UserWarning):
        result = prepared(tmp_path, range(7), [1, 1, np.nan, 1, -1, -1, -1]).capacity()
    try:
        assert result.summary().table().coulombic_efficiency_quality.tolist() == ["estimated"]
    finally:
        result.close()


@pytest.mark.parametrize(
    "settings", [{"mass_g": 0.005}, {"area_cm2": 2}, {"mass_g": 0.005, "area_cm2": 2}]
)
def test_normalization_preserves_all_original_columns_and_quality(tmp_path, settings):
    """Mass/area divisions are independent and additive for every detailed Q/E column."""
    result = prepared(tmp_path, range(4), [1, 1, -1, -1]).capacity(include_energy=True)
    try:
        original = result.details().table()
        normalized = result.details().normalize(**settings).table()
        pd.testing.assert_frame_equal(normalized[original.columns], original)
        for basis, suffix in (("mass_g", "g"), ("area_cm2", "cm2")):
            if basis in settings:
                for name in original.columns:
                    if name.endswith(("_mAh", "_Wh")):
                        np.testing.assert_allclose(
                            normalized[f"{name}_{suffix}"], original[name] / settings[basis]
                        )
        summary = result.summary().normalize(**settings)
        assert summary.cycles([]).table().empty
        path = tmp_path / "normalized.parquet"
        summary.write_parquet(path)
        assert pq.read_table(path).num_rows == 1
    finally:
        result.close()


def test_normalization_budget_lifecycle_limits_and_collisions(tmp_path):
    """Nested lazy views release reservations on success, rejection and iterator closure."""
    from gregorithm import MaterializationLimitError

    budget = MemoryBudget(8 * 1024**2)
    result = prepared(tmp_path, range(4), [1, 1, -1, -1]).capacity(memory_budget=budget)
    view = result.summary().normalize(mass_g=1)
    with pytest.raises(MaterializationLimitError):
        view.table(max_rows=0)
    assert budget.used == 0

    iterator = view.iter_batches(batch_rows=1)
    next(iterator)
    with pytest.raises(ValueError, match="active"):
        result.close()
    iterator.close()
    assert budget.used == 0
    with pytest.raises(ValueError, match="overwrite"):
        result.normalize(mass_g=1).normalize(mass_g=2).table()
    assert budget.used == 0
    for settings in ({}, {"mass_g": 0}, {"area_cm2": float("nan")}, {"mass_g": True}):
        with pytest.raises(ValueError):
            result.normalize(**settings)
    view.close()
    with pytest.raises(ValueError, match="closed"):
        view.table()
    assert budget.used == 0


def test_nonpositive_net_denominator_and_unclamped_efficiency(tmp_path):
    """A retained short reverse excursion can invalidate net charge; >100% stays visible."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    result = prepared(first, range(7), [1, 1, -100, 1, 1, -1, -1]).capacity()
    try:
        row = result.summary().table().iloc[0]
        assert row.charge_capacity_mAh < 0
        assert np.isnan(row.coulombic_efficiency_percent)
        assert row.coulombic_efficiency_reason == "nonpositive_charge"
    finally:
        result.close()
    result = prepared(second, [0, 2, 5, 7], [1, 1, -2, -2]).capacity()
    try:
        assert result.summary().table().coulombic_efficiency_percent.iloc[0] > 100
        normalized = result.normalize(mass_g=0.005, mass_basis="active material").normalize(
            area_cm2=2, area_basis="geometric"
        )
        settings = normalized.report["view_operations"]
        assert settings[0]["settings"]["mass_basis"] == "active material"
        assert settings[1]["settings"]["area_basis"] == "geometric"
        assert normalized.cycles([1]).report["selected_cycles"] == (1,)
        assert "capacity_mAh_g" in normalized.table()
        assert "capacity_mAh_cm2" in normalized.table()
    finally:
        result.close()
