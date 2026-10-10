"""Exact selected-time integration tests using point-signal boundaries."""

import numpy as np
import pytest
from test_half_cycles import prepared

from gregorithm import MemoryBudget, interval_parts


@pytest.mark.parametrize("batch", [1, 2, 100])
def test_exact_clipped_boundaries_match_analytic_integral(tmp_path, batch):
    """Clip inside original intervals without rounding to recorded timestamps."""
    source = prepared(
        tmp_path,
        [0, 4, 10],
        [-2, 2, 4],
        [3, 4, 5],
        batch_rows=batch,
    )
    result = source.time_window(1, 7)
    expected_a = interval_parts(0, -2, 3, 4, 2, 4, lo=0.25, hi=1)
    expected_b = interval_parts(4, 2, 4, 10, 4, 5, lo=0, hi=0.5)
    expected = np.array(expected_a) + np.array(expected_b)
    assert result.values["charge_mAh"] == pytest.approx(expected[0])
    assert result.values["discharge_mAh"] == pytest.approx(expected[1])
    assert result.values["charge_Wh"] == pytest.approx(expected[2])
    assert result.values["discharge_Wh"] == pytest.approx(expected[3])
    assert result.values["duration_s"] == pytest.approx(6)
    assert result.report["start_s"] == 1 and result.report["end_s"] == 7


def test_window_missing_support_is_partial_and_never_connected(tmp_path):
    """Retain known clipped contributions while reporting unavailable internal support."""
    source = prepared(tmp_path, range(7), [1, 1, np.nan, np.nan, 1, 1, 1])
    with pytest.warns(UserWarning):
        result = source.time_window(0.5, 5.5, include_energy=False)
    assert result.values["capacity_quality"] == "partial"
    assert result.values["capacity_covered_s"] == pytest.approx(2.0)
    assert result.values["duration_s"] == pytest.approx(5)


def test_window_validation_and_budget_release(tmp_path):
    """Reject extrapolated/reversed windows and release the shared managed budget."""
    source = prepared(tmp_path, [0, 1, 2], [1, 1, 1])
    budget = MemoryBudget(8 * 1024**2)
    result = source.time_window(0.25, 1.75, memory_budget=budget)
    assert result.values["net_mAh"] == pytest.approx(1.5 / 3600)
    assert budget.used == 0
    with pytest.raises(ValueError, match="inside observed"):
        source.time_window(-1, 1)
    with pytest.raises(ValueError, match="later"):
        source.time_window(1, 1)
