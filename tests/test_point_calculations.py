"""Independent mathematical regressions for global identity-free integration."""

from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.integrate import quad

from gregorithm import MemoryBudget, ReadOptions, prepare_source
from gregorithm.point_calculations import (
    EstimatedMeasurementWarning,
    IncompleteIntegrationWarning,
    directional_integrals,
    estimate_isolated,
)


def test_exact_energy_and_sign_split_against_quadrature():
    """Compare Q/E with independent quadrature, including opposite signs and signed V."""
    rng = np.random.default_rng(2026)
    for _ in range(80):
        a, b, va, vb = rng.uniform(-10, 10, 4)
        dt = rng.uniform(0.01, 1000)
        actual = directional_integrals([0, dt], [a, b], [va, vb])[0]
        root = -a / (b - a)
        cuts = [0, root, 1] if 0 < root < 1 else [0, 1]
        expected = np.zeros(4)
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            sign = 1 if a + (b - a) * (lo + hi) / 2 >= 0 else -1
            q = dt * quad(lambda u: a + (b - a) * u, lo, hi)[0] / 3600
            e = dt * quad(lambda u: (a + (b - a) * u) * (va + (vb - va) * u), lo, hi)[0] / 3600000
            expected[0 if sign > 0 else 1] += sign * q
            expected[2 if sign > 0 else 3] += sign * e
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_energy_is_not_endpoint_power_trapezoid():
    """The product of two linear signals is integrated as a quadratic."""
    actual = directional_integrals([0, 10], [1, 3], [2, 6])[0]
    assert actual[2] == pytest.approx(10 * (2 + 4 + 8 / 3) / 3600000)
    assert actual[2] != pytest.approx(10 * (2 + 18) / 2 / 3600000)


def test_isolated_estimates_do_not_fill_runs_or_endpoints():
    """Immediate real neighbors and a bracket <=10 s are required; inputs stay unchanged."""
    original = np.array([1, np.nan, 3, np.nan, np.nan, 6, np.nan])
    estimated, mask = estimate_isolated([0, 5, 10, 11, 12, 13, 14], original)
    assert estimated[1] == 2
    assert mask.tolist() == [False, True, False, False, False, False, False]
    assert np.isnan(estimated[3:5]).all() and np.isnan(estimated[-1])
    assert np.isnan(original[1])
    rejected, flags = estimate_isolated([0, 5, 10.01], [1, np.nan, 3])
    assert np.isnan(rejected[1]) and not flags.any()


@pytest.mark.parametrize("batch_rows", [1, 2, 3, 100])
def test_estimation_cross_batch_invariance(tmp_path, batch_rows):
    """Carry original neighbors and effective integration endpoints across row groups."""
    path = tmp_path / "data.parquet"
    original = pa.table(
        {
            "time_s": [0.0, 1.0, 2.0, 3.0, 10.0, 30.0, 31.0, 32.0],
            "current_mA": [1.0, np.nan, 3.0, -1.0, -1.0, np.nan, np.nan, -1.0],
            "voltage_V": [2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0],
            "note": ["original"] * 8,
        }
    )
    pq.write_table(original, path, row_group_size=2)
    source = prepare_source(
        path, tmp_path / "prepared.parquet", read_options=ReadOptions(batch_rows=batch_rows)
    )
    budget = MemoryBudget(8 * 1024**2)
    with pytest.warns((EstimatedMeasurementWarning, IncompleteIntegrationWarning)):
        actual = source.totals(memory_budget=budget)
    assert budget.used == 0
    assert actual.report["counts"]["estimated_current_points"] == 1
    assert actual.report["counts"]["estimated_voltage_points"] == 1
    assert actual.values["capacity_quality"] == actual.values["energy_quality"] == "partial"
    expected = directional_integrals([0, 1, 2, 3, 10], [1, 2, 3, -1, -1], [2, 3, 4, 5, 6]).sum(
        axis=0
    )
    np.testing.assert_allclose(
        [
            actual.values[key]
            for key in ("charge_mAh", "discharge_mAh", "charge_Wh", "discharge_Wh")
        ],
        expected,
        atol=1e-12,
    )
    assert actual.values["capacity_covered_s"] == 10
    retained = pq.read_table(source.path)
    for name in ("time_s", "current_mA", "voltage_V"):
        np.testing.assert_array_equal(retained[name].to_numpy(), original[name].to_numpy())
    assert retained["note"].to_pylist() == original["note"].to_pylist()
    assert not set(actual.table().columns) & {"sample_id", "trace_id", "result_id"}


def test_capacity_only_has_no_voltage_dependency_or_gap_cutoff(tmp_path):
    """Fully measured long intervals stay valid; Q-only work needs no V column."""
    path = tmp_path / "current.parquet"
    pq.write_table(pa.table({"time_s": [0.0, 3600.0], "current_mA": [1.0, 1.0]}), path)
    source = prepare_source(path, tmp_path / "prepared.parquet")
    result = source.totals(include_energy=False)
    assert result.values["charge_mAh"] == 1
    assert result.values["capacity_quality"] == "complete"
    assert "charge_Wh" not in result.values


def test_missing_voltage_only_affects_energy(tmp_path):
    """Absent V does not erase valid Q or become zero energy."""
    path = tmp_path / "data.parquet"
    pq.write_table(
        pa.table({"time_s": [0.0, 1.0], "current_mA": [1.0, 1.0], "voltage_V": [np.nan, np.nan]}),
        path,
    )
    source = prepare_source(path, tmp_path / "prepared.parquet")
    with pytest.warns(IncompleteIntegrationWarning):
        result = source.totals()
    assert result.values["capacity_quality"] == "complete"
    assert result.values["charge_mAh"] == pytest.approx(1 / 3600)
    assert result.values["energy_quality"] == "unavailable"
    assert result.values["charge_Wh"] is None


def test_one_point_is_not_zero_capacity(tmp_path):
    """No observed interval gives unavailable quantities, not fabricated zeros."""
    path = tmp_path / "point.parquet"
    pq.write_table(pa.table({"time_s": [0.0], "current_mA": [1.0], "voltage_V": [4.0]}), path)
    source = prepare_source(path, tmp_path / "prepared.parquet")
    result = source.totals()
    assert result.values["net_mAh"] is None
    assert result.values["capacity_quality"] == "unavailable"


def test_irregular_mixed_sign_points_batch_invariant(tmp_path):
    """Compare totals across batch sizes without assuming a fixed sampling frequency."""
    rng = np.random.default_rng(7)
    t = np.cumsum(rng.uniform(0.1, 15, 1000))
    i = rng.uniform(-100, 100, 1000)
    v = rng.uniform(-1, 5, 1000)
    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"time_s": t, "current_mA": i, "voltage_V": v}), path)
    source = prepare_source(path, tmp_path / "prepared.parquet")
    first = replace(source, read_options=ReadOptions(batch_rows=17)).totals()
    second = replace(source, read_options=ReadOptions(batch_rows=251)).totals()
    for name in ("charge_mAh", "discharge_mAh", "net_mAh", "charge_Wh", "discharge_Wh", "net_Wh"):
        assert first.values[name] == pytest.approx(second.values[name], rel=1e-12, abs=1e-12)
    assert first.report["counts"] == second.report["counts"]
