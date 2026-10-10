"""Synthetic regression tests; run with tools/run_tests.py. Passing verifies software behavior, not experimental electrochemistry."""

import numpy as np
import pyarrow as pa
import pytest

from gregorithm import DQDVConfig, compute_dqdv
from gregorithm.dqdv import GridNeighbors
from gregorithm.io import TableView


@pytest.mark.parametrize("policy", ["first", "last", "mean"])
@pytest.mark.parametrize("batch", [1, 7, 1024])
def test_neighbors_exact(policy, batch):
    """Verify neighbors exact using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    rng = np.random.default_rng(42)
    v = rng.integers(0, 101, 1000) / 100
    q = rng.normal(size=1000)
    ids = np.arange(1000)
    grid = np.linspace(0, 1, 173)
    r = GridNeighbors(grid, policy)
    for start in range(0, len(v), batch):
        r.update(v[start : start + batch], q[start : start + batch], ids[start : start + batch])
    order = np.argsort(v, kind="stable")
    vs = v[order]
    qs = q[order]
    unique, starts, counts = np.unique(vs, return_index=True, return_counts=True)
    expected = (
        qs[starts]
        if policy == "first"
        else qs[starts + counts - 1]
        if policy == "last"
        else np.add.reduceat(qs, starts) / counts
    )
    actual, w = r.interpolate()
    np.testing.assert_allclose(actual, np.interp(grid, unique, expected), rtol=1e-10, atol=1e-12)


def curve(direction="charge", n=101):
    """Construct an affine synthetic Q(V) branch for exact derivative/interpolation tests, with configurable direction and sample count."""
    v = np.linspace(3, 4, n)
    q = 2 * v + 1
    if direction == "discharge":
        v = v[::-1]
        q = -2 * v + 9
    b = pa.RecordBatch.from_pydict(
        {
            "voltage_V": v,
            "capacity_mAh": q,
            "current_mA": np.full(n, 1 if direction == "charge" else -1),
            "sample_id": np.arange(n),
            "half_cycle_id": np.ones(n, dtype=np.int64),
            "cycle_id": np.ones(n, dtype=np.int64),
            "trace_id": ["synthetic"] * n,
            "direction": [direction] * n,
        }
    )
    return TableView(lambda: iter([b]))


@pytest.mark.parametrize("direction,slope", [("charge", 2), ("discharge", -2)])
def test_affine_curve(direction, slope):
    """Verify affine curve using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    result = compute_dqdv(curve(direction), config=DQDVConfig(trim_extremes_V=0))
    np.testing.assert_allclose(result.to_pandas().dqdv_mAh_V, slope, atol=1e-12)
    assert result.manifest["profile"] == "legacy_corrected" and result.manifest["n_effective"] == 10
    assert not result.manifest["dqdv_config"]["smoothing"]
    assert "sample_id" not in result.to_pandas()
    alt = compute_dqdv(
        curve(direction),
        config=DQDVConfig(profile="voltage_grid", voltage_step_V=0.03, trim_extremes_V=0),
    )
    assert alt.to_pandas().voltage_V.iloc[-1] < 4


def test_default_repeated_voltage_policy_is_approved_mean():
    """Use mean cumulative capacity at identical potentials unless explicitly overridden."""
    assert DQDVConfig().duplicates == "mean"


@pytest.mark.parametrize(
    "cap,expected,limited", [(20, 20, True), (200, 101, False), (None, 101, False)]
)
def test_custom_automatic_grid_cap(cap, expected, limited):
    """Verify configurable and disabled caps preserve an affine derivative and report the effective grid."""
    result = compute_dqdv(
        curve(), config=DQDVConfig(fraction=1, max_auto_points=cap, trim_extremes_V=0)
    )
    assert result.manifest["n_effective"] == expected
    assert result.manifest["auto_grid_limited"] is limited
    np.testing.assert_allclose(result.to_pandas().dqdv_mAh_V, 2, atol=1e-10)


@pytest.mark.parametrize("cap", [2, 3.5, True])
def test_invalid_automatic_grid_cap(cap):
    """Reject noninteger or undersized numerical caps before constructing a curve."""
    with pytest.raises(ValueError, match="max_auto_points"):
        DQDVConfig(max_auto_points=cap)


def test_smoothing_preserves_comparison_curve():
    """Preserve the independent unsmoothed result and report the physical filter width."""
    batch = next(curve().iter_batches())
    voltage = batch.column(batch.schema.get_field_index("voltage_V")).to_numpy()
    capacity = 2 * voltage + 0.01 * np.sin(100 * voltage)
    batch = batch.set_column(
        batch.schema.get_field_index("capacity_mAh"), "capacity_mAh", pa.array(capacity)
    )
    view = TableView(lambda: iter([batch]))
    raw = compute_dqdv(view, config=DQDVConfig(n_points=21, trim_extremes_V=0))
    filtered = compute_dqdv(view, config=DQDVConfig(n_points=21, trim_extremes_V=0, smoothing=True))
    actual = filtered.to_pandas()
    np.testing.assert_allclose(actual.dqdv_unsmoothed_mAh_V, raw.to_pandas().dqdv_mAh_V)
    np.testing.assert_allclose(actual.capacity_unsmoothed_mAh, raw.to_pandas().capacity_mAh)
    assert not np.allclose(actual.dqdv_mAh_V, actual.dqdv_unsmoothed_mAh_V)
    assert filtered.manifest["smoothing_width_V"] == pytest.approx(0.2)


def test_config_and_smoothing():
    """Verify config and smoothing using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    with pytest.raises(ValueError):
        DQDVConfig(n_points=10, fraction=0.2)
    with pytest.raises(ValueError):
        DQDVConfig(voltage_step_V=0.002)
    with pytest.raises(ValueError):
        compute_dqdv(
            curve(n=5),
            config=DQDVConfig(n_points=3, smoothing=True, window_points=5, trim_extremes_V=0),
        )
    large = compute_dqdv(curve(), config=DQDVConfig(n_points=6001, trim_extremes_V=0))
    assert len(large.to_pandas()) == 6001
