"""Synthetic regression tests; run with tools/run_tests.py. Passing verifies software behavior, not experimental electrochemistry."""

import numpy as np
import pytest

from gregorithm import *
from gregorithm.integration import integrated_batches


def values(trace, cfg):
    """Support values within this local verification helper; use the containing test/script rather than this private helper directly."""
    batches = list(integrated_batches(trace, trace.ensure_index(cfg), cfg))
    import pyarrow as pa

    return pa.Table.from_batches(batches).to_pandas()


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_integrals(backend):
    """Verify integrals using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    cfg = AnalysisConfig(backend=backend)
    a, b = calculate_increments([0, 3600], [1, 1], [4, 4], cfg)
    np.testing.assert_allclose(a[-1], [1, 0, 0.004, 0], rtol=1e-10, atol=1e-12)
    a, b = calculate_increments([0, 2], [1, -1], [4, 4], cfg)
    np.testing.assert_allclose(a[-1, :2], [0.5 / 3600, 0.5 / 3600], atol=1e-12)
    assert a[-1, 0] - a[-1, 1] == 0


@pytest.mark.parametrize("method", ["left", "right", "linear_signals"])
def test_energy_product(method):
    """Verify energy product using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    out = interval_parts(0, 1, 2, 6, 3, 4, method)
    expected = (
        6 * (2 + (4 + 2) / 2 + 4 / 3) / 3600000
        if method == "linear_signals"
        else 12 / 3600000
        if method == "left"
        else 72 / 3600000
    )
    assert out[2] == pytest.approx(expected)
    a = np.array(interval_parts(0, 1, 2, 6, 3, 4, method, 0, 0.3)) + np.array(
        interval_parts(0, 1, 2, 6, 3, 4, method, 0.3, 1)
    )
    np.testing.assert_allclose(a, out, atol=1e-12)


@pytest.mark.parametrize("batch", [1, 7, 1024, 262144])
@pytest.mark.parametrize(
    "order,expected",
    [("charge_then_discharge", [1, 2, 2, 3, 3]), ("discharge_then_charge", [1, 1, 2, 2, 3])],
)
def test_cycles_and_batches(make_trace, batch, order, expected):
    """Verify cycles and batches using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    trace = make_trace(np.arange(15), np.repeat([-1, 1, -1, 1, -1], 3))
    cfg = AnalysisConfig(batch_rows=batch, cycle_order=order, backend="numpy")
    data = values(trace, cfg)
    assert data.half_cycle_id.tolist() == np.repeat([1, 2, 3, 4, 5], 3).tolist()
    assert data.cycle_id.tolist() == np.repeat(expected, 3).tolist()
    assert data.global_net_mAh.iloc[-1] == pytest.approx(
        np.nansum(data.d_charge_mAh - data.d_discharge_mAh)
    )


def test_rest_cv_unconfirmed_and_reset(make_trace):
    """Verify rest cv unconfirmed and reset using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    trace = make_trace(
        [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2], [0, 0.005, 0.005, 0, 0.004, 0.003, -1, 0, 0, -1, -1]
    )
    d = values(trace, AnalysisConfig(batch_rows=1, backend="numpy"))
    assert d.half_cycle_id.tolist() == [0, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2]
    assert d.direction.iloc[1] == "charge"
    assert not d.q_interval_valid.iloc[8]
    assert d.d_charge_mAh.iloc[1] > 0.0


def test_nan_duplicate_gap(make_trace):
    """Verify nan duplicate gap using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    trace = make_trace([0, 1, 1, 2, 8], [1, np.nan, 1, 1, 1], [4, 4, 4, np.nan, 4])
    d = values(trace, AnalysisConfig(max_gap_s=2, backend="numpy"))
    assert not d.q_interval_valid.iloc[1]
    assert not d.q_interval_valid.iloc[2]
    assert d.q_interval_valid.iloc[3]
    assert not d.e_interval_valid.iloc[3]
    assert not d.q_interval_valid.iloc[4]
    assert d.coverage_partial.iloc[-1]


def test_empty_and_one(make_trace):
    """Verify empty and one using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    for t, i in [([], []), ([0], [1])]:
        trace = make_trace(t, i)
        result = analyze(trace, config=AnalysisConfig(backend="numpy"))
        assert result.status == "complete"


def test_density_invariant():
    """Verify density invariant using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    cfg = AnalysisConfig(backend="numpy")
    a, _ = calculate_increments([0, 2, 7], [1, -1, 3], [4, 3, 5], cfg)
    t = np.array([0, 1, 2, 3, 4, 5, 6, 7.0])
    i = np.interp(t, [0, 2, 7], [1, -1, 3])
    v = np.interp(t, [0, 2, 7], [4, 3, 5])
    b, _ = calculate_increments(t, i, v, cfg)
    np.testing.assert_allclose(a.sum(axis=0), b.sum(axis=0), atol=1e-12, rtol=1e-10)
