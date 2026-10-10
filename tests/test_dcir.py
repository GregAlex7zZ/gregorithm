"""Synthetic regression tests; run with tools/run_tests.py. Passing verifies software behavior, not experimental electrochemistry."""

import numpy as np
import pytest

from gregorithm import *


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("delay", [11.0, 12.0])
def test_dcir_signed_baseline_and_interpolation(make_trace, sign, delay):
    """Verify dcir signed baseline and interpolation using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    t = [0, 1, 3, 10, 11, 13, 20, 21]
    i = [0.00005, 0.00005, sign * 100, sign * 100, sign * 100, sign * 100, 0.00005, 0.00005]
    v = [4, 4, *([4 + sign * 0.1] * 4), 4, 4]
    tr = make_trace(t, i, v)
    events = manual_event(tr, sample_t0=1, sample_t1=2, sample_t2=5, sample_t3=6)
    before = events.to_dict()
    result = measure_dcir(tr, events, delay).to_pandas().iloc[0]
    assert result.valid and result.R_ohm == pytest.approx(
        1000 * (sign * 0.1) / (sign * 100 - 0.00005)
    )
    assert result.target_s == 1 + delay
    assert events.to_dict() == before and not events.events[0]["excluded"]
    other = measure_dcir(tr, events, delay).to_pandas()
    assert other.measurement_id.iloc[0] != result.measurement_id
    bad = measure_dcir(tr, events, 0.5).to_pandas().iloc[0]
    assert not bad.valid and bad.reason == "target_outside_active_support"


def test_dcir_nan_and_gap_and_nearest(make_trace):
    """Verify dcir nan and gap and nearest using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    tr = make_trace([0, 1, 2, 8, 15, 16], [0, 0, -100, -100, -100, 0], [4, 4, 3.9, np.nan, 3.9, 4])
    ev = manual_event(tr, sample_t0=1, sample_t1=2, sample_t2=4, sample_t3=5)
    assert not measure_dcir(tr, ev, 4).to_pandas().valid.iloc[0]
    assert (
        measure_dcir(tr, ev, 4, max_interp_gap_s=1).to_pandas().reason.iloc[0]
        == "interpolation_gap_exceeded"
    )
    assert measure_dcir(
        tr, ev, 1, method="nearest", nearest_tolerance_s=0.01
    ).to_pandas().R_ohm.iloc[0] == pytest.approx(1)
    with pytest.raises(ValueError):
        measure_dcir(tr, ev, [11, 12])
    with pytest.raises(ValueError):
        measure_dcir(tr, ev, 11, method="nearest")


def test_detection_symmetry_and_exclusion(make_trace):
    """Verify detection symmetry and exclusion using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    t = np.arange(250, dtype=float)
    i = np.r_[np.ones(100), np.zeros(5), -np.ones(10), np.zeros(5), np.ones(130)]
    events = []
    for sign in (-1, 1):
        tr = make_trace(t, i * sign, 4 + i * sign * 0.01)
        cfg = AnalysisConfig(backend="numpy", batch_rows=7)
        ev = detect_events(tr, cfg).select(candidate_status="candidate")
        assert len(ev.events) == 1 and ev.events[0]["candidate_status"] == "candidate"
        assert not ev.events[0]["excluded"]
        accepted = ev.decide(
            ev.events[0]["event_id"], "accepted", exclude=True, reason="Synthetic test known pulse"
        )
        original = analyze(tr, config=cfg)
        corrected = analyze(tr, config=cfg, events=accepted)
        assert corrected.samples.to_pandas().half_cycle_id.max() == 1
        assert original.samples.to_pandas().half_cycle_id.max() == 3
        a = original.samples.to_pandas().global_throughput_mAh.iloc[-1]
        b = corrected.samples.to_pandas().global_throughput_mAh.iloc[-1]
        assert a == pytest.approx(b, rel=1e-10, abs=1e-12)
        assert original.assignment_revision != corrected.assignment_revision
        assert ev.events[0]["event_id"] == accepted.events[0]["event_id"]
        events.append(ev.events[0])
    assert events[0]["relation_to_before"] == events[1]["relation_to_before"] == "opposite"


def test_gitt_ici_and_terminal_ambiguity(make_trace):
    """Verify gitt ici and terminal ambiguity using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    tr = make_trace(np.arange(60), np.tile(np.r_[np.ones(8), np.zeros(2)], 6))
    ev = detect_events(tr, AnalysisConfig(backend="numpy"))
    assert not any(e["auto_eligible"] for e in ev.events)
    assert tr.select(half_cycle_id=1).to_pandas().half_cycle_id.max() == 1
    tr = make_trace(
        np.arange(230), np.r_[np.ones(100), np.zeros(5), -np.ones(10), np.zeros(5), -np.ones(110)]
    )
    ev = detect_events(tr, AnalysisConfig(backend="numpy"))
    candidates = ev.select(candidate_status="candidate")
    assert len(candidates.events) == 1 and not candidates.events[0]["auto_eligible"]
    assert not candidates.auto_accept().events[0]["excluded"]
