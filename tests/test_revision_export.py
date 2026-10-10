"""Synthetic regression tests; run with tools/run_tests.py. Passing verifies software behavior, not experimental electrochemistry."""

import json

import numpy as np
import pytest

from gregorithm import (
    AnalysisConfig,
    CancellationToken,
    StepCorrection,
    TraceCollection,
    analyze,
    correct_steps,
    detect_events,
    reopen_result,
)


def test_manual_modes_and_revision(make_trace):
    """Verify manual modes and revision using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    trace = make_trace(np.arange(30), np.ones(30), np.linspace(3, 4, 30))
    cfg = AnalysisConfig(backend="numpy", batch_rows=7)
    old = analyze(trace, config=cfg)
    correction = StepCorrection(10, 19, "CV", "Manual protocol annotation", 0)
    index = correct_steps(trace, [correction], config=cfg)
    new = analyze(trace, config=cfg)
    assert old.assignment_revision != new.assignment_revision == index.revision
    data = new.samples.to_pandas()
    assert data.loc[10:19, "mode"].tolist() == ["CV"] * 10
    assert new.samples.to_pandas().global_charge_mAh.iloc[-1] == pytest.approx(
        old.samples.to_pandas().global_charge_mAh.iloc[-1], abs=1e-12
    )
    assert {"trace_id", "result_id", "step_id", "half_cycle_id", "cycle_id"} <= set(
        new.segments.to_pandas()
    )
    assert new.manifest["step_corrections"][0]["reason"] == correction.reason
    assert (
        reopen_result(new.directory).samples.to_pandas()["mode"].tolist() == data["mode"].tolist()
    )


def test_selected_export_and_lazy_reopen(make_trace):
    """Verify selected export and lazy reopen using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    trace = make_trace(np.arange(30), np.repeat([-1, 1, -1], 10))
    cfg = AnalysisConfig(backend="numpy")
    result = analyze(trace.select(cycle_id=2).head(3), config=cfg, export_samples=True)
    assert reopen_result(result.directory).samples.to_pandas().sample_id.tolist() == [10, 11, 12]
    lazy = analyze(trace, config=cfg, export_samples=False)
    assert len(reopen_result(lazy.directory).samples.to_pandas()) == 30
    assert "source_index" in lazy.manifest["files"]


def test_event_sample_selection(make_trace):
    """Verify event sample selection using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    t = np.arange(250, dtype=float)
    current = np.r_[np.ones(100), np.zeros(5), -np.ones(10), np.zeros(5), np.ones(130)]
    trace = make_trace(t, current, 4 + 0.01 * current)
    cfg = AnalysisConfig(backend="numpy")
    events = detect_events(trace, cfg)
    pulse = events.select(candidate_status="candidate").events[0]
    result = analyze(trace, config=cfg, events=events, export_samples=True)
    selected = result.samples.select(event_id=pulse["event_id"]).to_pandas()
    assert selected.sample_id.iloc[0] == pulse["sample_t0"]
    assert selected.sample_id.iloc[-1] == pulse["sample_t3"]
    assert "result_id" in result.events.to_pandas()


def test_parallel_cooperative_cancellation(make_trace):
    """Verify parallel cooperative cancellation using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    a = make_trace(np.arange(2000), np.ones(2000))
    b = make_trace(np.arange(2000), np.ones(2000))
    token = CancellationToken()
    progress = []

    def callback(event):
        """Capture test progress and trigger the explicitly prescribed cooperative cancellation condition."""
        progress.append(event)
        if event.trace_id is not None:
            token.cancel()

    result = analyze(
        TraceCollection((a, b)),
        config=AnalysisConfig(
            backend="numpy", workers=2, batch_rows=7, managed_memory_bytes=256 * 1024**2
        ),
        cancellation=token,
        on_progress=callback,
    )
    assert set(result.by_trace) == {a.trace_id, b.trace_id}
    assert any(r.status == "cancelled" for r in result.by_trace.values())
    assert progress[-1].status == "cancelled"
    for event in progress:
        json.dumps(event.to_dict())
