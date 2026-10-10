"""Synthetic regression tests; run with tools/run_tests.py. Passing verifies software behavior, not experimental electrochemistry."""

import json

import numpy as np
import pytest

from gregorithm import *
from gregorithm.plot_spec import PlotSpec, iter_plot_updates


def test_csv_coercion_mapping_and_decimal(tmp_path):
    """Verify csv coercion mapping and decimal using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    p = tmp_path / "data.txt"
    p.write_text("t;i;v;note\n0;0,005;4;one\n1;bad;4;two\n2;0,004;4;three\n", encoding="utf8")
    opt = ReadOptions(
        delimiter=";",
        decimal=",",
        column_mapping={"t": "time_s", "i": "current_mA", "v": "voltage_V"},
        batch_rows=1,
    )
    trace = open_trace(p, read_options=opt)
    r = analyze(trace, config=AnalysisConfig(batch_rows=1, backend="numpy"), export_samples=True)
    assert r.status == "partial"
    d = r.samples.to_pandas()
    assert d.note.tolist() == ["one", "two", "three"]
    assert np.isnan(d.current_mA.iloc[1])
    assert r.diagnostics.to_pandas().code.iloc[0] == "invalid_interval"


def test_head_filtered_and_projection_normalization(make_trace):
    """Verify head filtered and projection normalization using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    tr = make_trace(np.arange(30), np.repeat([-1, 1, -1], 10))
    r = analyze(tr.select(cycle_id=2).head(3), config=AnalysisConfig(backend="numpy"))
    assert r.samples.to_pandas().sample_id.tolist() == [10, 11, 12]
    assert not r.metrics.to_pandas().empty
    assert r.manifest["selection"]["head"] == 3
    r = analyze(
        tr.select(columns=["voltage_V"]),
        config=AnalysisConfig(backend="numpy", mass_g=0.2, reference_capacity_mAh=0.5),
        analyses=["c_rate"],
        export_samples=True,
    )
    d = r.samples.to_pandas()
    assert "voltage_V" in d and "current_mA" not in d
    reopened = reopen_result(r.directory)
    assert reopened.samples.to_pandas().columns.tolist() == d.columns.tolist()


def test_progress_terminal_and_cancel(make_trace):
    """Verify progress terminal and cancel using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    tr = make_trace(np.arange(40), np.ones(40))
    events = []
    analyze(tr, config=AnalysisConfig(batch_rows=1, backend="numpy"), on_progress=events.append)
    assert events[-1].status == "complete" and events[-1].phase == "completion"
    for event in events:
        json.dumps(event.to_dict())
    token = CancellationToken()

    def callback(e):
        """Capture test progress and trigger the explicitly prescribed cooperative cancellation condition."""
        if e.phase == "segmentation":
            token.cancel()

    tr2 = make_trace(np.arange(40), np.ones(40))
    with pytest.raises(Cancelled):
        analyze(
            tr2,
            config=AnalysisConfig(batch_rows=1, backend="numpy"),
            on_progress=callback,
            cancellation=token,
        )


def test_worker_invariance_and_file_failure(make_trace):
    """Verify worker invariance and file failure using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    a = make_trace(np.arange(9), np.repeat([-1, 1, -1], 3))
    b = make_trace(np.arange(9), np.repeat([-1, 1, -1], 3))
    sources = TraceCollection((a, b))
    serial = analyze(sources, config=AnalysisConfig(backend="numpy", workers=1))
    parallel = analyze(
        sources,
        config=AnalysisConfig(backend="numpy", workers=2, managed_memory_bytes=256 * 1024**2),
    )
    for tr in (a, b):
        sa = serial.by_trace[tr.trace_id]
        pr = parallel.by_trace[tr.trace_id]
        assert pr.status == "complete", pr.manifest.get("failure", pr.manifest)
        left = sa.metrics.to_pandas().drop(columns=["result_id", "assignment_revision"])
        right = pr.metrics.to_pandas().drop(columns=["result_id", "assignment_revision"])
        import pandas as pd

        pd.testing.assert_frame_equal(left, right)
        assert sa.samples.to_pandas().cycle_id.tolist() == pr.samples.to_pandas().cycle_id.tolist()
    with b.path.open("ab") as f:
        f.write(b"x")
    r = analyze(sources, config=AnalysisConfig(backend="numpy"))
    assert r.by_trace[a.trace_id].status == "complete" and r.by_trace[b.trace_id].status == "failed"


def test_disjoint_plot_no_bridge(make_trace):
    """Verify disjoint plot no bridge using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    tr = make_trace(np.arange(8), np.ones(8))
    updates = list(
        iter_plot_updates(tr.select(sample_id=[0, 1, 6, 7]), spec=PlotSpec(plot_mode="full"))
    )
    data = [u for u in updates if u.operation == "replace"][-1].data
    assert data["x"] == [0, 1, None, 6, 7]
