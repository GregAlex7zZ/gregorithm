"""Synthetic regression tests; run with tools/run_tests.py. Passing verifies software behavior, not experimental electrochemistry."""

import numpy as np
import pyarrow.parquet as pq
import pytest

from gregorithm import *
from gregorithm.analysis import reopen_result


def test_pipeline_normalization_lineage(make_trace, tmp_path):
    """Verify pipeline normalization lineage using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    trace = make_trace(
        np.arange(50), np.ones(50), np.linspace(3, 4, 50), extra=["a"] * 50, nullable=[None] * 50
    )
    r = analyze(
        trace,
        analyses=["capacity", "energy", "dqdv", "c_rate"],
        config=AnalysisConfig(
            batch_rows=7, backend="numpy", mass_g=0.5, area_cm2=2, reference_capacity_mAh=0.5
        ),
        export_samples=True,
    )
    df = r.samples.to_pandas()
    assert len(df) == 50 and df.extra.tolist() == ["a"] * 50
    np.testing.assert_array_equal(df.capacity_mAh_g, df.capacity_mAh / 0.5)
    np.testing.assert_array_equal(df.capacity_mAh_cm2, df.capacity_mAh / 2)
    assert df.c_rate_h_inv.iloc[0] == 2
    assert r.curves.to_pandas().half_cycle_id.nunique() == 1
    assert r.manifest["config_hash"] and r.manifest["source_revision"]
    p = tmp_path / "normalized.parquet"
    r.samples.write_parquet(p)
    assert p.with_suffix(".parquet.manifest.json").exists()
    assert pq.read_table(p).column("extra").to_pylist() == ["a"] * 50
    reopened = reopen_result(r.directory)
    assert reopened.manifest["result_id"] == r.result_id


def test_limits_and_selection(make_trace, tmp_path):
    """Verify limits and selection using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    trace = make_trace(np.arange(30), np.repeat([-1, 1, -1], 10))
    sel = trace.select(cycle_id=2, columns=["voltage_V"])
    d = sel.to_pandas()
    assert d.sample_id.tolist() == list(range(10, 30))
    with pytest.raises(MaterializationLimitError):
        sel.to_pandas(max_rows=19)
    with pytest.raises(MaterializationLimitError):
        sel.to_pandas(max_bytes=2)
    assert len(sel.head(3).to_pandas()) == 3
    assert "trace_id" in d
    with pytest.raises(ValueError):
        sel.write_parquet(trace.path)


def test_time_clip_no_bridge(make_trace):
    """Verify time clip no bridge using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    trace = make_trace([0, 10, 20], [1, 3, 1])
    cfg = AnalysisConfig(backend="numpy")
    r = analyze(trace.select(time_range=(5, 15)), config=cfg)
    m = r.metrics.to_pandas()
    q = m[
        (m.scope == "global")
        & (m.view == "measured")
        & (m.direction == "charge")
        & (m.metric == "capacity")
    ].value.iloc[0]
    assert q == pytest.approx(25 / 3600)
    assert r.samples.to_pandas().sample_id.tolist() == [1]
    assert r.samples.to_pandas().global_charge_mAh.iloc[0] == pytest.approx(20 / 3600)
    d = trace.select(sample_id=[0, 2]).to_pandas()
    assert d.sample_id.tolist() == [0, 2]
    r = analyze(trace.select(sample_id=[0, 2]), config=cfg)
    assert r.metrics.to_pandas().empty


def test_reference_cycle2_missing_and_manual(make_trace):
    """Verify reference cycle2 missing and manual using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    trace = make_trace(np.arange(9), np.repeat([-1, 1, -1], 3))
    cfg = AnalysisConfig(backend="numpy")
    r = analyze(trace, analyses=["c_rate"], config=cfg)
    assert r.manifest["c_rate_reference"]["group_id"] == 2
    assert r.manifest["c_rate_reference"]["direction"] == "discharge"
    trace2 = make_trace([0, 1, 2], [0.005, 0.005, 0.005])
    r = analyze(
        trace2,
        analyses=["c_rate"],
        config=AnalysisConfig(backend="numpy", reference_capacity_mAh=0.5),
    )
    np.testing.assert_allclose(r.samples.to_pandas().c_rate_h_inv, 0.01)
    r = analyze(trace2, analyses=["c_rate"], config=cfg)
    assert r.manifest["c_rate_reference"]["value_mAh"] is None


def test_manifest_relocation(make_trace, tmp_path):
    """Verify manifest relocation using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    import shutil

    trace = make_trace([0, 1, 2], [1, 1, 1])
    manifest = tmp_path / "trace.json"
    trace.save_manifest(manifest, strong=True)
    dest = tmp_path / "renamed.parquet"
    shutil.copy2(trace.path, dest)
    assert open_trace(dest, manifest=manifest).trace_id == trace.trace_id


def test_multiple_traces(make_trace):
    """Verify multiple traces using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    a = make_trace([0, 1, 2], [1, 1, 1])
    b = make_trace([0, 1, 2], [1, 1, 1])
    batch = analyze(TraceCollection((a, b)), config=AnalysisConfig(backend="numpy"))
    assert list(batch.by_trace) == [a.trace_id, b.trace_id]
    assert set(batch.table().to_pandas().trace_id) == {a.trace_id, b.trace_id}


def test_cancellation_and_source_mutation(make_trace):
    """Verify cancellation and source mutation using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    trace = make_trace([0, 1, 2], [1, 1, 1])
    token = CancellationToken()
    token.cancel()
    with pytest.raises(Cancelled):
        analyze(trace, cancellation=token)
    with trace.path.open("ab") as f:
        f.write(b"x")
    with pytest.raises(SourceChangedError):
        trace.select().to_pandas()
