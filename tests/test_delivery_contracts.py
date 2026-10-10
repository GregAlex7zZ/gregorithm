"""Scientific and notebook contracts added during delivery consolidation."""

import time

import numpy as np
import pyarrow as pa
import pytest

from gregorithm import (
    AmbiguousTimeSelectionError,
    AnalysisConfig,
    DQDVConfig,
    EventSelection,
    PlotSpec,
    analyze,
    compute_dqdv,
    from_pandas,
    integrate,
    normalize,
    open_traces,
)
from gregorithm.compare import compare_peaks, peak
from gregorithm.io import TableView
from gregorithm.plotting import plot_progressive


def test_normalization_preserves_nulls_types_and_extras():
    """Verify normalization preserves nulls types and extras using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    block = pa.RecordBatch.from_pydict(
        {
            "capacity_mAh": pa.array([1.0, None, float("nan")]),
            "note": pa.array(["a", None, "c"]),
            "count": pa.array([1, 2, 3], type=pa.int16()),
        }
    )
    view = normalize(TableView(lambda: iter([block])), mass_g=2.0, area_cm2=4.0)
    actual = next(view.iter_batches())
    assert actual.column("capacity_mAh_g").to_pylist()[:2] == [0.5, None]
    assert np.isnan(actual.column("capacity_mAh_g")[2].as_py())
    assert actual.column("count").equals(block.column("count"))
    assert actual.column("note").equals(block.column("note"))


def test_union_windows_integrate_overlap_once(make_trace):
    """Verify union windows integrate overlap once using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    trace = make_trace([0, 10, 20], [1, 1, 1])
    selection = trace.select(time_range=(2, 12)).union(trace.select(time_range=(8, 18)))
    result = analyze(selection, config=AnalysisConfig(backend="numpy", batch_rows=1))
    metrics = result.metrics.to_pandas()
    row = metrics.query(
        "scope == 'global' and view == 'measured' and direction == 'charge' and metric == 'capacity'"
    ).iloc[0]
    assert row.value == pytest.approx(16 / 3600)
    assert result.samples.to_pandas().sample_id.tolist() == [1]


def test_missing_fully_excluded_interval_does_not_invalidate_cycling(make_trace):
    """Verify missing fully excluded interval does not invalidate cycling using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    current = np.ones(10)
    current[4] = np.nan
    trace = make_trace(np.arange(10), current)
    event = {
        "trace_id": trace.trace_id,
        "source_revision": trace.source_revision,
        "event_id": "manual-known-diagnostic",
        "time_section_id": 0,
        "sample_t0": 2,
        "sample_t1": 3,
        "sample_t2": 6,
        "sample_t3": 7,
        "t0": 2.0,
        "t1": 3.0,
        "t2": 6.0,
        "t3": 7.0,
        "decision": "accepted",
        "excluded": True,
    }
    result = analyze(trace, events=EventSelection((event,)), config=AnalysisConfig(backend="numpy"))
    metrics = result.metrics.to_pandas()
    rows = metrics.query(
        "scope == 'global' and direction == 'charge' and metric == 'capacity'"
    ).set_index("view")
    assert rows.loc["cycling", "invalid_intervals"] == 0
    assert rows.loc["measured", "invalid_intervals"] > 0
    assert rows.loc["cycling", "value"] == pytest.approx(4 / 3600)


def test_invalid_outside_selection_keeps_selected_coverage(make_trace):
    """Verify invalid outside selection keeps selected coverage using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    current = np.ones(10)
    current[8] = np.nan
    result = analyze(
        make_trace(np.arange(10), current).select(time_range=(1, 3)),
        config=AnalysisConfig(backend="numpy"),
    )
    rows = result.metrics.to_pandas().query(
        "scope == 'global' and direction == 'charge' and metric == 'capacity'"
    )
    assert (rows.invalid_intervals == 0).all()
    assert result.status == "partial"


def test_step_accumulations_match_across_paths_and_batches(make_trace):
    """Verify step accumulations match across paths and batches using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    trace = make_trace(np.arange(30), np.r_[np.ones(10), np.zeros(5), -np.ones(15)])
    small = integrate(trace, config=AnalysisConfig(backend="numpy", batch_rows=1)).to_pandas()
    result = analyze(
        trace.select(time_range=(0, 29)), config=AnalysisConfig(backend="numpy", batch_rows=7)
    )
    selected = result.samples.to_pandas()
    for name in ("step_charge_mAh", "step_discharge_mAh", "step_charge_Wh", "step_discharge_Wh"):
        np.testing.assert_allclose(small[name], selected[name], atol=1e-12)


def branch_view(v, q, excluded=None):
    """Construct a synthetic one-branch Arrow view with point IDs, polarity, and optional diagnostic exclusions."""
    block = pa.RecordBatch.from_pydict(
        {
            "voltage_V": v,
            "capacity_mAh": q,
            "current_mA": np.ones(len(v)),
            "sample_id": np.arange(len(v)),
            "half_cycle_id": np.ones(len(v), dtype=np.int64),
            "cycle_id": np.ones(len(v), dtype=np.int64),
            "trace_id": ["synthetic"] * len(v),
            "direction": ["charge"] * len(v),
            "excluded": np.zeros(len(v), dtype=bool) if excluded is None else excluded,
        }
    )
    return TableView(lambda: iter([block]))


def test_common_grid_endpoints_are_not_shifted_to_samples():
    """Verify common grid endpoints are not shifted to samples using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    cfg = DQDVConfig(n_points=31, trim_extremes_V=0, e_lo=3.155, e_hi=3.865)
    for v in (np.linspace(3, 4, 101), np.linspace(3.001, 4.001, 76)):
        data = compute_dqdv(branch_view(v, 2 * v + 1), config=cfg).to_pandas()
        assert data.voltage_V.iloc[0] == pytest.approx(cfg.e_lo)
        assert data.voltage_V.iloc[-1] == pytest.approx(cfg.e_hi)
        np.testing.assert_allclose(data.dqdv_mAh_V, 2, atol=1e-12)


def test_dqdv_omits_diagnostic_support():
    """Verify dqdv omits diagnostic support using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    v = np.linspace(3, 4, 101)
    mask = np.zeros(len(v), dtype=bool)
    mask[40:60] = True
    q = 2 * v + 1
    q[mask] += 100
    curve = compute_dqdv(branch_view(v, q, mask), config=DQDVConfig(n_points=31, trim_extremes_V=0))
    data = curve.to_pandas()
    np.testing.assert_allclose(data.dqdv_mAh_V, 2, atol=1e-12)
    assert not set(data.support_left_sample_id).intersection(range(40, 60))


def test_peak_known_shape_shift_and_ambiguity():
    """Verify peak known shape shift and ambiguity using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    from scipy.special import erf

    v = np.linspace(3, 4, 201)
    curves = []
    for center in (3.4, 3.5):
        q = 0.05 * np.sqrt(np.pi) / 2 * erf((v - center) / 0.05)
        curves.append(
            compute_dqdv(branch_view(v, q), config=DQDVConfig(n_points=201, trim_extremes_V=0))
        )
    comparison = compare_peaks(*curves, (3.2, 3.7))
    assert comparison["deltaV_V"] == pytest.approx(0.1, abs=0.006)
    flat = compute_dqdv(
        branch_view(v, np.zeros(len(v))), config=DQDVConfig(n_points=201, trim_extremes_V=0)
    )
    outcome = peak(flat, (3, 4), max_candidates=3)
    assert outcome["ambiguous"] and outcome["candidates_truncated"]
    assert outcome["candidate_count"] == 201 and len(outcome["candidates"]) == 3


def test_result_time_window_requires_section_after_reset(make_trace):
    """Verify result time window requires section after reset using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    result = analyze(
        make_trace(np.r_[np.arange(5), np.arange(5)], np.ones(10)),
        config=AnalysisConfig(backend="numpy"),
    )
    with pytest.raises(AmbiguousTimeSelectionError):
        result.samples.select(time_s={"min": 1, "max": 3}).to_pandas()
    data = result.samples.select(time_s={"min": 1, "max": 3}, time_section_id=1).to_pandas()
    assert len(data) == 3


def test_opening_failure_isolated_and_pandas_adapter(tmp_path):
    """Verify opening failure isolated and pandas adapter using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    import pandas as pd

    frame = pd.DataFrame(
        {
            "time_s": [0, 1, 2],
            "current_mA": [1, 1, 1],
            "voltage_V": [4, 4, 4],
            "note": ["a", None, "b"],
        }
    )
    trace = from_pandas(frame, directory=tmp_path)
    assert trace.select().to_pandas().note.tolist() == frame.note.tolist()
    with pytest.raises(Exception, match="limits"):
        from_pandas(frame, directory=tmp_path, max_rows=2)
    result = analyze(
        open_traces([trace.path, tmp_path / "missing.parquet"]),
        config=AnalysisConfig(backend="numpy"),
    )
    assert sorted(r.status for r in result.by_trace.values()) == ["complete", "failed"]


def test_progressive_terminal_survives_full_queue():
    """Verify progressive terminal survives full queue using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    block = pa.RecordBatch.from_pydict(
        {"time_s": np.arange(20), "voltage_V": np.arange(20), "sample_id": np.arange(20)}
    )
    session = plot_progressive(TableView(lambda: iter([block])), spec=PlotSpec(queue_size=1))
    session.start()
    time.sleep(0.3)
    session.cancel()
    session.thread.join(timeout=3)
    assert not session.thread.is_alive()
    for _ in range(5):
        session.poll()
    assert session.status == "cancelled"
    assert session.last_update.status == "cancelled"
    session.close()


def test_presets_and_curve_series_identity():
    """Verify presets and curve series identity using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    assert PlotSpec.preset("dqdv_voltage").y == "dqdv_mAh_V"
    assert (
        PlotSpec.preset("voltage_capacity", max_points_per_figure=100).max_points_per_figure == 100
    )


@pytest.mark.parametrize("order", ["charge_then_discharge", "discharge_then_charge"])
def test_efficiency_never_inverts_and_complete_cycle_selection_is_valid(make_trace, order):
    """Verify efficiency never inverts and complete cycle selection is valid using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    signs = [1, -1, 1, -1] if order == "charge_then_discharge" else [-1, 1, -1, 1]
    current = np.repeat(signs, 10).astype(float)
    current[current < 0] *= 2
    trace = make_trace(np.arange(40), current)
    cfg = AnalysisConfig(backend="numpy", cycle_order=order)
    full = analyze(trace, config=cfg).metrics.to_pandas()
    selected = analyze(trace.select(cycle_id=2).with_config(cfg)).metrics.to_pandas()
    row = selected.query(
        "metric == 'coulombic_efficiency' and view == 'cycling' and cycle_id == 2"
    ).iloc[0]
    quantities = full.query(
        "metric == 'capacity' and scope == 'cycle' and view == 'cycling' and cycle_id == 2"
    ).set_index("direction")
    assert row.value == pytest.approx(
        quantities.loc["discharge", "value"] / quantities.loc["charge", "value"]
    )
    assert row.quality == "observed_complete"


def test_missing_interval_between_samples_invalidates_selected_window(make_trace):
    """Verify missing interval between samples invalidates selected window using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    trace = make_trace([0, 10, 20], [1, np.nan, 1])
    result = analyze(trace.select(time_range=(2, 8)), config=AnalysisConfig(backend="numpy"))
    rows = result.metrics.to_pandas().query(
        "scope == 'global' and view == 'cycling' and metric == 'capacity'"
    )
    assert len(rows) and (rows.invalid_intervals > 0).all()


def test_grid_profile_rejects_ignored_point_count():
    """Verify grid profile rejects ignored point count using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    with pytest.raises(ValueError, match="Point count"):
        DQDVConfig(profile="voltage_grid", n_points=100)


def test_dcir_rejects_loaded_baseline(make_trace):
    """Verify dcir rejects loaded baseline using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    from gregorithm import manual_event, measure_dcir

    trace = make_trace([0, 1, 2, 3, 4, 5], [1, 1, 100, 100, 100, 0])
    event = manual_event(trace, sample_t0=1, sample_t1=2, sample_t2=4, sample_t3=5)
    row = measure_dcir(trace, event, delay_s=2).to_pandas().iloc[0]
    assert not row.valid and row.reason == "baseline_not_rest"


def test_event_export_keeps_late_correction_fields_across_batches():
    """Verify event export keeps late correction fields across batches using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    records = tuple(
        {
            "event_id": str(j),
            "trace_id": "synthetic",
            "t0": None if j < 128 else 1.0,
            **({"correction_reason": "Protocol boundary review"} if j == 129 else {}),
        }
        for j in range(260)
    )
    blocks = list(EventSelection(records).table().iter_batches())
    assert max(block.num_rows for block in blocks) <= 128
    data = pa.Table.from_batches(blocks).to_pandas()
    assert data.loc[129, "correction_reason"] == "Protocol boundary review"
    assert data.loc[129, "t0"] == 1.0


def test_c_rate_preserves_nullable_current(tmp_path):
    """Verify c rate preserves nullable current using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    import pyarrow.parquet as pq

    from gregorithm import open_trace

    path = tmp_path / "nullable.parquet"
    pq.write_table(
        pa.table(
            {
                "time_s": [0.0, 1.0, 2.0],
                "current_mA": [1.0, None, 1.0],
                "voltage_V": [4.0, 4.0, 4.0],
            }
        ),
        path,
    )
    result = analyze(
        open_trace(path),
        analyses=("capacity", "c_rate"),
        config=AnalysisConfig(backend="numpy", reference_capacity_mAh=2.0),
    )
    values = (
        pa.Table.from_batches(list(result.samples.iter_batches()))
        .column("c_rate_h_inv")
        .to_pylist()
    )
    assert values == [0.5, None, 0.5]


def test_preview_limits_dqdv_support_globally_and_terminal_counts(make_trace):
    """Verify preview limits dqdv support globally and terminal counts using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    trace = make_trace(np.arange(200), np.r_[np.ones(100), -np.ones(100)], np.linspace(3, 4, 200))
    progress = []
    result = analyze(
        trace.select().head(50),
        analyses=("capacity", "dqdv"),
        config=AnalysisConfig(backend="numpy", dqdv=DQDVConfig(trim_extremes_V=0)),
        on_progress=progress.append,
    )
    curves = result.curves.to_pandas()
    assert set(curves.half_cycle_id) == {1}
    assert curves.support_right_sample_id.max() <= 49
    diagnostics = result.diagnostics.to_pandas()
    omitted = diagnostics.query("code == 'dqdv_insufficient_branch'")
    assert (omitted.analysis == "dqdv").all() and omitted.sample_start.isna().all()
    assert set(omitted.half_cycle_id) == {2}
    assert progress[-1].rows_selected == 50 and progress[-1].rows_read == 200
    assert progress[-1].batches_completed > 0


def test_serial_batch_reports_file_counters(make_trace):
    """Verify serial batch reports file counters using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    progress = []
    result = analyze(
        open_traces([make_trace([0, 1], [1, 1]).path, make_trace([0, 1], [1, 1]).path]),
        config=AnalysisConfig(backend="numpy"),
        on_progress=progress.append,
    )
    assert len(result.by_trace) == 2
    assert progress[-1].phase == "completion" and progress[-1].files_completed == 2
    assert progress[-1].files_total == 2 and progress[-1].rows_processed == 4


def test_runtime_provenance_and_metadata_snapshot(make_trace):
    """Verify runtime provenance and metadata snapshot using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    trace = make_trace([0, 1, 2], [1, 1, 1])
    trace.metadata = {"cell_id": "synthetic", "basis": {"mass_type": "active"}}
    result = analyze(trace, config=AnalysisConfig(backend="numpy"))
    trace.metadata["basis"]["mass_type"] = "different"
    assert result.manifest["metadata"]["basis"]["mass_type"] == "active"
    assert result.manifest["library_version"] == "0.1.0"
    assert "numpy" in result.manifest["runtime_versions"]
    assert "integration.py" in result.manifest["source_file_sha256"]
    assert result.manifest["elapsed_s"] >= 0 and result.manifest["completed_utc"]


def test_event_detection_configuration_survives_revisions_and_reopen(make_trace):
    """Verify event detection configuration survives revisions and reopen using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    from gregorithm import EventConfig, detect_events, reopen_result

    trace = make_trace(
        np.arange(250), np.r_[np.ones(100), np.zeros(5), -np.ones(10), np.zeros(5), np.ones(130)]
    )
    cfg = AnalysisConfig(backend="numpy", batch_rows=7)
    events = detect_events(trace, cfg, EventConfig(relative_duration=0.2))
    event = events.select(candidate_status="candidate").events[0]
    accepted = events.decide(
        event["event_id"], "accepted", exclude=True, reason="Synthetic known pulse"
    )
    result = analyze(trace, events=accepted, config=cfg)
    restored = reopen_result(result.directory)
    assert restored.trace._event_selection.metadata["detection_config"]["relative_duration"] == 0.2
    assert restored.trace._index.revision == result.assignment_revision
    again = detect_events(
        trace, AnalysisConfig(backend="numpy", batch_rows=1), EventConfig(relative_duration=0.2)
    )
    assert again.revision == events.revision


def test_lazy_reconstruction_rejects_changed_code_provenance(make_trace):
    """Verify lazy reconstruction rejects changed code provenance using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    import json

    from gregorithm import RevisionMismatchError, reopen_result

    result = analyze(make_trace([0, 1, 2], [1, 1, 1]), config=AnalysisConfig(backend="numpy"))
    path = result.directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf8"))
    manifest["source_file_sha256"]["integration.py"] = "changed-code"
    path.write_text(json.dumps(manifest), encoding="utf8")
    with pytest.raises(RevisionMismatchError, match="original library code"):
        reopen_result(result.directory)


def test_atomic_replacement_retries_only_transient_windows_locks(tmp_path, monkeypatch):
    """Verify bounded retries retain atomic replacement and do not mask permanent errors."""
    import os

    from gregorithm.io import replace_file

    source, destination = tmp_path / "staged.json", tmp_path / "final.json"
    source.write_text("new", encoding="utf8")
    destination.write_text("old", encoding="utf8")
    original = os.replace
    calls = []

    def locked(a, b):
        """Simulate two Windows scanner locks before allowing the real atomic operation."""
        calls.append((a, b))
        if len(calls) <= 2:
            error = PermissionError("Temporary sharing restriction")
            error.winerror = 5
            raise error
        return original(a, b)

    monkeypatch.setattr("gregorithm.io.os.replace", locked)
    if os.name == "nt":
        replace_file(source, destination, timeout_s=1)
        assert len(calls) == 3 and destination.read_text(encoding="utf8") == "new"
    else:
        with pytest.raises(PermissionError):
            replace_file(source, destination)

    def denied(a, b):
        """Simulate a permanent access error with no retryable Windows code."""
        raise PermissionError("Permanent denial")

    monkeypatch.setattr("gregorithm.io.os.replace", denied)
    with pytest.raises(PermissionError, match="Permanent"):
        replace_file(source, destination)


def test_sample_manifest_does_not_split_samples_as_curves(make_trace):
    """Verify sample tables remain plot sources when their result also has derivatives."""
    from gregorithm.plotting import plot

    trace = make_trace(
        np.arange(40),
        np.r_[np.ones(20), -np.ones(20)],
        np.r_[np.linspace(3, 4, 20), np.linspace(4, 3, 20)],
    )
    result = analyze(
        trace,
        analyses=("capacity", "dqdv"),
        config=AnalysisConfig(backend="numpy", dqdv=DQDVConfig(trim_extremes_V=0, n_points=10)),
    )
    assert len(result.manifest["curve_manifests"]) == 2
    plotted = plot(result.samples.head(10))
    assert plotted.status == "complete" and len(plotted.figure.data) == 1
    assert len(plotted.figure.data[0].x) == 10
    assert plotted.figure.data[0].customdata[0][1] == trace.trace_id


def test_progressive_curve_zoom_keeps_series_and_global_budget(make_trace):
    """Verify multiple curve views are not recursively expanded on viewport queries."""
    trace = make_trace(
        np.arange(40),
        np.r_[np.ones(20), -np.ones(20)],
        np.r_[np.linspace(3, 4, 20), np.linspace(4, 3, 20)],
    )
    result = analyze(
        trace,
        analyses=("capacity", "dqdv"),
        config=AnalysisConfig(backend="numpy", dqdv=DQDVConfig(trim_extremes_V=0, n_points=20)),
    )
    session = plot_progressive(
        result.curves, spec=PlotSpec.preset("dqdv_voltage", max_points_per_figure=100)
    )

    def consume():
        """Apply pending generation updates on the test consumer thread until terminal."""
        deadline = time.monotonic() + 10
        while session.status == "running" and time.monotonic() < deadline:
            session.poll()
            time.sleep(0.01)
        assert session.status == "complete"

    session.start()
    consume()
    assert len(session.figure.data) == 2
    for lo, hi in ((3.2, 3.8), (3.3, 3.7)):
        generation = session.query_viewport(lo, hi)
        consume()
        assert len(session.figure.data) == 2
        assert sum(len(series.x) for series in session.figure.data) <= 100
        assert session.last_update.generation_id == generation
    session.close()


def test_parquet_projection_reads_only_outputs_and_predicate_dependencies(tmp_path, monkeypatch):
    """Verify explicit projection avoids decoding unrelated wide columns without losing filter context."""
    import pyarrow.parquet as pq

    import gregorithm.io as io

    path = tmp_path / "wide.parquet"
    pq.write_table(
        pa.table(
            {
                "sample_id": [0, 1, 2],
                "cycle_id": [1, 2, 2],
                "capacity_mAh": [0.0, 1.0, 2.0],
                "unrelated": ["a", "b", "c"],
            }
        ),
        path,
    )
    original = io.source_batches
    requested = []

    def observed(source, options=io.ReadOptions(), columns=None, row_groups=None):
        """Record pushed-down columns and delegate to the real bounded reader."""
        requested.append(columns)
        yield from original(source, options, columns, row_groups)

    monkeypatch.setattr(io, "source_batches", observed)
    view = io.parquet_view(path).select(cycle_id=2, columns=["capacity_mAh"])
    data = view.to_pandas()
    assert data.sample_id.tolist() == [1, 2]
    assert set(data) == {"sample_id", "capacity_mAh"}
    assert requested and "cycle_id" in requested[0] and "unrelated" not in requested[0]


def test_aggregate_missing_values_are_strict_json_and_hover_describes_bins():
    """Verify aggregate gaps remain missing and hovered ranges use real source identity kinds."""
    import json

    from gregorithm import iter_plot_updates
    from gregorithm.plotting import plot

    block = pa.RecordBatch.from_pydict(
        {"time_s": np.arange(20), "voltage_V": np.full(20, np.nan), "curve_point_id": np.arange(20)}
    )
    view = TableView(lambda: iter([block]))
    updates = list(iter_plot_updates(view, spec=PlotSpec(plot_mode="aggregate")))
    for update in updates:
        json.dumps(update.to_dict(), allow_nan=False)
    data = next(update.data for update in updates if update.operation == "replace")
    assert all(value is None for value in data["y"])
    assert "first_curve_point_id" in data["aggregate"][0]
    assert "first_sample_id" not in data["aggregate"][0]
    plotted = plot(view, spec=PlotSpec(plot_mode="aggregate"))
    assert "valid_count" in plotted.figure.data[0].customdata[0][9]


def test_empty_known_views_preserve_normalized_schema_and_projection(make_trace):
    """Verify empty notebook selections retain actual columns/types and explicit projections."""
    trace = make_trace([0, 1, 2], [1, 1, 1], note=["a", None, "b"])
    result = analyze(
        trace,
        analyses=("capacity", "c_rate"),
        config=AnalysisConfig(backend="numpy", mass_g=0.5, reference_capacity_mAh=2.0),
    )
    empty = result.samples.select(sample_id=999).to_pandas()
    assert not len(empty)
    assert {"note", "capacity_mAh", "capacity_mAh_g", "c_rate_h_inv"} <= set(empty)
    selected = result.samples.select(sample_id=999, columns=["capacity_mAh_g"]).to_pandas()
    assert set(selected) == {
        "capacity_mAh_g",
        "trace_id",
        "result_id",
        "sample_id",
        "source_revision",
    }


def test_unconfirmed_pulse_and_residual_rest_reconcile_cycle_totals(make_trace):
    """Verify opposite transients are kept by sign/step without misassigning them to initial rest."""
    current = [0, 0, 1, 1, 1, -1, 1, 1, -0.00005, -0.00005, 1, 1]
    trace = make_trace(np.arange(len(current)), current)
    cfg = AnalysisConfig(backend="numpy", batch_rows=1)
    result = analyze(trace, config=cfg)
    samples = result.samples.to_pandas()
    assert set(samples.loc[2:, "half_cycle_id"]) == {1}
    metrics = result.metrics.to_pandas()
    for direction in ("charge", "discharge"):
        row = metrics.query(
            "scope == 'cycle' and group_id == 1 and view == 'cycling' and metric == 'capacity' and direction == @direction"
        ).iloc[0]
        assert row.value == pytest.approx(
            samples["cycle_" + direction + "_mAh"].iloc[-1], abs=1e-12
        )


@pytest.mark.parametrize("batch", [1, 7, 1024])
def test_unassigned_rest_preserves_returning_branch_origins(make_trace, batch):
    """Verify configurable rest association does not reset a resumed branch or lose front portions."""
    trace = make_trace(np.arange(15), np.r_[np.ones(5), np.full(5, -0.00005), np.ones(5)])
    cfg = AnalysisConfig(backend="numpy", batch_rows=batch, rest_assignment="unassigned")
    result = analyze(trace, config=cfg)
    samples = result.samples.to_pandas()
    assert (samples.loc[5:9, "half_cycle_id"] == 0).all()
    assert samples.loc[10, "capacity_mAh"] > samples.loc[4, "capacity_mAh"]
    assert samples.global_charge_mAh.iloc[-1] == pytest.approx(
        integrate(trace, config=AnalysisConfig(backend="numpy", batch_rows=batch))
        .to_pandas()
        .global_charge_mAh.iloc[-1],
        abs=1e-12,
    )
    rows = (
        result.metrics.to_pandas()
        .query("scope == 'cycle' and group_id == 1 and view == 'cycling' and metric == 'capacity'")
        .set_index("direction")
    )
    assert rows.loc["charge", "value"] == pytest.approx(
        samples.cycle_charge_mAh.iloc[-1], abs=1e-12
    )
    assert rows.loc["discharge", "value"] == pytest.approx(
        samples.cycle_discharge_mAh.iloc[-1], abs=1e-12
    )


def test_dqdv_rejects_unassigned_prefix(make_trace):
    """Verify an unresolved action cannot masquerade as a confirmed half-cycle derivative."""
    result = analyze(
        make_trace([0, 0.001, 0.002], [1, 1, 1], [3, 3.5, 4]),
        config=AnalysisConfig(backend="numpy"),
    )
    with pytest.raises(ValueError, match="positive confirmed"):
        compute_dqdv(result.samples, config=DQDVConfig(trim_extremes_V=0))
