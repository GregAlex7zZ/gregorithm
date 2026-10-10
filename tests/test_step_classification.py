"""Synthetic optional-step regressions; signal labels do not validate controller intent."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_half_cycles import prepared

from gregorithm import MemoryBudget, ReadOptions, StepConfig


@pytest.mark.parametrize("batch", [1, 2, 5, 100])
def test_rest_cc_cv_variable_and_short_final_support(tmp_path, batch):
    """Rest is explicit; stable active I is CC even at stable V; changing I uses V range."""
    source = prepared(
        tmp_path,
        range(18),
        [0, 0] + [1] * 4 + [1, 2, 3, 4] + [4, 3, 2, 1] + [0, 0] + [-1, -1],
        [4] * 10 + [4, 4.01, 4.02, 4.03] + [4] * 4,
        batch_rows=batch,
    )
    result = source.steps()
    try:
        frame = result.table()
        assert frame.step_type.tolist() == ["rest", "CC", "CV", "variable", "rest", "CC"]
        assert frame.start_s.tolist() == [0, 2, 6, 10, 14, 16]
        assert frame.end_s.tolist() == [1, 5, 9, 13, 15, 17]
        assert frame.mean_current_mA.tolist() == [0, 1, 2.5, 2.5, 0, -1]
        expanded = result.measurements().table()
        original = source.preview(100)
        pd.testing.assert_frame_equal(expanded[original.columns], original)
        assert "step_type" not in original and "step" not in expanded
        assert (
            expanded.step_type.tolist()
            == ["rest"] * 2 + ["CC"] * 4 + ["CV"] * 4 + ["variable"] * 4 + ["rest"] * 2 + ["CC"] * 2
        )
        assert result.types("CV").table().step.tolist() == [3]
        assert result.types([]).table().empty
        assert result.types("CV").measurements().table().time_s.tolist() == [6, 7, 8, 9]
    finally:
        result.close()


def test_windows_do_not_create_steps_and_level_reference_is_frozen(tmp_path):
    """Merge compatible windows and split stable new levels without creating a half-cycle."""
    source = prepared(tmp_path, range(16), [1] * 8 + [2] * 8)
    result = source.steps()
    try:
        frame = result.details().table()
        assert frame.step_type.tolist() == ["CC", "CC"]
        assert frame.start_s.tolist() == [0, 8]
        assert frame.reference_current_mA.tolist() == [1, 2]
        assert frame.assessment_windows.tolist() == [2, 2]
        assert frame.mean_current_mA.tolist() == [1, 2]
        assert frame.preceding_sample_s.iloc[1] == 7
        assert frame.following_sample_s.iloc[0] == 8
    finally:
        result.close()
    capacity = source.capacity()
    try:
        assert len(capacity.table()) == 1
    finally:
        capacity.close()


def test_sparse_extension_and_unknown_short_tail(tmp_path):
    """Fully measured sparse support extends beyond 3/10 s; a lone final sample is unknown."""
    result = prepared(tmp_path, [0, 100, 200], [1, 1, 1], batch_rows=1).steps()
    try:
        frame = result.details().table()
        assert frame.step_type.tolist() == ["CC", "unknown"]
        assert frame.window_span_min_s.iloc[0] == 100
        assert frame.window_span_max_s.iloc[0] == 100
        assert frame.end_s.tolist() == [100, 200]
        assert np.isnan(frame.mean_current_mA.iloc[1])
    finally:
        result.close()


def test_sparse_extension_counts_real_points_and_configurable_minimum(tmp_path):
    """An eligible missing point cannot end sparse extension or count as confirmation."""
    source = prepared(tmp_path, [0, 4, 5], [1, np.nan, 1])
    result = source.steps()
    try:
        frame = result.details().table()
        assert frame.step_type.tolist() == ["CC"]
        assert frame.n_real_current.tolist() == [2]
        assert frame.window_span_max_s.tolist() == [5]
    finally:
        result.close()
    root = tmp_path / "three"
    root.mkdir()
    result = prepared(root, [0, 100, 200], [1, 1, 1]).steps(config=StepConfig(min_real_samples=3))
    try:
        assert result.details().table().window_span_max_s.tolist() == [200]
    finally:
        result.close()


@pytest.mark.parametrize("batch", [1, 3, 100])
def test_isolated_estimate_and_unresolved_current_gap(tmp_path, batch):
    """Real support determines CC; eligible estimates preserve a window; unresolved gaps split it."""
    source = prepared(
        tmp_path, range(10), [1, 1, np.nan, 1, 1, np.nan, np.nan, 1, 1, 1], batch_rows=batch
    )
    result = source.steps()
    try:
        frame = result.details().table()
        assert frame.step_type.tolist() == ["CC", "unknown", "unknown", "CC"]
        assert frame.start_s.tolist() == [0, 4, 5, 7]
        assert frame.current_quality.iloc[0] == "estimated"
        assert frame.n_estimated_current.iloc[0] == 1
        expanded = result.measurements().table()
        assert expanded.current_mA.isna().sum() == 3
        assert expanded.step_type.iloc[2] == "CC"
    finally:
        result.close()


def test_missing_voltage_does_not_prevent_current_only_cc(tmp_path):
    """Stable current needs no voltage; changing current needs V support to establish CV."""
    result = prepared(tmp_path, range(8), [1] * 4 + [1, 2, 3, 4], [np.nan] * 8).steps()
    try:
        assert result.table().step_type.tolist() == ["CC", "unknown"]
    finally:
        result.close()


def test_manual_protocol_priority_and_rest_conflicts(tmp_path):
    """Known CV overrides both-flat inference; manual CC wins over explicitly mapped CV."""
    from gregorithm import AnnotationConflictWarning, prepare_source

    path = tmp_path / "raw.parquet"
    pq.write_table(
        pa.table(
            {
                "time_s": np.arange(10, dtype=float),
                "current_mA": [0, 0] + [1] * 8,
                "voltage_V": [4.0] * 10,
                "mode": [7] * 10,
            }
        ),
        path,
    )
    source = prepare_source(path, tmp_path / "prepared.parquet")
    with pytest.warns(AnnotationConflictWarning):
        result = source.steps(
            protocol_column="mode", protocol_mapping={7: "CV"}, manual_windows=[(6, 9, "CC")]
        )
    try:
        frame = result.details().table()
        assert frame.step_type.tolist() == ["rest", "CV", "CC"]
        assert frame.classification_origin.tolist() == ["inferred", "protocol", "manual"]
        assert frame.inferred_type.tolist() == ["rest", "CC", "CC"]
        assert result.report["annotation_contradictions"] == 2
        assert result.report["protocol_mapping"] == [{"value": 7, "step_type": "CV"}]
        assert result.measurements().table()["mode"].tolist() == [7] * 10
    finally:
        result.close()
    with pytest.raises(ValueError, match="Contradictory"):
        source.steps(manual_windows=[(2, 6, "CC"), (6, 9, "CV")])
    with pytest.raises(ValueError, match="retained"):
        source.steps(manual_windows=[(2.5, 5, "CV")])


def test_opt_in_and_resource_lifecycle(tmp_path, monkeypatch):
    """Ordinary capacity/energy/current never invoke classification or append step labels."""
    import gregorithm.step_classification as module
    from gregorithm import CancellationToken, Cancelled, MaterializationLimitError

    source = prepared(tmp_path, range(8), [1] * 4 + [-1] * 4)

    def forbidden(*args, **kwargs):
        """Fail if an ordinary calculation invokes the optional classifier."""
        raise AssertionError("Classification unexpectedly invoked")

    with monkeypatch.context() as patch:
        patch.setattr(module, "classify_steps", forbidden)
        source.totals()
        capacity = source.capacity(include_energy=True)
        capacity.close()
        source.c_rate(reference_mAh=1).mean()
    budget = MemoryBudget(8 * 1024**2)
    result = source.steps(memory_budget=budget)
    assert budget.used == 0
    with pytest.raises(MaterializationLimitError):
        result.table(max_rows=0)
    assert budget.used == 0
    iterator = result.measurements().iter_batches()
    next(iterator)
    with pytest.raises(ValueError, match="active"):
        result.close()
    iterator.close()
    assert budget.used == 0
    result.close()
    with pytest.raises(ValueError, match="closed"):
        result.table()
    token = CancellationToken()
    token.cancel()
    with pytest.raises(Cancelled):
        source.steps(memory_budget=budget, cancellation=token)
    assert budget.used == 0
    assert not list((tmp_path / ".gregorithm-work").iterdir())


def test_batch_equivalence_variable_cadence(tmp_path):
    """Compare full summaries under different batches with variable acquisition cadence."""
    source = prepared(
        tmp_path, [0, 0.1, 0.3, 3, 3.2, 3.3, 5, 6.2, 10, 20], [1, 1, 1, 1, 1.2, 1.2, 1.2, 1.2, 0, 0]
    )
    first = None
    for batch in [1, 2, 4, 100]:
        result = replace(source, read_options=ReadOptions(batch_rows=batch)).steps()
        try:
            frame = result.details().table()
            if first is not None:
                pd.testing.assert_frame_equal(
                    frame, first, check_exact=False, rtol=1e-13, atol=1e-13
                )
            first = frame
        finally:
            result.close()
    assert first.step_type.tolist() == ["CC", "CC", "rest"]


def test_invalid_operational_settings():
    """Reject invalid thresholds rather than silently changing scientific settings."""
    for settings in (
        {"window_s": 0},
        {"min_real_samples": True},
        {"voltage_range_V": -1},
        {"current_relative": float("nan")},
    ):
        with pytest.raises(ValueError):
            StepConfig(**settings)


def test_frozen_level_drift_and_inclusive_thresholds(tmp_path):
    """Gradual drift is compared with the first plateau; exact 5%/2mV equalities are stable."""
    root = tmp_path / "drift"
    root.mkdir()
    result = prepared(root, range(12), [1] * 4 + [1.04] * 4 + [1.08] * 4).steps()
    try:
        frame = result.details().table()
        assert frame.start_s.tolist() == [0, 8]
        assert frame.reference_current_mA.tolist() == [1, 1.08]
    finally:
        result.close()
    root = tmp_path / "threshold"
    root.mkdir()
    result = prepared(root, range(8), [19] * 4 + [20] * 4).steps()
    try:
        assert result.table().step_type.tolist() == ["CC"]
    finally:
        result.close()
    root = tmp_path / "voltage"
    root.mkdir()
    result = prepared(root, range(4), [1, 2, 3, 4], [0, 0.001, 0.0015, 0.002]).steps()
    try:
        assert result.table().step_type.tolist() == ["CV"]
    finally:
        result.close()


def test_transition_is_retained_as_distinct_variable_domain(tmp_path):
    """A recorded unstable transition is neither absorbed into CC nor discarded from Q/E."""
    source = prepared(
        tmp_path,
        range(12),
        [1] * 4 + [1, 1.5, 2, 3] + [3] * 4,
        [4] * 4 + [4, 4.01, 4.02, 4.03] + [4] * 4,
    )
    before = source.totals().values
    result = source.steps()
    try:
        assert result.table().step_type.tolist() == ["CC", "variable", "CC"]
        after = source.totals().values
        assert before == after
    finally:
        result.close()


def test_missing_current_cannot_be_annotated_into_measured_rest(tmp_path):
    """Known mode labels never turn absent I into zero or measured rest."""
    from gregorithm import AnnotationConflictWarning

    source = prepared(tmp_path, [0, 1, 2], [np.nan, np.nan, np.nan])
    with pytest.warns(AnnotationConflictWarning):
        result = source.steps(manual_windows=[(0, 2, "rest")])
    try:
        assert result.table().step_type.tolist() == ["unknown"]
        assert result.measurements().table().current_mA.isna().all()
    finally:
        result.close()


def test_real_plateau_reference_does_not_use_estimated_window_endpoint(tmp_path):
    """A last missing value interpolated toward a later level cannot move the old CC reference."""
    source = prepared(tmp_path, range(8), [1, 1, 1, np.nan, 2, 2, 2, 2], batch_rows=1)
    result = source.steps()
    try:
        frame = result.details().table()
        assert frame.step_type.tolist() == ["CC", "CC"]
        assert frame.reference_current_mA.tolist() == [1, 2]
        assert frame.start_s.tolist() == [0, 4]
        assert frame.n_estimated_current.tolist() == [1, 0]
    finally:
        result.close()


def test_measurement_projection_export_and_explicit_step_numbers(tmp_path):
    """Only explicit numeric-step requests create that column; all original payload survives export."""
    source = prepared(tmp_path, range(8), [1] * 4 + [0] * 4)
    result = source.steps()
    try:
        view = result.measurements(include_step_number=True)
        frame = view.table()
        assert frame.step.tolist() == [1] * 4 + [2] * 4
        assert frame.step_type.tolist() == ["CC"] * 4 + ["rest"] * 4
        path = tmp_path / "labels.parquet"
        view.write_parquet(path)
        original = pq.read_table(source.path)
        exported = pq.read_table(path)
        assert original.equals(exported.select(original.schema.names))
        blocks = list(view.iter_batches(columns=["time_s", "note"]))
        assert blocks[0].schema.names == ["time_s", "note", "step_type", "step"]
        with pytest.raises(FileExistsError):
            view.write_parquet(path)
    finally:
        result.close()


def test_projection_cancellation_stale_source_and_unmapped_protocol(tmp_path):
    """Early failures release budget; missing mapped values never become guessed modes."""
    from gregorithm import (
        CancellationToken,
        Cancelled,
        SourceChangedError,
        UnmappedProtocolWarning,
        prepare_source,
    )

    path = tmp_path / "raw.parquet"
    pq.write_table(
        pa.table(
            {
                "time_s": [0.0, 1.0, 2.0, 3.0],
                "current_mA": [1.0] * 4,
                "voltage_V": [4.0] * 4,
                "mode": [2] * 4,
            }
        ),
        path,
    )
    source = prepare_source(path, tmp_path / "prepared.parquet")
    budget = MemoryBudget(8 * 1024**2)
    with pytest.warns(UnmappedProtocolWarning):
        result = source.steps(
            protocol_column="mode", protocol_mapping={1: "CV"}, memory_budget=budget
        )
    try:
        assert result.table().step_type.tolist() == ["CC"]
        with pytest.raises(ValueError, match="Projection"):
            list(result.measurements().iter_batches(columns=["missing"]))
        assert budget.used == 0
        token = CancellationToken()
        iterator = result.measurements().iter_batches(cancellation=token)
        next(iterator)
        token.cancel()
        with pytest.raises(Cancelled):
            next(iterator)
        assert budget.used == 0
        pq.write_table(pa.table({"time_s": [0.0], "current_mA": [2.0]}), source.path)
        with pytest.raises(SourceChangedError):
            result.measurements().preview()
        assert budget.used == 0
    finally:
        result.close()


def test_explicit_cc_anchors_first_independently_stable_reference(tmp_path):
    """Known CC may precede a stable plateau; real support freezes the later reference."""
    source = prepared(tmp_path, range(12), [1, 2, 3, 4] + [2] * 4 + [4] * 4)
    result = source.steps(manual_windows=[(0, 11, "CC")])
    try:
        frame = result.details().table()
        assert frame.step_type.tolist() == ["CC", "CC"]
        assert frame.reference_current_mA.tolist() == [2, 4]
        assert frame.reference_start_s.tolist() == [4, 8]
        assert frame.reference_end_s.tolist() == [7, 11]
        assert frame.reference_real_samples.tolist() == [4, 4]
        assert frame.inferred_type.iloc[0] == "mixed"
    finally:
        result.close()


def test_within_step_mean_c_rate_excludes_front_and_uses_fixed_reference(tmp_path):
    """Scale actual within-step means, not sample means or switching-front averages."""
    source = prepared(tmp_path, [0, 1, 2, 3, 4, 5, 6, 7], [1] * 4 + [4] * 4)
    budget = MemoryBudget(8 * 1024**2)
    steps = source.steps(memory_budget=budget)
    try:
        rates = steps.types("CC").c_rate(reference_mAh=2)
        frame = rates.table()
        assert frame.mean_current_mA.tolist() == [1, 4]
        assert frame.mean_c_rate_h_inv.tolist() == [0.5, 2]
        assert frame.covered_s.tolist() == [3, 3]
        assert frame.c_rate_quality.tolist() == ["complete", "complete"]
        assert budget.used == 0
        assert rates.types([]).table().empty
        iterator = rates.iter_batches(batch_rows=1)
        next(iterator)
        with pytest.raises(ValueError, match="active"):
            steps.close()
        iterator.close()
        assert budget.used == 0
    finally:
        steps.close()
