"""Regressions for CC-only prepared-source derivative support planning."""

import sqlite3
from contextlib import closing

import numpy as np
import pytest
from test_half_cycles import prepared

from gregorithm import DQDVConfig, compare_prepared_dqdv_peaks, eligible_cc_domains
from gregorithm.prepared_dqdv import DQDVDomain, build_dqdv_curves, spool_dqdv_support


def _half(half, cycle, direction, start, end):
    return {
        "half_cycle": half,
        "cycle": cycle,
        "direction": direction,
        "first_sample_s": start,
        "last_sample_s": end,
    }


def _step(step, kind, direction, start, end):
    return {
        "step": step,
        "step_type": kind,
        "direction": direction,
        "start_s": start,
        "end_s": end,
    }


def test_only_cc_with_matching_half_cycle_direction_is_admitted():
    """Exclude CV, rest, variable, unknown and opposite-direction support."""
    halves = [_half(1, 1, "charge", 1, 10)]
    steps = [
        _step(1, "rest", "rest", 0, 1),
        _step(2, "CC", "charge", 1, 3),
        _step(3, "CV", "charge", 3, 5),
        _step(4, "variable", "charge", 5, 6),
        _step(5, "unknown", "charge", 6, 7),
        _step(6, "CC", "discharge", 7, 8),
        _step(7, "CC", "charge", 8, 12),
    ]
    domains = eligible_cc_domains(halves, steps)
    assert [(d.step, d.start_s, d.end_s) for d in domains] == [(2, 1, 3), (7, 8, 10)]


def test_continuous_cc_steps_remain_separate_and_keep_readable_labels():
    """Never join distinct current-level steps inside the same half-cycle."""
    halves = [_half(4, 2, "discharge", 0, 9)]
    steps = [
        _step(10, "CC", "discharge", 0, 4),
        _step(11, "CC", "discharge", 5, 9),
    ]
    domains = eligible_cc_domains(halves, steps)
    assert [(d.cycle, d.half_cycle, d.step, d.direction) for d in domains] == [
        (2, 4, 10, "discharge"),
        (2, 4, 11, "discharge"),
    ]


def test_cycle_selection_preserves_global_numbers():
    """Filter requested cycles without renumbering their half-cycles or steps."""
    halves = [_half(1, 1, "charge", 0, 2), _half(2, 2, "charge", 3, 5)]
    steps = [_step(1, "CC", "charge", 0, 2), _step(2, "CC", "charge", 3, 5)]
    domains = eligible_cc_domains(halves, steps, cycles=[2])
    assert [(d.cycle, d.half_cycle, d.step) for d in domains] == [(2, 2, 2)]


@pytest.mark.parametrize("cycles", [[0], [True], [1.5], 2])
def test_invalid_cycle_selection_is_rejected(cycles):
    """Reject ambiguous cycle requests before scientific support planning."""
    with pytest.raises(ValueError, match="cycles"):
        eligible_cc_domains([], [], cycles=cycles)


def _domain(end=7):
    return DQDVDomain(1, 1, 1, "charge", 0.0, float(end))


def test_isolated_current_and_voltage_estimates_do_not_split_support(tmp_path):
    """Admit approved isolated estimates only after CC eligibility is established."""
    source = prepared(
        tmp_path,
        range(6),
        [1, 1, np.nan, 1, 1, 1],
        [3.0, 3.1, 3.2, np.nan, 3.4, 3.5],
    )
    support = spool_dqdv_support(source, [_domain(5)], scratch_dir=tmp_path)
    try:
        portions = support.portions()
        assert len(portions) == 1
        assert portions[0]["n_samples"] == 6
        assert portions[0]["estimated_current"] == 1
        assert portions[0]["estimated_voltage"] == 1
        assert portions[0]["partial_context"] == 0
    finally:
        support.close()


def test_remaining_current_gap_splits_curve_and_uses_local_origin(tmp_path):
    """Never bridge a missing run; resume supported differences with partial context."""
    source = prepared(tmp_path, range(8), [1, 1, np.nan, np.nan, 1, 1, 1, 1])
    support = spool_dqdv_support(source, [_domain()], scratch_dir=tmp_path)
    try:
        portions = support.portions()
        assert [row["n_samples"] for row in portions] == [2, 3]
        assert [row["partial_context"] for row in portions] == [0, 1]
        with closing(sqlite3.connect(support.path)) as database:
            origins = database.execute(
                "SELECT capacity_mAh FROM samples WHERE sequence=1 ORDER BY portion"
            ).fetchall()
        assert origins == [(0.0,), (0.0,)]
    finally:
        support.close()


def test_isolated_point_beyond_ten_second_bracket_splits_support(tmp_path):
    """Keep the approved 10 s interpolation ceiling strict and configurable."""
    source = prepared(tmp_path, [0, 1, 12, 13, 14, 15], [1, 1, np.nan, 1, 1, 1])
    support = spool_dqdv_support(source, [_domain(15)], scratch_dir=tmp_path)
    try:
        portions = support.portions()
        assert len(portions) == 2
        assert all(row["estimated_current"] == 0 for row in portions)
        assert portions[1]["partial_context"] == 1
    finally:
        support.close()


def test_bounded_curve_has_readable_labels_and_signed_derivative(tmp_path):
    """Construct a uniform-grid curve without public curve or measurement IDs."""
    source = prepared(tmp_path, range(11), [1] * 11, np.linspace(3, 4, 11))
    support = spool_dqdv_support(source, [_domain(10)], scratch_dir=tmp_path)
    try:
        result = build_dqdv_curves(
            support,
            config=DQDVConfig(n_points=7, trim_extremes_V=0),
            scratch_dir=tmp_path,
        )
        try:
            frame = result.table()
            assert set(
                frame[["cycle", "half_cycle", "step", "portion"]].itertuples(index=False, name=None)
            ) == {(1, 1, 1, 1)}
            assert not {"curve_id", "curve_point_id", "sample_id"} & set(frame.columns)
            np.testing.assert_allclose(frame.dqdv_mAh_V, 1 / 360, rtol=1e-11)
            outcome = result.outcomes().iloc[0]
            assert outcome.valid == 1 and outcome.n_grid == 7
        finally:
            result.close()
    finally:
        support.close()


def test_insufficient_voltage_support_remains_visible_as_outcome(tmp_path):
    """Retain a readable unavailable outcome instead of merging or fabricating support."""
    source = prepared(tmp_path, range(5), [1] * 5, [3, 3, 3.1, 3.1, 3.1])
    support = spool_dqdv_support(source, [_domain(4)], scratch_dir=tmp_path)
    try:
        result = build_dqdv_curves(
            support, config=DQDVConfig(trim_extremes_V=0), scratch_dir=tmp_path
        )
        try:
            assert result.table().empty
            outcome = result.outcomes().iloc[0]
            assert outcome.valid == 0
            assert outcome.reason == "fewer_than_3_distinct_voltages"
        finally:
            result.close()
    finally:
        support.close()


def _affine_result(tmp_path, *, smoothing=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = prepared(tmp_path, range(11), [1] * 11, np.linspace(3, 4, 11))
    support = spool_dqdv_support(source, [_domain(10)], scratch_dir=tmp_path)
    result = build_dqdv_curves(
        support,
        config=DQDVConfig(
            n_points=7,
            trim_extremes_V=0,
            smoothing=smoothing,
            window_points=5,
            polyorder=2,
        ),
        scratch_dir=tmp_path,
    )
    support.close()
    return result


def test_peak_reports_ties_edges_and_no_node_identifiers(tmp_path):
    """Report all bounded tied grid extrema without choosing a fabricated peak."""
    result = _affine_result(tmp_path)
    try:
        with closing(sqlite3.connect(result.path)) as database:
            database.execute("UPDATE curve_points SET dqdv_mAh_V=2")
            database.commit()
        peak = result.peak(
            cycle=1, half_cycle=1, step=1, portion=1, window_V=(3, 4), max_candidates=3
        )
        assert peak["valid"] and peak["ambiguous"]
        assert peak["candidate_count"] == 7 and peak["candidates_truncated"]
        assert peak["at_window_edge"] and peak["at_support_edge"]
        assert not any("id" in key for candidate in peak["candidates"] for key in candidate)
    finally:
        result.close()


def test_smoothed_result_keeps_unsmoothed_comparison_and_magnitude_is_explicit(tmp_path):
    """Preserve signed data while exposing opt-in smoothing comparison and magnitude."""
    result = _affine_result(tmp_path, smoothing=True)
    try:
        frame = result.table()
        assert frame.dqdv_unsmoothed_mAh_V.notna().all()
        magnitude = result.magnitude_table()
        np.testing.assert_allclose(magnitude.abs_dqdv_mAh_V, np.abs(frame.dqdv_mAh_V))
        assert "abs_dqdv_mAh_V" not in frame
    finally:
        result.close()


def test_peak_comparison_is_comparison_minus_reference(tmp_path):
    """Calculate a shift only for unambiguous same-direction signed extrema."""
    reference = _affine_result(tmp_path / "reference")
    comparison = _affine_result(tmp_path / "comparison")
    key = {"cycle": 1, "half_cycle": 1, "step": 1, "portion": 1}
    try:
        for result, voltage in ((reference, 3.5), (comparison, 3.6666666666666665)):
            with closing(sqlite3.connect(result.path)) as database:
                database.execute("UPDATE curve_points SET dqdv_mAh_V=0")
                database.execute(
                    "UPDATE curve_points SET dqdv_mAh_V=2 WHERE ABS(voltage_V-?)<1e-12",
                    (voltage,),
                )
                database.commit()
        compared = compare_prepared_dqdv_peaks(reference, key, comparison, key, (3, 4))
        assert compared["comparable"]
        assert compared["deltaV_V"] == pytest.approx(1 / 6)
    finally:
        reference.close()
        comparison.close()


def test_final_plot_uses_readable_series_and_display_only_magnitude(tmp_path):
    """Create a bounded interactive figure without exposing technical row identities."""
    result = _affine_result(tmp_path)
    try:
        plotted = result.plot(max_points=5)
        assert plotted.status == "complete"
        assert len(plotted.figure.data) == 1
        trace = plotted.figure.data[0]
        assert trace.name == "cycle 1 | half-cycle 1 | step 1 | portion 1:0"
        assert "ID=" not in trace.hovertemplate
        assert len(trace.x) <= 5
        magnitude = result.plot(magnitude=True, max_points=5)
        assert magnitude.spec.y == "abs_dqdv_mAh_V"
        np.testing.assert_allclose(magnitude.figure.data[0].y, np.abs(plotted.figure.data[0].y))
    finally:
        result.close()


def test_progressive_plot_emits_updates_before_completion(tmp_path):
    """Publish bounded running curve updates and a durable terminal status."""
    result = _affine_result(tmp_path)
    try:
        session = result.plot_progressive(max_points=20)
        try:
            session.start()
            updates = list(session.iter_updates())
            assert any(
                update.status == "running" and update.operation != "status" for update in updates
            )
            assert updates[-1].status == "complete"
        finally:
            session.close()
    finally:
        result.close()
