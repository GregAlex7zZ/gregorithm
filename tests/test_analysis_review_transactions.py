"""GITT/ICI tests: event analysis, windows, review transitions and transactional batch scenarios.

Expected values are computed here from the synthetic generating formulas, never read back from
the library under test.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gregorithm.diagnostics import DiagnosticCode
from gregorithm.elaboration import Step
from gregorithm.gitt_ici import (
    AnalysisOptions,
    AnalysisRecord,
    AnalysisSettings,
    AppliedSettings,
    BatchTask,
    CanonicalSample,
    CanonicalSegment,
    ComputeOutput,
    DelacourtOffsetMode,
    DelacourtResult,
    DiffusionMethod,
    DiffusionResult,
    FitWindowRequest,
    IndicatorColor,
    IntermediateCache,
    IrCorrection,
    MethodVariant,
    OcvSource,
    OcvTargetPlan,
    PhysicalGuidance,
    PhysicalInputs,
    Protocol,
    RecalculationScope,
    ResultIdentity,
    ResultSnapshot,
    ResultState,
    StaleBatchError,
    WindowBounds,
    WindowMode,
    WindowRole,
    associate_events,
    build_event_data,
    commit_batch,
    default_method_variants,
    delacourt_quality,
    delacourt_variant,
    delacourt_voltage,
    fit_sqrt_time,
    ocv_target_dependencies,
    plan_analysis,
    refine_physically_guided,
    resolve_guidance,
    run_batch,
    select_window,
)
from gregorithm.gitt_ici.windows import _series, _window_stats
from gregorithm.progress import CancellationToken, OperationCancelled, ProgressEvent

FINGERPRINT = "c" * 64
V_OVER_A = 1e-5
GEOMETRY = PhysicalInputs(1e-9, 1e-4, 1e-6)
PULSE_S = 100
REST_S = 60
SLOPE = 0.004
DELTA_U = 0.01
ETA = 0.005
REST_SLOPE = -0.003
U_START = 3.5

PulseFn = Callable[[int, float, float, float], float]


def ideal_pulse(_index: int, t: float, u_before: float, _u_after: float) -> float:
    return u_before + ETA + SLOPE * math.sqrt(t)


def dataset(
    pulses: int = 3,
    pulse_fn: PulseFn = ideal_pulse,
    *,
    polarity: int = 1,
    current: float = 1e-3,
) -> tuple[tuple[CanonicalSegment, ...], tuple[CanonicalSample, ...]]:
    """Rest - (CC pulse - rest)*n with sqrt-time pulses and sqrt-time early relaxations."""
    samples: list[CanonicalSample] = []
    segments: list[CanonicalSegment] = []
    state = {"row": 0, "t": 0.0, "cap": 0.0}

    def add(step: Step, count: int, voltage: Callable[[float], float], amps: float) -> None:
        first_row, first_t = state["row"], state["t"] + 1.0
        for k in range(1, count + 1):
            state["t"] += 1.0
            state["cap"] += abs(amps) / 3600
            cap = state["cap"]
            samples.append(
                CanonicalSample(
                    int(state["row"]),
                    state["t"],
                    amps,
                    voltage(float(k)),
                    1,
                    1,
                    cap if polarity > 0 else 0.0,
                    cap if polarity < 0 else 0.0,
                )
            )
            state["row"] += 1
        segments.append(
            CanonicalSegment(
                len(segments),
                int(first_row),
                int(state["row"]) - 1,
                first_t,
                state["t"],
                1,
                1,
                int(step),
                polarity if step is Step.CC else 0,
            )
        )

    add(Step.REST, REST_S, lambda _t: U_START, 0.0)
    for index in range(pulses):
        before = U_START + polarity * index * DELTA_U
        after = before + polarity * DELTA_U
        add(
            Step.CC,
            PULSE_S,
            lambda t, i=index, b=before, a=after: pulse_fn(i, t, b, a),
            polarity * current,
        )
        intercept = after - REST_SLOPE * math.sqrt(REST_S)
        add(Step.REST, REST_S, lambda t, a=intercept: a + REST_SLOPE * math.sqrt(t), 0.0)
    return tuple(segments), tuple(samples)


def prepared(protocol: Protocol = Protocol.GITT, **kwargs: object):  # type: ignore[no-untyped-def]
    segments, samples = dataset(**kwargs)  # type: ignore[arg-type]
    events = associate_events(segments, protocol=protocol, elaboration_fingerprint=FINGERPRINT)
    return build_event_data(events, segments, samples)


def d_formula(delta_u: float, tau: float, response: float) -> float:
    return 4 / math.pi * V_OVER_A**2 * (delta_u / (tau * response)) ** 2


VA = MethodVariant(DiffusionMethod.GITT_VA, OcvSource.REST_SAMPLE)
DELTA = MethodVariant(DiffusionMethod.GITT_DELTA, OcvSource.REST_SAMPLE)
ICI_PSEUDO = MethodVariant(DiffusionMethod.ICI, OcvSource.ICI_INTERCEPT)
ICI_SAMPLED = MethodVariant(DiffusionMethod.ICI, OcvSource.REST_SAMPLE)
SETTINGS = AnalysisSettings(geometry=GEOMETRY)


def run(data, variants, settings=SETTINGS, previous=None, **kwargs):  # type: ignore[no-untyped-def]
    plan = plan_analysis(data, variants=variants, settings=settings, **kwargs)
    return run_batch(plan.tasks, previous=previous or ResultSnapshot())


# --- options and planning -------------------------------------------------------------------


def test_d31_d32_defaults_compute_nothing_and_preselect_methods() -> None:
    gitt = AnalysisOptions(Protocol.GITT)
    assert gitt.auto_analysis_on_recognition is False
    assert gitt.selected_variants() == (VA, DELTA)
    assert AnalysisOptions(Protocol.ICI).selected_variants() == (ICI_PSEUDO,)
    assert default_method_variants(Protocol.GITT) == (VA, DELTA)
    assert delacourt_variant().variant == DelacourtOffsetMode.FIXED.value


def test_method_variants_reject_undefined_combinations() -> None:
    with pytest.raises(ValueError):
        MethodVariant(DiffusionMethod.GITT_VA, OcvSource.ICI_INTERCEPT)
    with pytest.raises(ValueError):
        MethodVariant(DiffusionMethod.GITT_SPHERE_FULL, OcvSource.REST_SAMPLE, "other")


def test_global_and_local_plans_cover_exact_targets() -> None:
    data = prepared()
    global_plan = plan_analysis(data, variants=(VA, DELTA), settings=SETTINGS)
    assert len(global_plan.tasks) == 6
    target = data[1].event.event_id
    local = plan_analysis(
        data,
        variants=(VA, DELTA),
        settings=SETTINGS,
        scope=RecalculationScope.LOCAL,
        event_id=target,
    )
    assert {task.identity.event_id for task in local.tasks} == {target}
    assert {task.identity.method for task in local.tasks} == {
        DiffusionMethod.GITT_VA,
        DiffusionMethod.GITT_DELTA,
    }
    with pytest.raises(ValueError, match="event_id"):
        plan_analysis(data, variants=(VA,), settings=SETTINGS, scope=RecalculationScope.LOCAL)


def test_incomplete_event_is_skipped_with_diagnostic() -> None:
    segments, samples = dataset()
    truncated = segments[:-1]
    events = associate_events(
        truncated, protocol=Protocol.GITT, elaboration_fingerprint=FINGERPRINT, complete_only=False
    )
    data = build_event_data(events, truncated, samples)
    plan = plan_analysis(data, variants=(VA,), settings=SETTINGS)
    assert len(plan.tasks) == 2
    assert [item.code for item in plan.skipped] == [DiagnosticCode.GITT_ICI_INCOMPLETE_EVENT]


def test_event_data_uses_canonical_rows_and_previous_current_reference() -> None:
    data = prepared()
    assert len(data[0].active) == PULSE_S and len(data[0].rest_after) == REST_S
    assert data[0].before_reference is None
    assert data[1].before_reference == data[0].active[-1]


# --- end-to-end scientific checks -----------------------------------------------------------


def test_gitt_va_and_delta_match_independent_formulas_and_coordinates() -> None:
    data = prepared()
    outcome = run(data, (VA, DELTA))
    tau = float(PULSE_S)
    va_expected = d_formula(DELTA_U, tau, SLOPE)
    delta_expected = (
        4 / (math.pi * tau) * V_OVER_A**2 * (DELTA_U / (ETA + SLOPE * math.sqrt(tau))) ** 2
    )
    for item in data:
        va = outcome.snapshot.current(VA.identity(item.event.event_id))
        delta = outcome.snapshot.current(DELTA.identity(item.event.event_id))
        assert va is not None and delta is not None
        assert va.diffusion_m2_s == pytest.approx(va_expected, rel=1e-8, abs=0)
        assert delta.diffusion_m2_s == pytest.approx(delta_expected, rel=1e-9, abs=0)
        assert va.quality is not None and va.quality.statistical is IndicatorColor.GREEN
        assert va.quality.physical is IndicatorColor.NOT_AVAILABLE
        assert delta.quality is not None and delta.quality.overall is IndicatorColor.GREEN
        assert va.coordinates is not None
        assert va.coordinates.capacity_start_ah == item.active[0].charge_capacity_ah
        assert va.coordinates.capacity_end_ah == item.active[-1].charge_capacity_ah


def test_gitt_delta_with_intercept_ir_equals_gitt_va_under_ideal_conditions() -> None:
    data = prepared()
    settings = replace(SETTINGS, ir_correction=IrCorrection.GITT_INTERCEPT)
    outcome = run(data, (VA, DELTA), settings=settings)
    for item in data:
        va = outcome.snapshot.current(VA.identity(item.event.event_id))
        delta = outcome.snapshot.current(DELTA.identity(item.event.event_id))
        assert va is not None and delta is not None
        assert delta.diffusion_m2_s == pytest.approx(va.diffusion_m2_s, rel=1e-8, abs=0)


def test_ir_sample_correction_reproduces_the_100_over_81_example() -> None:
    data = prepared()
    sampled = run(
        data,
        (DELTA,),
        settings=replace(SETTINGS, ir_correction=IrCorrection.SELECTED_SAMPLE),
    )
    intercept = run(
        data,
        (DELTA,),
        settings=replace(SETTINGS, ir_correction=IrCorrection.GITT_INTERCEPT),
    )
    identity = DELTA.identity(data[0].event.event_id)
    ratio = sampled.snapshot.current(identity).diffusion_m2_s / (  # type: ignore[union-attr,operator]
        intercept.snapshot.current(identity).diffusion_m2_s  # type: ignore[union-attr]
    )
    assert ratio == pytest.approx(100 / 81, rel=1e-9, abs=0)


def test_ici_pseudo_pairs_consecutive_interruptions_and_first_keeps_intermediates() -> None:
    data = prepared(Protocol.ICI)
    outcome = run(data, (ICI_PSEUDO,))
    first = outcome.snapshot.current(ICI_PSEUDO.identity(data[0].event.event_id))
    assert first is not None and first.state is ResultState.PROVISIONAL
    assert isinstance(first.result, DiffusionResult)
    assert first.result.diffusion_m2_s is None
    assert first.result.unavailable_reason == "ici_previous_interruption_missing"
    assert first.result.delta_u_v is None
    assert isinstance(first.details, dict) and first.details["intermediates"]["k_ohm_sqrt_s"] == (
        pytest.approx(-REST_SLOPE / 1e-3, rel=1e-12, abs=0)
    )
    expected = d_formula(DELTA_U, float(PULSE_S), REST_SLOPE)
    for item in data[1:]:
        record = outcome.snapshot.current(ICI_PSEUDO.identity(item.event.event_id))
        assert record is not None and record.diffusion_m2_s == pytest.approx(
            expected, rel=1e-8, abs=0
        )
        assert record.quality is not None
        assert record.quality.physical is IndicatorColor.NOT_AVAILABLE


def test_ici_applied_to_gitt_uses_same_pulse_ocv_pair() -> None:
    data = prepared()
    outcome = run(data, (ICI_SAMPLED,))
    expected = d_formula(DELTA_U, float(PULSE_S), REST_SLOPE)
    for item in data:
        record = outcome.snapshot.current(ICI_SAMPLED.identity(item.event.event_id))
        assert record is not None and record.diffusion_m2_s == pytest.approx(
            expected, rel=1e-8, abs=0
        )


def test_discharge_branch_keeps_signs_and_capacity_coordinate() -> None:
    data = prepared(polarity=-1)
    outcome = run(data, (VA, DELTA))
    expected = d_formula(DELTA_U, float(PULSE_S), SLOPE)
    record = outcome.snapshot.current(VA.identity(data[1].event.event_id))
    assert record is not None and record.diffusion_m2_s == pytest.approx(expected, rel=1e-8, abs=0)
    delta = outcome.snapshot.current(DELTA.identity(data[1].event.event_id))
    assert delta is not None and delta.quality is not None
    assert delta.quality.overall is IndicatorColor.RED  # ideal pulse rises while OCV falls
    assert record.coordinates is not None and record.coordinates.polarity == -1
    assert record.coordinates.capacity_start_ah == data[1].active[0].discharge_capacity_ah


def test_missing_geometry_keeps_intermediates_without_fabricated_d() -> None:
    data = prepared()
    outcome = run(data, (VA,), settings=AnalysisSettings())
    record = outcome.snapshot.current(VA.identity(data[0].event.event_id))
    assert record is not None and record.state is ResultState.PROVISIONAL
    assert isinstance(record.result, DiffusionResult)
    assert record.result.diffusion_m2_s is None
    assert record.result.unavailable_reason == "physical_input_missing"
    assert record.result.delta_u_v == pytest.approx(DELTA_U, rel=1e-12, abs=0)
    accepted = outcome.snapshot.accept_all(
        method=DiffusionMethod.GITT_VA, variant="default", ocv_source=OcvSource.REST_SAMPLE
    )
    assert accepted.current(record.identity).state is ResultState.PROVISIONAL  # type: ignore[union-attr]


def test_delacourt_recovers_diffusion_and_requires_radius() -> None:
    radius = GEOMETRY.particle_radius_m
    assert radius is not None
    true_log_theta = -1.3

    def sphere(_index: int, t: float, before: float, after: float) -> float:
        return float(
            delacourt_voltage(
                t,
                log10_theta_tau=true_log_theta,
                pulse_duration_s=float(PULSE_S),
                initial_ocv_v=before,
                final_ocv_v=after,
                offset_v=0.0,
            )
        )

    data = prepared(pulse_fn=sphere)
    variant = delacourt_variant()
    # Ideal data without an ohmic jump: the exact model offset is zero, so no IR source is used.
    exact = replace(SETTINGS, delacourt_ir_correction=IrCorrection.NONE)
    outcome = run(data, (variant,), settings=exact)
    record = outcome.snapshot.current(variant.identity(data[1].event.event_id))
    assert record is not None and isinstance(record.result, DelacourtResult)
    expected = 10**true_log_theta * radius**2 / PULSE_S
    assert record.diffusion_m2_s == pytest.approx(expected, rel=1e-6, abs=0)
    no_radius = run(data, (variant,), settings=AnalysisSettings(PhysicalInputs(1e-9, 1e-4)))
    failed = no_radius.snapshot.current(variant.identity(data[1].event.event_id))
    assert failed is not None and failed.state is ResultState.ERROR
    assert failed.error is not None and failed.error.startswith("particle_radius_required")


def test_relaxation_extrapolated_source_is_on_demand_and_warns_far_target() -> None:
    data = prepared()
    fitted = MethodVariant(DiffusionMethod.GITT_VA, OcvSource.RELAXATION_EXTRAPOLATED)
    missing = run(data, (fitted,))
    assert missing.recoverable_errors == 3  # no target plan: never silently sampled instead
    settings = replace(SETTINGS, ocv_targets=OcvTargetPlan(10_800))
    outcome = run(data, (fitted,), settings=settings)
    record = outcome.snapshot.current(fitted.identity(data[1].event.event_id))
    assert record is not None and isinstance(record.details, dict)
    after = record.details["ocv_after"]
    assert isinstance(after, dict) and after["source"] == "relaxation_extrapolated"
    assert after["warning"] == DiagnosticCode.GITT_ICI_EXTRAPOLATION_WARNING.value
    assert after["extrapolation_ratio"] == pytest.approx(10_800 / REST_S, rel=1e-12, abs=0)


# --- windows and indicators -----------------------------------------------------------------


def window_samples(
    voltage: Callable[[float], float], count: int = 120
) -> tuple[CanonicalSample, ...]:
    return tuple(
        CanonicalSample(index, float(index), 1.0, voltage(float(index)), 1, 1, 0.0, 0.0)
        for index in range(1, count + 1)
    )


def test_prefix_sum_window_slopes_match_reference_ols() -> None:
    rng = np.random.default_rng(7)
    rows = window_samples(lambda t: 3 + 0.01 * math.sqrt(t) + 1e-4 * rng.standard_normal())
    series = _series(rows, 0.0)
    for start, end in ((0, 10), (5, 80), (30, 119)):
        slope, sse, _, _ = _window_stats(series, np.array([start]), np.array([end]))
        reference = fit_sqrt_time(
            rows,
            reference_time_s=0.0,
            fit_start_s=float(series.distinct[start]),
            fit_end_s=float(series.distinct[end]),
        )
        assert slope[0] == pytest.approx(reference.slope_v_per_sqrt_s, rel=1e-9, abs=0)
        assert math.sqrt(sse[0] / (end - start + 1)) == pytest.approx(
            reference.rmse_v, rel=1e-6, abs=0
        )


def test_statistical_search_prefers_longest_adequate_window() -> None:
    rows = window_samples(lambda t: 3 + 0.01 * math.sqrt(t))
    selected = select_window(
        rows,
        reference_time_s=0.0,
        protocol=Protocol.GITT,
        allowed=WindowBounds(1, 120),
        mode=WindowMode.AUTO_STATISTICAL,
    )
    assert selected.selected == WindowBounds(1, 120)
    assert selected.quality.statistical is IndicatorColor.GREEN


def test_curvature_is_detected_by_neighbourhood_stability_and_avoided() -> None:
    def bent(t: float) -> float:
        return 3 + 0.01 * math.sqrt(t) + (0.0 if t <= 40 else 1e-3 * (t - 40))

    rows = window_samples(bent)
    fixed = select_window(
        rows,
        reference_time_s=0.0,
        protocol=Protocol.GITT,
        allowed=WindowBounds(1, 120),
        mode=WindowMode.AUTO_FIXED,
    )
    assert fixed.selected == WindowBounds(1, 120)
    assert fixed.quality.statistical is IndicatorColor.RED
    assert fixed.quality.q_stability_pct is not None and fixed.quality.q_stability_pct > 5
    assert fixed.quality.needs_review
    # An operator-edited allowed range keeps the search inside the diffusive regime.
    restricted = select_window(
        rows,
        reference_time_s=0.0,
        protocol=Protocol.GITT,
        allowed=WindowBounds(1, 40),
        mode=WindowMode.AUTO_STATISTICAL,
    )
    assert restricted.selected == WindowBounds(1, 40)
    assert restricted.fit.slope_v_per_sqrt_s == pytest.approx(0.01, rel=1e-9, abs=0)
    assert restricted.quality.statistical is IndicatorColor.GREEN


def test_zero_slope_and_two_point_manual_windows_never_receive_false_green() -> None:
    rng = np.random.default_rng(1)
    flat = window_samples(lambda _t: 3 + 1e-4 * rng.standard_normal())
    selection = select_window(
        flat,
        reference_time_s=0.0,
        protocol=Protocol.GITT,
        allowed=WindowBounds(1, 120),
        mode=WindowMode.AUTO_FIXED,
    )
    assert selection.quality.statistical is IndicatorColor.NOT_AVAILABLE
    assert selection.quality.reason == "slope_not_identifiable"
    manual = select_window(
        window_samples(lambda t: 3 + 0.01 * math.sqrt(t)),
        reference_time_s=0.0,
        protocol=Protocol.GITT,
        allowed=WindowBounds(1, 120),
        mode=WindowMode.MANUAL,
        manual=WindowBounds(1, 2),
    )
    assert manual.quality.q_linearity_pct is None
    assert manual.quality.statistical is IndicatorColor.NOT_AVAILABLE
    assert manual.quality.needs_review is False


def test_ici_guided_mode_is_rejected_in_selector_and_n_d_in_analysis() -> None:
    rows = window_samples(lambda t: 3 - 0.01 * math.sqrt(t))
    with pytest.raises(ValueError, match="not validated"):
        select_window(
            rows,
            reference_time_s=0.0,
            protocol=Protocol.ICI,
            allowed=WindowBounds(1, 120),
            mode=WindowMode.PHYSICALLY_GUIDED,
            particle_radius_m=1e-6,
            preliminary_diffusion_m2_s=1e-14,
            physical_coefficient=0.1,
        )
    data = prepared(Protocol.ICI)
    windows = {
        (item.event.event_id, WindowRole.REST_AFTER): FitWindowRequest(WindowMode.PHYSICALLY_GUIDED)
        for item in data
    }
    outcome = run(data, (ICI_PSEUDO,), windows=windows)
    record = outcome.snapshot.current(ICI_PSEUDO.identity(data[1].event.event_id))
    assert record is not None and isinstance(record.details, dict)
    assert record.details["window"]["reason"] == "ici_useful_time_not_validated"
    assert record.quality is not None and record.quality.physical is IndicatorColor.NOT_AVAILABLE


def test_guided_refinement_stays_inside_useful_time_and_reports_non_convergence() -> None:
    rows = window_samples(lambda t: 3 + 0.01 * math.sqrt(t))
    refinement = refine_physically_guided(
        rows,
        reference_time_s=0.0,
        allowed=WindowBounds(1, 120),
        particle_radius_m=1e-6,
        initial_diffusion_m2_s=1e-13,
        physical_coefficient=5.0,
        diffusion_from_fit=lambda _fit: 1e-13,
    )
    assert refinement.converged and refinement.iterations == 2
    assert refinement.selection.selected.end_s <= 50.0
    assert refinement.selection.quality.physical is IndicatorColor.GREEN
    stopped = refine_physically_guided(
        rows,
        reference_time_s=0.0,
        allowed=WindowBounds(1, 120),
        particle_radius_m=1e-6,
        initial_diffusion_m2_s=1e-13,
        physical_coefficient=5.0,
        diffusion_from_fit=lambda _fit: None,
    )
    assert not stopped.converged and stopped.reason == "diffusion_unavailable_after_fit"


def test_delacourt_residual_structure_lowers_fit_color_despite_high_r2() -> None:
    times = tuple(float(value) for value in range(1, 41))
    residuals = tuple(1e-4 * math.sin(value / 8) for value in times)
    result = DelacourtResult(
        1e-14, 1e-16, -1.0, 0.1, 100.0, 0.0, tuple(range(40)), times,
        tuple(3 + 0.01 * value for value in times), residuals, 1e-4, 0.999, 4e-7,
        False, True, 1.0, 50,
    )  # fmt: skip
    quality = delacourt_quality(result)
    assert quality.statistical is IndicatorColor.RED
    assert quality.physical is IndicatorColor.GREEN
    assert quality.overall is IndicatorColor.RED


# --- review and transactions ----------------------------------------------------------


def accepted_snapshot(data, variants=(VA, DELTA)):  # type: ignore[no-untyped-def]
    snapshot = run(data, variants).snapshot
    for variant in variants:
        snapshot = snapshot.accept_all(
            method=variant.method, variant=variant.variant, ocv_source=variant.ocv_source
        )
    return snapshot


def test_scenario_1_successful_recalculation_returns_to_provisional() -> None:
    data = prepared()
    snapshot = accepted_snapshot(data)
    identity = VA.identity(data[0].event.event_id)
    assert snapshot.current(identity).exportable  # type: ignore[union-attr]
    outcome = run(
        data,
        (VA,),
        previous=snapshot,
        scope=RecalculationScope.LOCAL,
        event_id=data[0].event.event_id,
        windows={
            (data[0].event.event_id, WindowRole.PULSE): FitWindowRequest(
                WindowMode.MANUAL, WindowBounds(10, 80)
            )
        },
    )
    snapshot = commit_batch(snapshot, outcome)
    current = snapshot.current(identity)
    assert current is not None and current.state is ResultState.PROVISIONAL
    assert not current.exportable
    assert len([r for r in snapshot.records if r.identity == identity]) == 1


def test_scenario_2_failed_recalculation_replaces_value_and_batch_continues() -> None:
    data = prepared()
    snapshot = accepted_snapshot(data)
    windows = {
        (data[0].event.event_id, WindowRole.PULSE): FitWindowRequest(
            WindowMode.MANUAL, WindowBounds(10.0, 10.5)
        )
    }
    outcome = run(data, (VA,), previous=snapshot, windows=windows)
    snapshot = commit_batch(snapshot, outcome)
    failed = snapshot.current(VA.identity(data[0].event.event_id))
    assert failed is not None and failed.state is ResultState.ERROR and failed.result is None
    assert failed not in snapshot.exportable_records()
    assert outcome.recoverable_errors == 1 and outcome.completed == 3
    other = snapshot.current(VA.identity(data[1].event.event_id))
    assert other is not None and other.state is ResultState.PROVISIONAL and other.has_diffusion


def test_scenario_3_cancelled_batch_preserves_snapshot_and_cache(tmp_path: Path) -> None:
    data = prepared()
    snapshot = accepted_snapshot(data)
    token = CancellationToken()
    events: list[ProgressEvent] = []

    def sink(event: ProgressEvent) -> None:
        events.append(event)
        if event.completed == 2:
            token.cancel()

    plan = plan_analysis(data, variants=(VA,), settings=SETTINGS)
    cache = IntermediateCache(tmp_path / "cache")
    with pytest.raises(OperationCancelled):
        run_batch(plan.tasks, previous=snapshot, cache=cache, progress=sink, cancel=token)
    assert all(record.state is ResultState.ACCEPTED for record in snapshot.records)
    assert not tuple((tmp_path / "cache").glob("*.json"))


def test_scenario_4_deselected_method_is_kept_without_recalculation() -> None:
    data = prepared()
    snapshot = run(data, (VA, ICI_SAMPLED)).snapshot
    ici_before = snapshot.current(ICI_SAMPLED.identity(data[0].event.event_id))
    outcome = run(data, (VA,), previous=snapshot)
    snapshot = commit_batch(snapshot, outcome)
    assert snapshot.current(ICI_SAMPLED.identity(data[0].event.event_id)) == ici_before
    assert all(record.identity.method is DiffusionMethod.GITT_VA for record in outcome.records)


def test_scenario_5_unapplied_edit_changes_nothing_apply_invalidates_dependents() -> None:
    data = prepared()
    delacourt = delacourt_variant()
    snapshot = accepted_snapshot(data, (VA, DELTA, delacourt))
    settings = AppliedSettings({"active_volume_cm3": 1e-3}, {"active_volume_cm3": 1e-3})
    edited = settings.edit({"active_volume_cm3": 2e-3})
    assert edited.dependency_fingerprints() == settings.dependency_fingerprints()
    applied, changed = edited.apply()
    assert changed and applied.applied == {"active_volume_cm3": 2e-3}
    new_settings = replace(SETTINGS, geometry=PhysicalInputs(2e-9, 1e-4, 1e-6))
    stale = snapshot.invalidate(new_settings.dependency_fingerprints())
    for record in stale.records:
        assert record.stale is (record.identity.method is not DiffusionMethod.GITT_SPHERE_FULL)
    assert {r.identity.method for r in stale.exportable_records()} == {
        DiffusionMethod.GITT_SPHERE_FULL
    }
    with pytest.raises(ValueError, match="stale"):
        stale.accept(VA.identity(data[0].event.event_id))
    refreshed = commit_batch(stale, run(data, (VA, DELTA), settings=new_settings, previous=stale))
    va = refreshed.current(VA.identity(data[0].event.event_id))
    assert va is not None and not va.stale and va.state is ResultState.PROVISIONAL
    assert va.diffusion_m2_s == pytest.approx(
        4 * d_formula(DELTA_U, PULSE_S, SLOPE), rel=1e-8, abs=0
    )


def test_scenario_6_target_override_invalidates_only_dependent_rest() -> None:
    data = prepared()
    fitted = MethodVariant(DiffusionMethod.GITT_VA, OcvSource.RELAXATION_EXTRAPOLATED)
    plan = OcvTargetPlan(10_800)
    settings = replace(SETTINGS, ocv_targets=plan)
    snapshot = run(data, (fitted, VA), settings=settings).snapshot
    rest = data[2].event.rest_after_segment_id
    assert rest is not None
    override = plan.with_override(rest, 3_600)

    stale = snapshot.invalidate(ocv_target_dependencies(override, [rest]))
    flagged = {record.identity for record in stale.records if record.stale}
    assert flagged == {fitted.identity(data[2].event.event_id)}


def test_scenario_7_provisional_guidance_is_traced_and_never_accepted() -> None:
    data = prepared()
    snapshot = run(data, (VA,)).snapshot
    guidance = PhysicalGuidance(VA, physical_coefficient=4.0)
    value = resolve_guidance(snapshot, data[0].event.event_id, VA)
    assert value is not None and value.provisional
    windows = {
        (item.event.event_id, WindowRole.PULSE): FitWindowRequest(WindowMode.PHYSICALLY_GUIDED)
        for item in data
    }
    outcome = run(
        data,
        (VA,),
        previous=snapshot,
        windows=windows,
        guidance=guidance,
        guidance_snapshot=snapshot,
    )
    record = outcome.snapshot.current(VA.identity(data[0].event.event_id))
    assert record is not None and isinstance(record.details, dict)
    traced = record.details["guidance"]
    assert isinstance(traced, dict) and traced["source"]["provisional"] is True
    useful = 4.0 * 1e-12 / value.diffusion_m2_s
    assert record.details["window"]["selected_s"][1] <= useful
    assert record.quality is not None and record.quality.physical is IndicatorColor.GREEN
    assert snapshot.current(value.identity).state is ResultState.PROVISIONAL  # type: ignore[union-attr]
    # No implicit fallback: a source without a value for the event leaves physics N/D.
    empty = run(data, (VA,), windows=windows, guidance=guidance, guidance_snapshot=ResultSnapshot())
    missing = empty.snapshot.current(VA.identity(data[0].event.event_id))
    assert missing is not None and isinstance(missing.details, dict)
    assert missing.details["guidance"] == {
        "applied": False,
        "reason": "guidance_diffusion_unavailable",
    }
    assert missing.quality is not None
    assert missing.quality.physical is IndicatorColor.NOT_AVAILABLE


def test_scenario_8_uncertainty_only_when_estimable() -> None:
    rng = np.random.default_rng(3)

    def noisy(_index: int, t: float, before: float, _after: float) -> float:
        return before + ETA + SLOPE * math.sqrt(t) + 1e-5 * rng.standard_normal()

    data = prepared(pulse_fn=noisy)
    outcome = run(data, (VA,))
    record = outcome.snapshot.current(VA.identity(data[0].event.event_id))
    assert record is not None and isinstance(record.result, DiffusionResult)
    assert record.result.standard_uncertainty_m2_s is not None
    assert 0 < record.result.standard_uncertainty_m2_s < record.diffusion_m2_s  # type: ignore[operator]
    windows = {
        (data[0].event.event_id, WindowRole.PULSE): FitWindowRequest(
            WindowMode.MANUAL, WindowBounds(1, 2)
        )
    }
    two = run(data, (VA,), windows=windows).snapshot.current(VA.identity(data[0].event.event_id))
    assert two is not None and isinstance(two.result, DiffusionResult)
    assert two.result.standard_uncertainty_m2_s is None


def test_scenario_9_non_positive_d_is_not_log_plottable_but_kept() -> None:
    identity = ResultIdentity("e", DiffusionMethod.GITT_DELTA, "default", OcvSource.REST_SAMPLE)
    zero = AnalysisRecord(
        identity,
        "fp",
        ResultState.PROVISIONAL,
        DiffusionResult(DiffusionMethod.GITT_DELTA, 0.0, None, 0.0, 10.0, None, 0.01, None),
        None,
    )
    snapshot = ResultSnapshot((zero,))
    assert zero.has_diffusion and not zero.log_plottable
    assert snapshot.current(identity) == zero


def test_scenario_10_no_accepted_results_means_nothing_exportable() -> None:
    data = prepared()
    snapshot = run(data, (VA,)).snapshot
    assert snapshot.exportable_records() == ()
    revoked = snapshot.accept(VA.identity(data[0].event.event_id)).revoke(
        VA.identity(data[0].event.event_id)
    )
    assert revoked.exportable_records() == ()


def test_accept_all_preserves_rejections_and_is_idempotent() -> None:
    data = prepared()
    snapshot = run(data, (VA, DELTA)).snapshot
    rejected = VA.identity(data[1].event.event_id)
    snapshot = snapshot.reject(rejected)
    once = snapshot.accept_all(
        method=DiffusionMethod.GITT_VA, variant="default", ocv_source=OcvSource.REST_SAMPLE
    )
    twice = once.accept_all(
        method=DiffusionMethod.GITT_VA, variant="default", ocv_source=OcvSource.REST_SAMPLE
    )
    assert once == twice and once.fingerprint == twice.fingerprint
    assert once.current(rejected).state is ResultState.REJECTED  # type: ignore[union-attr]
    assert all(
        record.state is ResultState.PROVISIONAL
        for record in once.records
        if record.identity.method is DiffusionMethod.GITT_DELTA
    )


def test_late_batch_is_rejected_after_review_change() -> None:
    data = prepared()
    snapshot = run(data, (VA,)).snapshot
    outcome = run(data, (VA,), previous=snapshot)
    changed = snapshot.accept(VA.identity(data[0].event.event_id))
    with pytest.raises(StaleBatchError):
        commit_batch(changed, outcome)
    # Deterministic recomputation of unchanged inputs commits to identical content.
    assert commit_batch(snapshot, outcome).fingerprint == snapshot.fingerprint
    recalculated = run(
        data,
        (VA,),
        previous=changed,
        scope=RecalculationScope.LOCAL,
        event_id=data[0].event.event_id,
    )
    committed = commit_batch(changed, recalculated)
    assert committed.current(VA.identity(data[0].event.event_id)).state is (  # type: ignore[union-attr]
        ResultState.PROVISIONAL
    )


def test_cache_reuses_automatic_windows_and_misses_on_threshold_change(tmp_path: Path) -> None:
    data = prepared()
    cache = IntermediateCache(tmp_path / "cache")
    first_plan = plan_analysis(data, variants=(VA, DELTA), settings=SETTINGS)
    first = run_batch(first_plan.tasks, previous=ResultSnapshot(), cache=cache)
    second = run_batch(first_plan.tasks, previous=first.snapshot, cache=cache)
    assert first.cache_hits == 0 and second.cache_hits == 3
    for a, b in zip(first.records, second.records, strict=True):
        assert a.diffusion_m2_s == b.diffusion_m2_s
    stricter = replace(
        SETTINGS,
        thresholds=replace(SETTINGS.thresholds, linearity_green_pct=1.0),
    )
    third_plan = plan_analysis(data, variants=(VA,), settings=stricter)
    third = run_batch(third_plan.tasks, previous=ResultSnapshot(), cache=cache)
    assert third.cache_hits == 0


def test_tasks_require_generation_dependency() -> None:
    with pytest.raises(ValueError, match="generation"):
        BatchTask(
            ResultIdentity("e", DiffusionMethod.GITT_VA, "default", OcvSource.REST_SAMPLE),
            (("event", "e"),),
            {},
            lambda _cached: ComputeOutput(
                DiffusionResult(DiffusionMethod.GITT_VA, 1.0, None, 0.1, 1.0, 1.0, None, None)
            ),
        )


def test_progress_reports_impulses_and_method_phases() -> None:
    data = prepared()
    events: list[ProgressEvent] = []
    plan = plan_analysis(data, variants=(VA, DELTA), settings=SETTINGS)
    run_batch(plan.tasks, previous=ResultSnapshot(), progress=events.append)
    assert {event.unit for event in events} == {"impulses"}
    assert {event.phase for event in events} >= {
        "starting",
        "calculating:gitt_va",
        "calculating:gitt_delta",
        "complete",
    }
    assert [event.completed for event in events if event.phase == "complete"] == [3]


def test_large_batch_scales_linearly() -> None:
    def compute(_cached: object) -> ComputeOutput:
        return ComputeOutput(
            DiffusionResult(DiffusionMethod.GITT_VA, 1e-14, None, 0.1, 10.0, 0.02, None, None)
        )

    tasks = tuple(
        BatchTask(
            ResultIdentity(
                f"e{index:06d}", DiffusionMethod.GITT_VA, "default", OcvSource.REST_SAMPLE
            ),
            (("generation", "g"),),
            {"i": index},
            compute,
        )
        for index in range(20_000)
    )
    started = time.perf_counter()
    outcome = run_batch(tasks, previous=ResultSnapshot())
    assert len(outcome.snapshot.records) == 20_000
    assert time.perf_counter() - started < 30


def test_corrupted_cache_value_is_a_miss_not_a_batch_failure(tmp_path: Path) -> None:
    data = prepared()
    plan = plan_analysis(data, variants=(VA,), settings=SETTINGS)
    cache = IntermediateCache(tmp_path / "cache")
    first = run_batch(plan.tasks, previous=ResultSnapshot(), cache=cache)
    for path in (tmp_path / "cache").glob("*.json"):
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace('"pulse":[', '"pulse":["bad",'), encoding="utf-8")
    second = run_batch(plan.tasks, previous=ResultSnapshot(), cache=cache)
    assert second.recoverable_errors == 0
    for a, b in zip(first.records, second.records, strict=True):
        assert a.diffusion_m2_s == b.diffusion_m2_s


def test_intercept_ir_uses_the_guided_gitt_pulse_window() -> None:
    data = prepared()
    snapshot = run(data, (VA,)).snapshot
    windows = {
        (item.event.event_id, WindowRole.PULSE): FitWindowRequest(WindowMode.PHYSICALLY_GUIDED)
        for item in data
    }
    settings = replace(SETTINGS, ir_correction=IrCorrection.GITT_INTERCEPT)
    outcome = run(
        data,
        (VA, DELTA),
        settings=settings,
        windows=windows,
        guidance=PhysicalGuidance(VA, physical_coefficient=4.0),
        guidance_snapshot=snapshot,
    )
    event_id = data[0].event.event_id
    va = outcome.snapshot.current(VA.identity(event_id))
    delta = outcome.snapshot.current(DELTA.identity(event_id))
    assert va is not None and delta is not None
    assert isinstance(va.details, dict) and isinstance(delta.details, dict)
    assert delta.details["ir_window"]["selected_s"] == va.details["window"]["selected_s"]
