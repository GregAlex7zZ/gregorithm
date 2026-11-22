"""GITT/ICI analysis regression tests: Delacourt defaults, window ranking, canonical adapter,
radius-only Delacourt, non-CC events, resistances and the compact public API.

Every scientific expectation is computed here from independent mathematics: the spherical
response from its own eigenvalues or from its Laplace-domain power series, D from the closed-form
formulas, and resistances from the generating values. No expected value is produced by the kernel
under test.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

import gregorithm
from gregorithm.elaboration import ElaborationRequest, Step, elaborate
from gregorithm.gitt_ici import (
    SPHERICAL_SWITCH_THETA,
    AnalysisSettings,
    CanonicalSample,
    CanonicalSegment,
    DelacourtOffsetMode,
    DelacourtParameters,
    DelacourtResult,
    DiffusionMethod,
    DiffusionResult,
    InitialWindowRule,
    IrCorrection,
    MethodVariant,
    OcvSource,
    PhysicalInputs,
    Protocol,
    ResultSnapshot,
    ResultState,
    WindowBounds,
    WindowMode,
    WindowRanking,
    associate_events,
    build_event_data,
    delacourt_variant,
    fit_delacourt,
    load_canonical_scope,
    physical_inputs_si,
    plan_analysis,
    run_batch,
    select_window,
    spherical_f,
    suggest_initial_window,
)
from gregorithm.gitt_ici.contracts import LengthUnit
from gregorithm.importing import ImportRequest, import_source

FINGERPRINT = "d" * 64
VA = MethodVariant(DiffusionMethod.GITT_VA, OcvSource.REST_SAMPLE)
DELTA = MethodVariant(DiffusionMethod.GITT_DELTA, OcvSource.REST_SAMPLE)
ICI = MethodVariant(DiffusionMethod.ICI, OcvSource.REST_SAMPLE)
SPHERE = delacourt_variant()
RADIUS = 1e-6


# --- independent references for the spherical response ---------------------------------------


def _eigenvalues(count: int) -> np.ndarray:
    exact = [
        brentq(
            lambda x: math.tan(x) - x, j * math.pi + 1e-9, (j + 0.5) * math.pi - 1e-9, xtol=1e-15
        )
        for j in range(1, 61)
    ]
    q = (np.arange(61, count + 1) + 0.5) * math.pi
    return np.concatenate((exact, q - 1 / q - 2 / (3 * q**3) - 13 / (15 * q**5)))


LAMBDA_SQUARED = _eigenvalues(60_000) ** 2
TAIL = 0.1 - float(np.sum(1 / LAMBDA_SQUARED[::-1]))  # Σ 1/λ² = 1/10 exactly


def eigen_reference(theta: float) -> float:
    """f = 3θ + 2Σ(1 - e^{-λ²θ})/λ² + 2·tail, accurate for θ ≥ 1e-5 with 60 000 terms."""
    terms = -np.expm1(-LAMBDA_SQUARED * theta) / LAMBDA_SQUARED
    return 3 * theta + 2 * math.fsum(terms[::-1]) + 2 * TAIL


def laplace_series_reference(theta: float) -> float:
    """Short-time series Σ θ^{(k+1)/2}/Γ((k+3)/2) of 1/(s(√s - 1)), exponentially accurate."""
    return math.fsum(theta ** ((k + 1) / 2) / math.gamma((k + 3) / 2) for k in range(80))


@pytest.mark.parametrize(
    "theta", [1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 1.9e-2, 2e-2, 0.05, 0.3, 2.0, 40.0]
)
def test_a01_spherical_response_matches_independent_eigen_series(theta: float) -> None:
    assert float(spherical_f(theta)) == pytest.approx(eigen_reference(theta), rel=1e-11, abs=0)


@pytest.mark.parametrize("theta", [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1.99e-2])
def test_a01_short_time_matches_independent_laplace_power_series(theta: float) -> None:
    assert float(spherical_f(theta)) == pytest.approx(
        laplace_series_reference(theta), rel=1e-13, abs=0
    )


def test_a01_switch_is_continuous_and_limits_hold() -> None:
    below = float(spherical_f(np.nextafter(SPHERICAL_SWITCH_THETA, 0)))
    above = float(spherical_f(SPHERICAL_SWITCH_THETA))
    assert above == pytest.approx(below, rel=1e-14, abs=0)
    assert float(spherical_f(0.0)) == 0.0
    theta = 1e-12
    assert float(spherical_f(theta)) == pytest.approx(
        2 * math.sqrt(theta / math.pi), rel=1e-5, abs=0
    )
    assert float(spherical_f(60.0)) == pytest.approx(3 * 60.0 + 0.2, rel=1e-15, abs=0)


@pytest.mark.parametrize("log_theta", [-3.0, -2.0, -1.5, -1.0])
def test_a01_delacourt_fit_recovers_d_from_independently_generated_data(log_theta: float) -> None:
    tau, u0, u1 = 100.0, 3.5, 3.51
    theta_tau = 10**log_theta
    rows = [
        CanonicalSample(
            i, t, 1e-3, u0 + (u1 - u0) * eigen_reference(theta_tau * t / tau) / (3 * theta_tau),
            1, 1, 0.0, 0.0,
        )
        for i, t in enumerate(np.arange(1.0, 101.0))
    ]  # fmt: skip
    result = fit_delacourt(
        rows, reference_time_s=0.0, parameters=DelacourtParameters(RADIUS, tau, u0, u1)
    )
    assert result.diffusion_m2_s == pytest.approx(theta_tau * RADIUS**2 / tau, rel=1e-6, abs=0)


# --- synthetic canonical events ---------------------------------------------------------------


def events_dataset(
    pulse_step: Step = Step.CC,
    *,
    jump: float = 0.0,
    log_theta: float | None = None,
    pulses: int = 2,
) -> tuple[tuple[CanonicalSegment, ...], list[CanonicalSample]]:
    """rest - (pulse - rest)*n; sqrt-time or exact spherical pulses with an optional ohmic jump."""
    samples: list[CanonicalSample] = []
    segments: list[CanonicalSegment] = []
    row, now, ocv = 0, 0.0, 3.5

    def add(step: Step, values: list[float], amps: float) -> None:
        nonlocal row, now
        first_row, first_t = row, now + 1
        for value in values:
            now += 1
            samples.append(CanonicalSample(row, now, amps, value, 1, 1, now / 3.6e6, 0.0))
            row += 1
        segments.append(
            CanonicalSegment(
                len(segments), first_row, row - 1, first_t, now, 1, 1, int(step),
                1 if step is not Step.REST else 0,
            )
        )  # fmt: skip

    add(Step.REST, [ocv] * 60, 0.0)
    for _ in range(pulses):
        if log_theta is None:
            pulse = [ocv + 0.005 + jump + 0.004 * math.sqrt(t) for t in range(1, 101)]
        else:
            theta_tau = 10**log_theta
            pulse = [
                ocv + jump + 0.01 * eigen_reference(theta_tau * t / 100) / (3 * theta_tau)
                for t in range(1, 101)
            ]
        add(pulse_step, pulse, 1e-3)
        ocv += 0.01
        add(
            Step.REST,
            [ocv + 0.003 * math.sqrt(60) - 0.003 * math.sqrt(t) for t in range(1, 61)],
            0.0,
        )
    return tuple(segments), samples


def analyse(segments, samples, variants, settings, protocol=Protocol.GITT):  # type: ignore[no-untyped-def]
    events = associate_events(segments, protocol=protocol, elaboration_fingerprint=FINGERPRINT)
    data = build_event_data(events, segments, samples)
    plan = plan_analysis(data, variants=variants, settings=settings)
    return data, run_batch(plan.tasks, previous=ResultSnapshot())


# --- Delacourt fixed-offset default -------------------------------------------------------


def test_a02_default_fixed_offset_uses_initial_jump_and_records_it() -> None:
    settings = AnalysisSettings(geometry=PhysicalInputs(particle_radius_m=RADIUS))
    assert settings.ir_correction is IrCorrection.NONE
    assert settings.delacourt_ir_correction is IrCorrection.SELECTED_SAMPLE
    segments, samples = events_dataset(log_theta=-1.0, jump=0.02)
    data, outcome = analyse(segments, samples, (SPHERE,), settings)
    record = outcome.snapshot.current(SPHERE.identity(data[0].event.event_id))
    assert record is not None and isinstance(record.details, dict)
    v0 = data[0].rest_before[-1].voltage_v
    first = data[0].active[0]
    expected_c = first.voltage_v - v0  # jump + diffusion during the first second
    ir = record.details["delacourt_ir"]
    assert isinstance(ir, dict)
    assert ir["mode"] == "selected_sample"
    assert ir["sample_row"] == first.source_row and ir["sample_time_s"] == 1.0
    assert ir["offset_v"] == pytest.approx(expected_c, rel=1e-12, abs=0)
    fixed = record.details["fixed_offset"]
    assert isinstance(fixed, dict)
    assert fixed["baseline_v"] == 0.0  # sampled U0 equals V0
    assert fixed["total_offset_v"] == pytest.approx(expected_c, rel=1e-12, abs=0)
    assert isinstance(record.result, DelacourtResult)
    assert record.result.offset_v == fixed["total_offset_v"]


def test_a02_missing_initial_sample_is_an_error_without_fallback() -> None:
    settings = AnalysisSettings(geometry=PhysicalInputs(particle_radius_m=RADIUS))
    segments, samples = events_dataset(log_theta=-1.0)
    first_active = segments[1].first_row
    samples = [sample for sample in samples if sample.source_row != first_active]
    data, outcome = analyse(segments, samples, (SPHERE, DELTA), settings)
    failed = outcome.snapshot.current(SPHERE.identity(data[0].event.event_id))
    assert failed is not None and failed.state is ResultState.ERROR
    assert failed.error is not None and failed.error.startswith("delacourt_ir_sample_missing")
    delta = outcome.snapshot.current(DELTA.identity(data[0].event.event_id))
    assert delta is not None and delta.state is ResultState.PROVISIONAL  # IR none: unaffected


def test_a02_free_offset_never_computes_or_subtracts_an_ir_constant() -> None:
    settings = AnalysisSettings(geometry=PhysicalInputs(particle_radius_m=RADIUS))
    free = delacourt_variant(DelacourtOffsetMode.FREE)
    segments, samples = events_dataset(log_theta=-1.0, jump=0.02)
    data, outcome = analyse(segments, samples, (free,), settings)
    record = outcome.snapshot.current(free.identity(data[0].event.event_id))
    assert record is not None and isinstance(record.details, dict)
    assert "delacourt_ir" not in record.details and "fixed_offset" not in record.details
    assert isinstance(record.result, DelacourtResult)
    assert record.result.offset_v == pytest.approx(0.02, rel=1e-6, abs=0)
    assert record.diffusion_m2_s == pytest.approx(0.1 * RADIUS**2 / 100, rel=1e-6, abs=0)


def test_a02_other_methods_keep_their_ir_defaults_and_results() -> None:
    geometry = PhysicalInputs(1e-9, 1e-4, RADIUS)
    segments, samples = events_dataset()
    base = AnalysisSettings(geometry=geometry)
    changed = replace(base, delacourt_ir_correction=IrCorrection.NONE)
    data, first = analyse(segments, samples, (VA, DELTA, ICI), base)
    _, second = analyse(segments, samples, (VA, DELTA, ICI), changed)
    for a, b in zip(first.records, second.records, strict=True):
        assert a.diffusion_m2_s == b.diffusion_m2_s
    delta = first.snapshot.current(DELTA.identity(data[0].event.event_id))
    assert delta is not None and isinstance(delta.details, dict)
    assert delta.details["ir"] == {
        "mode": "none",
        "reference_row": data[0].rest_before[-1].source_row,
        "offset_v": 0.0,
    }
    # Changing only the Delacourt IR source invalidates only fixed-offset Delacourt records.
    _, mixed = analyse(segments, samples, (VA, DELTA, SPHERE), base)
    stale = mixed.snapshot.invalidate(changed.dependency_fingerprints())
    assert {r.identity.method for r in stale.records if r.stale} == {
        DiffusionMethod.GITT_SPHERE_FULL
    }


# --- windows ------------------------------------------------------------------------------


def bent_rows() -> tuple[CanonicalSample, ...]:
    return tuple(
        CanonicalSample(
            i, float(i), 1.0, 3 + 0.01 * math.sqrt(i) + (0.0 if i <= 40 else 1e-3 * (i - 40)),
            1, 1, 0.0, 0.0,
        )
        for i in range(1, 121)
    )  # fmt: skip


def test_a03_default_ranking_is_unchanged_but_flags_excluded_early_data() -> None:
    selection = select_window(
        bent_rows(),
        reference_time_s=0.0,
        protocol=Protocol.GITT,
        allowed=WindowBounds(1, 120),
        mode=WindowMode.AUTO_STATISTICAL,
    )
    assert selection.selected.start_s > 1  # LONGEST rule: longest window, outside the sqrt regime
    assert selection.early_data_excluded and selection.quality.needs_review
    anchored = select_window(
        bent_rows(),
        reference_time_s=0.0,
        protocol=Protocol.GITT,
        allowed=WindowBounds(1, 120),
        mode=WindowMode.AUTO_STATISTICAL,
        ranking=WindowRanking.EARLY_ANCHORED,
    )
    assert anchored.selected.start_s == 1 and not anchored.early_data_excluded
    assert anchored.fit.slope_v_per_sqrt_s == pytest.approx(0.01, rel=0.05, abs=0)
    assert abs(selection.fit.slope_v_per_sqrt_s / 0.01 - 1) > 1  # documents the LONGEST-rule bias


def test_a03_manual_and_fixed_windows_are_never_flagged() -> None:
    for mode, manual in ((WindowMode.MANUAL, WindowBounds(30, 90)), (WindowMode.AUTO_FIXED, None)):
        selection = select_window(
            bent_rows(),
            reference_time_s=0.0,
            protocol=Protocol.GITT,
            allowed=WindowBounds(1, 120),
            mode=mode,
            manual=manual,
        )
        assert not selection.early_data_excluded


def test_a03_leading_fraction_proposal_is_explicit_and_uses_real_times() -> None:
    rows = bent_rows()
    full = suggest_initial_window(rows, reference_time_s=0.0, protocol=Protocol.GITT)
    assert full == WindowBounds(1, 120)
    lead = suggest_initial_window(
        rows,
        reference_time_s=0.0,
        protocol=Protocol.GITT,
        rule=InitialWindowRule.LEADING_FRACTION,
        fraction=0.25,
    )
    assert lead == WindowBounds(1, 30)  # 1 + 0.25*119 = 30.75 -> last real time 30
    with pytest.raises(ValueError):
        suggest_initial_window(
            rows,
            reference_time_s=0.0,
            protocol=Protocol.GITT,
            rule=InitialWindowRule.LEADING_FRACTION,
        )
    with pytest.raises(ValueError):
        suggest_initial_window(rows, reference_time_s=0.0, protocol=Protocol.GITT, fraction=0.5)
    assert AnalysisSettings().window_ranking is WindowRanking.LONGEST
    assert AnalysisSettings().initial_window_rule is InitialWindowRule.FULL_SEGMENT


# --- canonical adapter, end to end -------------------------------------------------------


def gitt_csv(path: Path) -> None:
    lines = ["time_s,current_a,voltage_v"]
    now, ocv = 0, 3.5
    for _ in range(60):
        now += 1
        lines.append(f"{now},0,{ocv}")
    for _ in range(3):
        for k in range(1, 101):
            now += 1
            lines.append(f"{now},0.001,{ocv + 0.005 + 0.004 * math.sqrt(k):.15f}")
        ocv += 0.01
        intercept = ocv + 0.003 * math.sqrt(60)
        for k in range(1, 61):
            now += 1
            lines.append(f"{now},0,{intercept - 0.003 * math.sqrt(k):.15f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_a04_canonical_pipeline_end_to_end(tmp_path: Path) -> None:
    gitt_csv(tmp_path / "gitt.csv")
    store = tmp_path / "store"
    imported = import_source(ImportRequest(tmp_path / "gitt.csv", store, "s", "j1", "g1"))
    elaborated = elaborate(
        ElaborationRequest(
            imported.ref.path / "points" / "canonical.parquet",
            imported.report.content_hash,
            store,
            "s",
            "j2",
            "g2",
        )
    )
    scope = load_canonical_scope(
        elaborated.ref.path, half_cycles=[1], elaboration_fingerprint=elaborated.fingerprint
    )
    events = scope.events(Protocol.GITT)
    assert len(events) == 3 and all(event.constant_current for event in events)
    assert events[0].rest_before_segment_id not in scope.scope_segment_ids  # context rest
    data = build_event_data(events, scope.segments, scope.samples)
    settings = AnalysisSettings(geometry=PhysicalInputs(1e-9, 1e-4))
    plan = plan_analysis(data, variants=(VA, DELTA), settings=settings)
    outcome = run_batch(plan.tasks, previous=ResultSnapshot())
    expected_va = 4 / math.pi * 1e-10 * (0.01 / (100 * 0.004)) ** 2
    expected_delta = 4 / (math.pi * 100) * 1e-10 * (0.01 / (0.005 + 0.004 * 10)) ** 2
    for item in data:
        va = outcome.snapshot.current(VA.identity(item.event.event_id))
        delta = outcome.snapshot.current(DELTA.identity(item.event.event_id))
        assert va is not None and delta is not None
        assert va.diffusion_m2_s == pytest.approx(expected_va, rel=1e-7, abs=0)
        assert delta.diffusion_m2_s == pytest.approx(expected_delta, rel=1e-9, abs=0)
    with pytest.raises(ValueError, match="fingerprint"):
        load_canonical_scope(elaborated.ref.path, elaboration_fingerprint="0" * 64)
    with pytest.raises(ValueError, match="elaboration"):
        load_canonical_scope(imported.ref.path)
    manifest = json.loads((elaborated.ref.path / "manifest.json").read_text(encoding="utf-8"))
    assert scope.elaboration_fingerprint == manifest["content_hash"]


def test_a04_scope_restricts_points_and_keeps_context(tmp_path: Path) -> None:
    gitt_csv(tmp_path / "gitt.csv")
    store = tmp_path / "store"
    imported = import_source(ImportRequest(tmp_path / "gitt.csv", store, "s", "j1", "g1"))
    elaborated = elaborate(
        ElaborationRequest(
            imported.ref.path / "points" / "canonical.parquet",
            imported.report.content_hash,
            store,
            "s",
            "j2",
            "g2",
        )
    )
    empty = load_canonical_scope(elaborated.ref.path, cycles=[99])
    assert empty.segments == () and empty.samples == () and empty.events(Protocol.GITT) == ()
    full = load_canonical_scope(elaborated.ref.path)
    rows = [sample.source_row for sample in full.samples]
    assert rows == sorted(rows) and len(rows) == len(set(rows))
    for segment in full.segments:
        inside = [s for s in full.samples if segment.first_row <= s.source_row <= segment.last_row]
        assert len(inside) == segment.last_row - segment.first_row + 1


# --- radius-only Delacourt, non-CC events, resistances ------------------------


def test_a09_delacourt_runs_with_radius_only_and_va_reports_missing_inputs() -> None:
    radius_only = physical_inputs_si(particle_radius=1.0, particle_radius_unit=LengthUnit.UM)
    assert radius_only.active_volume_m3 is None and radius_only.particle_radius_m == 1e-6
    settings = AnalysisSettings(geometry=radius_only, delacourt_ir_correction=IrCorrection.NONE)
    segments, samples = events_dataset(log_theta=-1.0)
    data, outcome = analyse(segments, samples, (SPHERE, VA), settings)
    sphere = outcome.snapshot.current(SPHERE.identity(data[0].event.event_id))
    assert sphere is not None and sphere.diffusion_m2_s == pytest.approx(
        0.1 * RADIUS**2 / 100, rel=1e-6, abs=0
    )
    va = outcome.snapshot.current(VA.identity(data[0].event.event_id))
    assert va is not None and isinstance(va.result, DiffusionResult)
    assert va.result.unavailable_reason == "physical_input_missing"
    assert va.result.response_v_per_sqrt_s is not None and va.result.response_v_per_sqrt_s > 0
    assert va.result.delta_u_v == pytest.approx(0.01, rel=1e-9, abs=0)
    with pytest.raises(ValueError):
        PhysicalInputs(active_volume_m3=-1.0)


def test_a12_non_cc_active_segment_keeps_intermediates_without_d() -> None:
    segments, samples = events_dataset(pulse_step=Step.OTHER)
    events = associate_events(segments, protocol=Protocol.GITT, elaboration_fingerprint=FINGERPRINT)
    assert events and not any(event.constant_current for event in events)
    settings = AnalysisSettings(geometry=PhysicalInputs(1e-9, 1e-4, RADIUS))
    data, outcome = analyse(segments, samples, (VA, DELTA, ICI, SPHERE), settings)
    event_id = data[0].event.event_id
    for variant in (VA, DELTA, ICI):
        record = outcome.snapshot.current(variant.identity(event_id))
        assert record is not None and record.state is ResultState.PROVISIONAL
        assert isinstance(record.result, DiffusionResult)
        assert record.result.diffusion_m2_s is None
        assert record.result.unavailable_reason == "constant_current_required"
        assert record.result.delta_u_v == pytest.approx(0.01, rel=1e-9, abs=0)
    sphere = outcome.snapshot.current(SPHERE.identity(event_id))
    assert sphere is not None and sphere.state is ResultState.ERROR
    assert sphere.error is not None and sphere.error.startswith("constant_current_required")


def test_a12_cc_event_identities_are_unchanged_by_non_cc_neighbours() -> None:
    cc_segments, _ = events_dataset(pulses=2)
    mixed = list(cc_segments)
    mixed[3] = replace(mixed[3], step=int(Step.CV))
    cc_ids = associate_events(
        cc_segments, protocol=Protocol.GITT, elaboration_fingerprint=FINGERPRINT
    )
    mixed_ids = associate_events(mixed, protocol=Protocol.GITT, elaboration_fingerprint=FINGERPRINT)
    assert cc_ids[0].event_id == mixed_ids[0].event_id and mixed_ids[0].constant_current
    assert not mixed_ids[1].constant_current


def test_a14_resistances_are_reported_with_signed_differences() -> None:
    segments, samples = events_dataset(jump=0.02)
    settings = AnalysisSettings(geometry=PhysicalInputs(1e-9, 1e-4))
    data, outcome = analyse(segments, samples, (VA,), settings)
    record = outcome.snapshot.current(VA.identity(data[0].event.event_id))
    assert record is not None and isinstance(record.details, dict)
    assert "resistances" not in record.details  # single source: measure_ir_drops
    drops = gregorithm.measure_ir_drops(data[0], settings)
    r_on, r_off = drops.pre_pulse, drops.post_pulse
    expected_dv = 0.005 + 0.02 + 0.004  # E(1 s) - V0 of the generating pulse
    assert r_on.delta_voltage_v == pytest.approx(expected_dv, rel=1e-9, abs=0)
    assert r_on.delta_current_a == pytest.approx(1e-3, rel=1e-12, abs=0)
    assert r_on.resistance_ohm == pytest.approx(expected_dv / 1e-3, rel=1e-9, abs=0)
    assert r_off.delta_current_a == pytest.approx(-1e-3, rel=1e-12, abs=0)


# --- compact public API --------------------------------------------------------------------


def test_a13_root_exports_compact_api_and_keeps_core_names() -> None:
    names = set(gregorithm.__all__)
    assert {
        "load_canonical_scope",
        "plan_analysis",
        "run_batch",
        "commit_batch",
        "publish_snapshot",
        "load_snapshot",
        "export_accepted_results",
    } <= names
    assert {"gitt_va", "fit_sqrt_time", "spherical_f", "select_window"}.isdisjoint(names)
    assert gregorithm.WindowRequest.__module__ == "gregorithm.elaboration"
    assert {"import_source", "elaborate", "detect_dcir", "analyze_dqdv", "export_table"} <= names

    def module_of(name: str) -> str:
        return str(getattr(getattr(gregorithm, name), "__module__", ""))

    gitt_names = {name for name in names if module_of(name).startswith("gregorithm.gitt_ici")}
    assert len(gitt_names) == 65
