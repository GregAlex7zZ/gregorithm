"""Pre-pulse / post-pulse IR drops: measurement, manual samples, single source and invalidation.

Expected values come from the generating formulas of `events_dataset` (rest at 3.5 V, pulse
``3.5 + 0.005 + jump + 0.004·√t`` at 1 mA, following rest ``3.51 + 0.003·√60 - 0.003·√t`` at
0 A, one sample per second); nothing is taken from the code under test.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
from test_analysis_regressions import analyse, events_dataset

import gregorithm
from gregorithm.gitt_ici import (
    AnalysisSettings,
    DiffusionMethod,
    IrCorrection,
    IrDropSide,
    IrSampleTime,
    MethodVariant,
    OcvSource,
    PhysicalInputs,
    ResultState,
    delacourt_variant,
    ir_sample_dependencies,
    measure_ir_drops,
    plan_analysis,
    run_batch,
)

DELTA = MethodVariant(DiffusionMethod.GITT_DELTA, OcvSource.REST_SAMPLE)
VA = MethodVariant(DiffusionMethod.GITT_VA, OcvSource.REST_SAMPLE)
SPHERE = delacourt_variant()
PRE, POST = IrDropSide.PRE_PULSE, IrDropSide.POST_PULSE
JUMP = 0.02
GEOMETRY = PhysicalInputs(1e-9, 1e-4, 1e-6)


def _pulse(t: float) -> float:
    return 3.5 + 0.005 + JUMP + 0.004 * math.sqrt(t)


def _rest_after(t: float) -> float:
    return 3.51 + 0.003 * math.sqrt(60) - 0.003 * math.sqrt(t)


def _first_event(settings=None, **kwargs):  # type: ignore[no-untyped-def]
    segments, samples = events_dataset(jump=JUMP)
    data, _ = analyse(segments, samples, (VA,), settings or AnalysisSettings(geometry=GEOMETRY))
    return data[0], measure_ir_drops(data[0], settings or AnalysisSettings(), **kwargs)


def test_both_ir_drops_use_the_documented_real_samples_and_signed_differences() -> None:
    data, drops = _first_event()
    pre, post = drops.pre_pulse, drops.post_pulse
    assert pre.reference.source_row == data.rest_before[-1].source_row == 59
    assert pre.measured.source_row == 60 and pre.measured.relative_time_s == 1.0
    assert pre.delta_voltage_v == pytest.approx(_pulse(1) - 3.5, rel=1e-12, abs=0)
    assert pre.delta_current_a == pytest.approx(1e-3, rel=1e-12, abs=0)
    assert pre.resistance_ohm == pytest.approx((_pulse(1) - 3.5) / 1e-3, rel=1e-9, abs=0)
    assert post.reference.source_row == data.active[-1].source_row == 159
    assert post.measured.source_row == 160 and post.reason is None
    assert post.delta_voltage_v == pytest.approx(_rest_after(1) - _pulse(100), rel=1e-12, abs=0)
    assert post.delta_current_a == pytest.approx(-1e-3, rel=1e-12, abs=0)  # signed
    assert post.resistance_ohm == pytest.approx(
        abs(_rest_after(1) - _pulse(100)) / 1e-3, rel=1e-9, abs=0
    )
    assert not pre.manual and not post.manual


def test_post_pulse_time_is_independent_of_the_pre_pulse_time() -> None:
    _, drops = _first_event(AnalysisSettings(post_pulse_sample_time_s=3.0))
    assert drops.pre_pulse.measured.relative_time_s == 1.0
    assert drops.post_pulse.measured.relative_time_s == 3.0
    assert drops.post_pulse.delta_voltage_v == pytest.approx(
        _rest_after(3) - _pulse(100), rel=1e-12, abs=0
    )
    with pytest.raises(ValueError):
        AnalysisSettings(post_pulse_sample_time_s=0.0)


def test_manual_samples_are_exact_rows_and_invalid_rows_give_no_value() -> None:
    data, _ = _first_event()
    event = data.event.event_id
    _, drops = _first_event(samples={(event, PRE): 64, (event, POST): 165})
    assert drops.pre_pulse.manual and drops.pre_pulse.measured.source_row == 64
    assert drops.pre_pulse.delta_voltage_v == pytest.approx(_pulse(5) - 3.5, rel=1e-12, abs=0)
    assert drops.post_pulse.measured.relative_time_s == 6.0
    _, drops = _first_event(samples={(event, PRE): 165})  # not a pulse row
    assert (
        drops.pre_pulse.reason == "manual_sample_invalid" and drops.pre_pulse.resistance_ohm is None
    )
    assert drops.pre_pulse.reference is not None and drops.pre_pulse.measured is None


def test_missing_rest_and_zero_current_step_are_reported_without_values() -> None:
    data, _ = _first_event()
    settings = AnalysisSettings()
    lone = measure_ir_drops(replace(data, rest_before=(), rest_after=()), settings)
    assert lone.pre_pulse.reason == "rest_before_missing" and lone.pre_pulse.reference is None
    assert lone.post_pulse.reason == "rest_after_missing" and lone.post_pulse.measured is None
    flat = replace(data, rest_after=tuple(replace(s, current_a=1e-3) for s in data.rest_after))
    zero = measure_ir_drops(flat, settings).post_pulse
    assert zero.reason == "zero_current_step" and zero.resistance_ohm is None
    assert zero.delta_voltage_v is not None and zero.delta_current_a == 0.0


def test_ir_correction_uses_the_same_pre_pulse_sample_and_delacourt_default_stays_exclusive() -> (
    None
):
    segments, samples = events_dataset(jump=JUMP)
    settings = AnalysisSettings(geometry=GEOMETRY)
    data, outcome = analyse(segments, samples, (DELTA, SPHERE), settings)
    event = data[0].event.event_id
    delta = outcome.snapshot.current(DELTA.identity(event))
    sphere = outcome.snapshot.current(SPHERE.identity(event))
    assert delta.details["ir"]["mode"] == "none"  # Delacourt only
    drops = measure_ir_drops(data[0], settings)
    assert sphere.details["delacourt_ir"]["sample_row"] == drops.pre_pulse.measured.source_row
    assert sphere.details["delacourt_ir"]["offset_v"] == pytest.approx(
        drops.pre_pulse.delta_voltage_v, rel=1e-12, abs=0
    )
    assert "resistances" not in delta.details and "resistances" not in sphere.details  # no copies
    manual = {(event, PRE): 63}
    sampled = AnalysisSettings(geometry=GEOMETRY, ir_correction=IrCorrection.SELECTED_SAMPLE)
    plan = plan_analysis(data, variants=(DELTA, SPHERE), settings=sampled, ir_samples=manual)
    staged = run_batch(plan.tasks, previous=gregorithm.ResultSnapshot()).snapshot
    drops = measure_ir_drops(data[0], sampled, samples=manual)
    for variant, label in ((DELTA, "ir"), (SPHERE, "delacourt_ir")):
        detail = staged.current(variant.identity(event)).details[label]
        assert detail["sample_row"] == drops.pre_pulse.measured.source_row == 63
        assert detail["manual_sample"] is True


def test_manual_pre_pulse_sample_invalidates_only_its_dependent_series() -> None:
    segments, samples = events_dataset(jump=JUMP)
    settings = AnalysisSettings(geometry=GEOMETRY, ir_correction=IrCorrection.SELECTED_SAMPLE)
    data, outcome = analyse(segments, samples, (VA, DELTA, SPHERE), settings)
    first, second = (item.event.event_id for item in data)
    changed = outcome.snapshot.invalidate(
        ir_sample_dependencies({(first, PRE): 63}, [first, second])
    )
    stale = {(r.identity.event_id, r.identity.method) for r in changed.records if r.stale}
    assert stale == {(first, DiffusionMethod.GITT_DELTA), (first, DiffusionMethod.GITT_SPHERE_FULL)}
    post_only = outcome.snapshot.invalidate(ir_sample_dependencies({(first, POST): 170}, [first]))
    assert not any(r.stale for r in post_only.records)  # R_off feeds no D
    plain = AnalysisSettings(geometry=GEOMETRY)
    tasks = plan_analysis(data, variants=(VA, DELTA), settings=plain).tasks
    assert all(not any(k.startswith("ir_sample:") for k, _ in t.dependencies) for t in tasks)
    later = plan_analysis(
        data,
        variants=(VA, DELTA, SPHERE),
        settings=replace(settings, post_pulse_sample_time_s=5.0),
    ).tasks
    before = plan_analysis(data, variants=(VA, DELTA, SPHERE), settings=settings).tasks
    assert [t.dependencies for t in later] == [t.dependencies for t in before]  # no dependency


def test_invalid_manual_pre_pulse_sample_is_an_error_record_without_fallback() -> None:
    segments, samples = events_dataset(jump=JUMP)
    settings = AnalysisSettings(geometry=GEOMETRY)
    data, _ = analyse(segments, samples, (VA,), settings)
    event = data[0].event.event_id
    plan = plan_analysis(
        data, variants=(SPHERE, VA), settings=settings, ir_samples={(event, PRE): 165}
    )
    snapshot = run_batch(plan.tasks, previous=gregorithm.ResultSnapshot()).snapshot
    failed = snapshot.current(SPHERE.identity(event))
    assert failed.state is ResultState.ERROR and failed.error.startswith(
        "delacourt_ir_sample_invalid"
    )
    assert snapshot.current(VA.identity(event)).state is ResultState.PROVISIONAL


def test_a_local_sample_time_applies_the_same_rule_to_one_side_of_one_impulse() -> None:
    segments, samples = events_dataset(jump=JUMP)
    settings = AnalysisSettings(geometry=GEOMETRY)
    data, outcome = analyse(segments, samples, (SPHERE,), settings)
    first, second = (item.event.event_id for item in data)
    local = {(first, PRE): IrSampleTime(4.0)}
    drops = measure_ir_drops(data[0], settings, samples=local)
    assert drops.pre_pulse.local_time and not drops.pre_pulse.manual
    assert drops.pre_pulse.target_time_s == 4.0 and drops.pre_pulse.measured.relative_time_s == 4.0
    assert drops.pre_pulse.delta_voltage_v == pytest.approx(_pulse(4) - 3.5, rel=1e-12, abs=0)
    assert (
        measure_ir_drops(data[1], settings, samples=local).pre_pulse.measured.relative_time_s == 1.0
    )
    assert measure_ir_drops(data[0], settings, samples=local).post_pulse.target_time_s == 1.0
    plan = plan_analysis(data, variants=(SPHERE,), settings=settings, ir_samples=local)
    snapshot = run_batch(plan.tasks, previous=gregorithm.ResultSnapshot()).snapshot
    ir = snapshot.current(SPHERE.identity(first)).details["delacourt_ir"]
    assert ir["target_time_s"] == 4.0 and ir["local_time"] is True  # the Delacourt IR follows it
    assert ir["sample_row"] == drops.pre_pulse.measured.source_row
    changed = outcome.snapshot.invalidate(ir_sample_dependencies(local, [first, second]))
    assert {r.identity.event_id for r in changed.records if r.stale} == {first}
    with pytest.raises(ValueError):
        IrSampleTime(0.0)
