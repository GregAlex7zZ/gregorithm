"""Rest-level OCV products: sampled and extrapolated kept apart, manual sampled rows, consistency
with the D results and targeted invalidation.

`events_dataset` rests: the first at 3.5 V, then ``ocv' + 0.003*sqrt(60) - 0.003*sqrt(t)`` for
t = 1..60 s, so the last sample of each later rest is exactly ``ocv'``.
"""

from __future__ import annotations

import math

import pytest
from test_analysis_regressions import analyse, events_dataset

import gregorithm
from gregorithm.gitt_ici import (
    AnalysisSettings,
    DiffusionMethod,
    MethodVariant,
    OcvSource,
    OcvTargetPlan,
    PhysicalInputs,
    measure_rest_ocv,
    ocv_sample_dependencies,
    plan_analysis,
    run_batch,
)

GEOMETRY = PhysicalInputs(1e-9, 1e-4)
SAMPLED = MethodVariant(DiffusionMethod.GITT_VA, OcvSource.REST_SAMPLE)
EXTRAPOLATED = MethodVariant(DiffusionMethod.GITT_VA, OcvSource.RELAXATION_EXTRAPOLATED)
DELTA = MethodVariant(DiffusionMethod.GITT_DELTA, OcvSource.REST_SAMPLE)


def _data(settings):  # type: ignore[no-untyped-def]
    segments, samples = events_dataset()
    data, outcome = analyse(segments, samples, (SAMPLED, EXTRAPOLATED, DELTA), settings)
    return data, outcome


def test_every_rest_is_measured_once_with_separate_sampled_and_extrapolated_points() -> None:
    settings = AnalysisSettings(geometry=GEOMETRY, ocv_targets=OcvTargetPlan(600.0))
    data, outcome = _data(settings)
    rests = measure_rest_ocv(data, settings)
    assert [r.rest_segment_id for r in rests] == [0, 2, 4]
    first, second = rests[0], rests[1]
    assert first.sampled.voltage_v == pytest.approx(3.5, rel=1e-12, abs=0)
    assert second.sampled.source_row == second.samples[-1].source_row
    assert second.sampled.voltage_v == pytest.approx(3.51, rel=1e-12, abs=0)
    assert (
        first.extrapolated is None and first.extrapolated_reason == "relaxation_reference_missing"
    )
    assert second.extrapolated is not None and second.sampled is not None  # never replaced
    assert second.extrapolated.source is OcvSource.RELAXATION_EXTRAPOLATED
    first_record = outcome.snapshot.current(EXTRAPOLATED.identity(data[0].event.event_id))
    assert first_record.error.startswith("before_relaxation_reference_missing")  # same reason
    record = outcome.snapshot.current(EXTRAPOLATED.identity(data[1].event.event_id))
    after = record.details["ocv_before"]
    assert after["rest_segment_id"] == 2
    assert second.extrapolated.voltage_v == pytest.approx(after["voltage_v"], rel=1e-12, abs=0)
    assert second.fit.target_time_s == after["target_time_s"] == 600.0
    assert second.fit.beta_v < 0  # -0.003*sqrt(t) decreases with ln(t)
    plain = measure_rest_ocv(data, settings, extrapolate=False)
    assert all(r.extrapolated is None and r.extrapolated_reason == "not_requested" for r in plain)


def test_manual_sampled_ocv_uses_the_exact_row_and_refuses_rows_of_other_segments() -> None:
    settings = AnalysisSettings(geometry=GEOMETRY)
    data, _ = _data(settings)
    rest = measure_rest_ocv(data, settings)[1]
    row = rest.samples[29].source_row  # t = 30 s
    manual = measure_rest_ocv(data, settings, samples={2: row})[1]
    assert manual.manual_sample and manual.sampled.source_row == row
    expected = 3.51 + 0.003 * math.sqrt(60) - 0.003 * math.sqrt(30)
    assert manual.sampled.voltage_v == pytest.approx(expected, rel=1e-12, abs=0)
    wrong = measure_rest_ocv(data, settings, samples={2: data[0].active[3].source_row})[1]
    assert wrong.sampled is None and wrong.sampled_reason == "ocv_sample_invalid"


def test_manual_sampled_ocv_enters_the_d_results_and_invalidates_only_sampled_series() -> None:
    settings = AnalysisSettings(geometry=GEOMETRY, ocv_targets=OcvTargetPlan(600.0))
    data, outcome = _data(settings)
    rest = measure_rest_ocv(data, settings)[1]
    row = rest.samples[29].source_row
    changed = outcome.snapshot.invalidate(ocv_sample_dependencies({2: row}, [0, 2, 4]))
    stale = {
        (r.identity.event_id, r.identity.method, r.identity.ocv_source)
        for r in changed.records
        if r.stale
    }
    first, second = (item.event.event_id for item in data)
    assert stale == {
        (first, DiffusionMethod.GITT_VA, OcvSource.REST_SAMPLE),
        (first, DiffusionMethod.GITT_DELTA, OcvSource.REST_SAMPLE),
        (second, DiffusionMethod.GITT_VA, OcvSource.REST_SAMPLE),  # rest 2 is before pulse 2
        (second, DiffusionMethod.GITT_DELTA, OcvSource.REST_SAMPLE),
    }
    plan = plan_analysis(data, variants=(SAMPLED,), settings=settings, ocv_samples={2: row})
    snapshot = run_batch(plan.tasks, previous=gregorithm.ResultSnapshot()).snapshot
    record = snapshot.current(SAMPLED.identity(first))
    assert record.details["ocv_after"]["source_row"] == row
    assert record.details["ocv_after"]["manual_sample"] is True
    assert record.result.delta_u_v == pytest.approx(
        rest.samples[29].voltage_v - 3.5, rel=1e-12, abs=0
    )
    bad = plan_analysis(data, variants=(SAMPLED,), settings=settings, ocv_samples={2: 0})
    failed = run_batch(bad.tasks, previous=gregorithm.ResultSnapshot()).snapshot.current(
        SAMPLED.identity(first)
    )
    assert failed.error.startswith("after_ocv_sample_invalid")


def test_sampled_ocv_at_a_global_or_local_time_is_the_last_real_sample_within_it() -> None:
    """Regression: the sampled OCV can be read at a chosen time (global or per rest) instead of
    the last sample, without interpolation, and the D results use exactly the same point."""
    settings = AnalysisSettings(geometry=GEOMETRY, ocv_sample_time_s=30.5)
    data, _ = _data(AnalysisSettings(geometry=GEOMETRY))
    rest = measure_rest_ocv(data, settings)[1]
    assert rest.sampled.source_row == rest.samples[29].source_row  # t = 30 s <= 30.5 s
    assert rest.sample_time_s == 30.5 and not rest.manual_sample
    local = measure_rest_ocv(data, settings, samples={2: gregorithm.OcvSampleTime(10.0)})[1]
    assert local.sampled.source_row == local.samples[9].source_row and local.manual_sample
    assert local.sample_time_s == 10.0
    other = measure_rest_ocv(data, settings, samples={2: gregorithm.OcvSampleTime(10.0)})[2]
    assert other.sample_time_s == 30.5  # the other rests keep the global time
    early = measure_rest_ocv(data, settings, samples={2: gregorithm.OcvSampleTime(0.5)})[1]
    assert early.sampled is None and early.sampled_reason == "ocv_no_sample_within_time"
    row = rest.samples[5].source_row
    exact = measure_rest_ocv(data, settings, samples={2: row})[1]
    assert exact.sampled.source_row == row and exact.sample_time_s is None  # a row wins
    plan = plan_analysis(data, variants=(SAMPLED,), settings=settings)
    snapshot = run_batch(plan.tasks, previous=gregorithm.ResultSnapshot()).snapshot
    record = snapshot.current(SAMPLED.identity(data[0].event.event_id))
    assert record.details["ocv_after"]["source_row"] == rest.sampled.source_row
    assert record.details["ocv_after"]["sample_time_s"] == 30.5
    with pytest.raises(ValueError):
        AnalysisSettings(ocv_sample_time_s=0.0)
    with pytest.raises(ValueError):
        gregorithm.OcvSampleTime(-1.0)


def test_sample_time_dependencies_keep_old_fingerprints_and_mark_only_changed_rests() -> None:
    """Rows and the last-sample default keep their fingerprints; a global or local time changes
    exactly the rests it applies to."""
    base = ocv_sample_dependencies({2: 7}, [0, 2])
    assert base == ocv_sample_dependencies({2: 7}, [0, 2], None)
    timed = ocv_sample_dependencies({2: 7}, [0, 2], 30.0)
    assert timed["ocv_sample:2"] == base["ocv_sample:2"]  # an exact row ignores the global time
    assert timed["ocv_sample:0"] != base["ocv_sample:0"]
    local = ocv_sample_dependencies({0: gregorithm.OcvSampleTime(30.0)}, [0], None)
    assert local == ocv_sample_dependencies({}, [0], 30.0)


def test_fit_start_of_one_rest_changes_only_that_rest_and_its_dependency() -> None:
    """Regression: the LOG_TIME_2P fit start can be set for one rest without changing the
    others, and the D results with extrapolated OCV use the same fit."""
    plan = OcvTargetPlan(600.0)
    settings = AnalysisSettings(geometry=GEOMETRY, ocv_targets=plan)
    data, _ = _data(settings)
    before = measure_rest_ocv(data, settings)
    local = AnalysisSettings(geometry=GEOMETRY, ocv_targets=plan.with_fit_start(2, 0.5))
    after = measure_rest_ocv(data, local)
    assert after[1].fit.fit_start_s == pytest.approx(0.5 * after[1].fit.fit_end_s, rel=1e-12, abs=0)
    assert after[1].fit.fit_start_s > before[1].fit.fit_start_s
    assert after[2].fit == before[2].fit  # other rests untouched
    deps_before = gregorithm.ocv_target_dependencies(plan, [2, 4])
    deps_after = gregorithm.ocv_target_dependencies(local.ocv_targets, [2, 4])
    assert deps_after["ocv_target:4"] == deps_before["ocv_target:4"]
    assert deps_after["ocv_target:2"] != deps_before["ocv_target:2"]
    tasks = plan_analysis(data, variants=(EXTRAPOLATED,), settings=local).tasks
    snapshot = run_batch(tasks, previous=gregorithm.ResultSnapshot()).snapshot
    record = snapshot.current(EXTRAPOLATED.identity(data[1].event.event_id))
    assert record.details["ocv_before"]["fit_s"][0] == pytest.approx(
        after[1].fit.fit_start_s, rel=1e-12, abs=0
    )
    assert plan.with_fit_start(2, 0.5).with_override(2, 900.0).fit_start_overrides == ((2, 0.5),)
    with pytest.raises(ValueError):
        plan.with_fit_start(2, 1.0)
