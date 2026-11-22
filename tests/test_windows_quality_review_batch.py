from __future__ import annotations

import math
from pathlib import Path

import pytest

from gregorithm.gitt_ici import (
    AnalysisRecord,
    AppliedSettings,
    BatchTask,
    CacheWrite,
    CanonicalSample,
    ComputeOutput,
    DelacourtResult,
    DiffusionMethod,
    DiffusionResult,
    IndicatorColor,
    IntermediateCache,
    OcvSource,
    OcvTargetPlan,
    Protocol,
    QualityAssessment,
    RecoverableAnalysisError,
    ResultIdentity,
    ResultSnapshot,
    ResultState,
    WindowBounds,
    WindowMode,
    cache_key,
    delacourt_quality,
    gitt_delta_quality,
    physical_indicator,
    physical_useful_time,
    run_batch,
    select_window,
    spherical_f,
    suggest_initial_window,
)
from gregorithm.progress import CancellationToken, OperationCancelled, ProgressEvent


def samples(count: int = 20) -> tuple[CanonicalSample, ...]:
    return tuple(
        CanonicalSample(
            index,
            float(index),
            1.0,
            3.0 + 0.02 * math.sqrt(index) + 1e-5 * math.sin(index),
            1,
            1,
            index / 1000,
            0.0,
        )
        for index in range(1, count + 1)
    )


def identity(
    event: str,
    method: DiffusionMethod = DiffusionMethod.GITT_VA,
    variant: str = "default",
    source: OcvSource = OcvSource.REST_SAMPLE,
) -> ResultIdentity:
    return ResultIdentity(event, method, variant, source)


def diffusion(
    method: DiffusionMethod = DiffusionMethod.GITT_VA, value: float = 1e-14
) -> DiffusionResult:
    return DiffusionResult(method, value, None, 0.1, 10.0, 0.02, None, None)


def record(
    result_identity: ResultIdentity,
    *,
    state: ResultState = ResultState.PROVISIONAL,
    value: float = 1e-14,
) -> AnalysisRecord:
    return AnalysisRecord(
        result_identity,
        f"fingerprint-{value}",
        state,
        diffusion(result_identity.method, value),
        None,
    )


def test_initial_and_statistical_windows_stay_inside_real_segment() -> None:
    rows = samples()
    suggested = suggest_initial_window(rows, reference_time_s=0, protocol=Protocol.GITT)
    assert suggested == WindowBounds(1.0, 20.0)
    selected = select_window(
        rows,
        reference_time_s=0,
        protocol=Protocol.GITT,
        allowed=WindowBounds(2.0, 18.0),
        mode=WindowMode.AUTO_STATISTICAL,
    )
    assert 2 <= selected.selected.start_s < selected.selected.end_s <= 18
    assert len(set(selected.fit.relative_times_s)) >= 5
    assert selected.candidate_count > 1


def test_first_automatic_pass_does_not_require_preliminary_d() -> None:
    selected = select_window(
        samples(),
        reference_time_s=0,
        protocol=Protocol.ICI,
        allowed=WindowBounds(1, 20),
        mode=WindowMode.AUTO_STATISTICAL,
    )
    assert selected.useful_time_s is None
    assert selected.quality.physical is IndicatorColor.NOT_AVAILABLE


def test_physical_guidance_requires_inputs_and_clamps_search() -> None:
    rows = samples(30)
    with pytest.raises(ValueError, match="requires radius"):
        select_window(
            rows,
            reference_time_s=0,
            protocol=Protocol.GITT,
            allowed=WindowBounds(1, 30),
            mode=WindowMode.PHYSICALLY_GUIDED,
        )
    selected = select_window(
        rows,
        reference_time_s=0,
        protocol=Protocol.GITT,
        allowed=WindowBounds(1, 30),
        mode=WindowMode.PHYSICALLY_GUIDED,
        particle_radius_m=1e-5,
        preliminary_diffusion_m2_s=1e-12,
        physical_coefficient=0.1,
    )
    assert selected.useful_time_s == pytest.approx(10.0, rel=1e-12, abs=0)
    assert selected.selected.end_s <= 10.0


def test_ici_physical_guidance_is_not_validated_and_never_reuses_gitt_rule() -> None:
    with pytest.raises(ValueError, match="not validated"):
        physical_useful_time(
            protocol=Protocol.ICI,
            particle_radius_m=1e-5,
            diffusion_m2_s=1e-12,
            coefficient=0.1,
        )


def test_manual_window_may_have_fewer_than_five_distinct_times() -> None:
    selected = select_window(
        samples(6),
        reference_time_s=0,
        protocol=Protocol.GITT,
        allowed=WindowBounds(1, 6),
        mode=WindowMode.MANUAL,
        manual=WindowBounds(1, 2),
    )
    assert len(selected.fit.source_rows) == 2
    assert selected.quality.statistical is IndicatorColor.NOT_AVAILABLE


def test_physical_indicator_and_delta_quality_are_non_blocking_contracts() -> None:
    assert physical_indicator(5, 10) is IndicatorColor.GREEN
    assert physical_indicator(12, 10) is IndicatorColor.ORANGE
    assert physical_indicator(16, 10) is IndicatorColor.RED
    assert physical_indicator(5, None) is IndicatorColor.NOT_AVAILABLE
    consistent = DiffusionResult(
        DiffusionMethod.GITT_DELTA, 1e-14, None, 0.01, 10.0, None, 0.02, None
    )
    delta = gitt_delta_quality(consistent, current_sign=1)
    assert delta.statistical is IndicatorColor.NOT_APPLICABLE
    assert delta.physical is IndicatorColor.NOT_APPLICABLE
    assert delta.overall is IndicatorColor.GREEN
    assert gitt_delta_quality(consistent, current_sign=-1).overall is IndicatorColor.RED


def test_delacourt_overall_is_worst_of_fit_and_identifiability() -> None:
    result = DelacourtResult(
        1e-14,
        None,
        -1.0,
        0.1,
        100.0,
        0.0,
        (1, 2, 3),
        (1.0, 2.0, 3.0),
        (3.0, 3.1, 3.2),
        (0.0, 0.0, 0.0),
        0.0,
        1.0,
        0.0,
        True,
        False,
        1.0,
        10,
    )
    quality = delacourt_quality(result)
    assert quality.statistical is IndicatorColor.GREEN
    assert quality.physical is IndicatorColor.RED
    assert quality.overall is IndicatorColor.RED


def test_accept_all_is_scoped_and_revoke_is_reversible() -> None:
    va = record(identity("e1"))
    ici = record(identity("e1", DiffusionMethod.ICI, source=OcvSource.ICI_INTERCEPT))
    snapshot = ResultSnapshot((va, ici)).accept_all(
        method=DiffusionMethod.GITT_VA,
        variant="default",
        ocv_source=OcvSource.REST_SAMPLE,
    )
    assert snapshot.current(va.identity).state is ResultState.ACCEPTED  # type: ignore[union-attr]
    assert snapshot.current(ici.identity).state is ResultState.PROVISIONAL  # type: ignore[union-attr]
    snapshot = snapshot.revoke(va.identity)
    assert snapshot.current(va.identity).state is ResultState.PROVISIONAL  # type: ignore[union-attr]


def test_red_quality_does_not_block_acceptance() -> None:
    target = identity("red")
    red = QualityAssessment(
        DiffusionMethod.GITT_VA,
        IndicatorColor.RED,
        IndicatorColor.RED,
        IndicatorColor.RED,
        99.0,
        99.0,
    )
    snapshot = ResultSnapshot(
        (AnalysisRecord(target, "red-fingerprint", ResultState.PROVISIONAL, diffusion(), red),)
    ).accept(target)
    assert snapshot.current(target).state is ResultState.ACCEPTED  # type: ignore[union-attr]


def test_recalculation_replaces_same_identity_and_resets_acceptance() -> None:
    target = identity("e1")
    snapshot = ResultSnapshot((record(target, state=ResultState.ACCEPTED, value=1e-14),))
    recalculated = record(target, state=ResultState.ACCEPTED, value=2e-14)
    snapshot = snapshot.replace_calculation(recalculated)
    assert len(snapshot.records) == 1
    current = snapshot.current(target)
    assert current is not None and current.state is ResultState.PROVISIONAL
    assert isinstance(current.result, DiffusionResult)
    assert current.result.diffusion_m2_s == 2e-14


def test_d73_error_replaces_old_value_while_other_methods_remain() -> None:
    target = identity("e1")
    other = identity("e1", DiffusionMethod.ICI, source=OcvSource.ICI_INTERCEPT)
    snapshot = ResultSnapshot((record(target, state=ResultState.ACCEPTED), record(other)))
    error = AnalysisRecord(target, "new", ResultState.ERROR, None, None, "fit failed")
    snapshot = snapshot.replace_calculation(error)
    assert snapshot.current(target).state is ResultState.ERROR  # type: ignore[union-attr]
    assert snapshot.current(target).result is None  # type: ignore[union-attr]
    assert snapshot.current(other) is not None


def test_applied_settings_do_not_change_until_explicit_apply() -> None:
    state = AppliedSettings({"active_volume_m3": 1e-6}, {"active_volume_m3": 1e-6})
    edited = state.edit({"active_volume_m3": 2e-6})
    assert edited.applied == {"active_volume_m3": 1e-6}
    applied, changed = edited.apply()
    assert changed and applied.applied == {"active_volume_m3": 2e-6}


def test_ocv_target_override_changes_only_one_rest() -> None:
    plan = OcvTargetPlan(10_800).with_override(7, 3_600)
    assert plan.effective_target_s(7) == 3_600
    assert plan.effective_target_s(8) == 10_800


def test_batch_isolates_recoverable_error_and_reports_progress() -> None:
    events: list[ProgressEvent] = []
    good_id, bad_id = identity("good"), identity("bad")

    def good(_cached):
        return ComputeOutput(diffusion(), cache_value={"slope": 0.02})

    def bad(_cached):
        raise RecoverableAnalysisError("local failure")

    outcome = run_batch(
        (
            BatchTask(good_id, (("generation", "a"),), {"window": 1}, good),
            BatchTask(bad_id, (("generation", "a"),), {"window": 1}, bad),
        ),
        previous=ResultSnapshot(),
        progress=events.append,
    )
    assert outcome.completed == 2 and outcome.recoverable_errors == 1
    assert outcome.snapshot.current(good_id).state is ResultState.PROVISIONAL  # type: ignore[union-attr]
    assert outcome.snapshot.current(bad_id).state is ResultState.ERROR  # type: ignore[union-attr]
    assert events[0].phase == "starting" and events[-1].phase == "complete"
    assert any(event.phase == "calculating:gitt_va" for event in events)
    assert events[-1].completed == events[-1].total == 2 and events[-1].unit == "impulses"


def test_cancelled_batch_returns_no_snapshot_and_commits_no_cache(tmp_path: Path) -> None:
    token = CancellationToken()
    cache = IntermediateCache(tmp_path / "cache")
    previous = ResultSnapshot((record(identity("old"), state=ResultState.ACCEPTED),))

    def cancel_during_first(_cached):
        token.cancel()
        return ComputeOutput(diffusion(), cache_value={"computed": True})

    with pytest.raises(OperationCancelled):
        run_batch(
            (
                BatchTask(
                    identity("new"),
                    (("generation", "a"),),
                    {"window": 1},
                    cancel_during_first,
                ),
            ),
            previous=previous,
            cache=cache,
            cancel=token,
        )
    assert previous.current(identity("old")).state is ResultState.ACCEPTED  # type: ignore[union-attr]
    assert not tuple((tmp_path / "cache").glob("*.json"))


def test_cache_reuses_exact_dependencies_and_invalidates_only_changed(tmp_path: Path) -> None:
    cache = IntermediateCache(tmp_path / "cache")
    dependencies = {"generation": "a", "ocv": "one"}
    key = cache_key("gitt_va", dependencies, {"window": [1, 2]})
    cache.commit((CacheWrite(key, tuple(sorted(dependencies.items())), {"slope": 0.02}),))
    assert cache.get(key, dependencies) == {"slope": 0.02}
    assert cache.get(key, {"generation": "a", "ocv": "two"}) is None
    assert not cache.invalidate_changed({"unrelated": "x"})
    removed = cache.invalidate_changed({"ocv": "two"})
    assert len(removed) == 1 and cache.get(key, dependencies) is None


def test_second_batch_reports_inter_execution_cache_hit(tmp_path: Path) -> None:
    cache = IntermediateCache(tmp_path / "cache")
    target = identity("cached")

    def compute(cached):
        value = {"slope": 0.02} if cached is None else cached
        return ComputeOutput(diffusion(), cache_value=value)

    task = BatchTask(target, (("generation", "a"),), {"window": 1}, compute)
    first = run_batch((task,), previous=ResultSnapshot(), cache=cache)
    second = run_batch((task,), previous=first.snapshot, cache=cache)
    assert first.cache_hits == 0 and second.cache_hits == 1
    assert second.snapshot.current(target).state is ResultState.PROVISIONAL  # type: ignore[union-attr]


def test_spherical_transition_is_value_and_slope_continuous() -> None:
    for boundary in (5e-3, 2e-2):
        epsilon = boundary * 1e-6
        left, center, right = (
            float(spherical_f(boundary - epsilon)),
            float(spherical_f(boundary)),
            float(spherical_f(boundary + epsilon)),
        )
        scale = max(abs(center), 1e-12)
        assert abs(right - left) / scale < 1e-5
        left_slope = (center - left) / epsilon
        right_slope = (right - center) / epsilon
        assert abs(right_slope - left_slope) / max(abs(left_slope), 1e-12) < 2e-3
