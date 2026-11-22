from __future__ import annotations

import math

import pytest

from gregorithm.elaboration import Step
from gregorithm.gitt_ici import (
    AreaUnit,
    CanonicalSample,
    CanonicalSegment,
    LengthUnit,
    OcvSource,
    Protocol,
    VolumeUnit,
    associate_events,
    extrapolated_ocv,
    fit_log_time,
    last_sample_at_or_before,
    physical_inputs_si,
    sampled_ocv,
)

FINGERPRINT = "a" * 64


def segment(
    segment_id: int,
    first: int,
    last: int,
    step: Step,
    *,
    half_cycle: int = 1,
    polarity: int = 1,
) -> CanonicalSegment:
    return CanonicalSegment(
        segment_id, first, last, float(first), float(last), 1, half_cycle, int(step), polarity
    )


def sample(row: int, time_s: float, voltage_v: float) -> CanonicalSample:
    return CanonicalSample(row, time_s, 0.0, voltage_v, 1, 1, row / 1000, row / 2000)


def test_protocol_is_explicit_and_event_identity_is_stable() -> None:
    segments = (
        segment(10, 0, 2, Step.REST, half_cycle=0, polarity=0),
        segment(11, 3, 5, Step.CC),
        segment(12, 6, 9, Step.REST, polarity=0),
    )
    gitt = associate_events(segments, protocol=Protocol.GITT, elaboration_fingerprint=FINGERPRINT)
    ici = associate_events(segments, protocol=Protocol.ICI, elaboration_fingerprint=FINGERPRINT)
    assert len(gitt) == len(ici) == 1
    assert gitt[0].protocol is Protocol.GITT
    assert ici[0].protocol is Protocol.ICI
    assert gitt[0].event_id != ici[0].event_id
    assert gitt == associate_events(
        tuple(reversed(segments)), protocol=Protocol.GITT, elaboration_fingerprint=FINGERPRINT
    )


def test_incomplete_gitt_is_not_reinterpreted_but_ici_is_complete() -> None:
    segments = (segment(1, 0, 2, Step.CC), segment(2, 3, 5, Step.REST, polarity=0))
    assert not associate_events(
        segments, protocol=Protocol.GITT, elaboration_fingerprint=FINGERPRINT
    )
    incomplete = associate_events(
        segments,
        protocol=Protocol.GITT,
        elaboration_fingerprint=FINGERPRINT,
        complete_only=False,
    )
    assert len(incomplete) == 1 and not incomplete[0].complete
    assert associate_events(segments, protocol=Protocol.ICI, elaboration_fingerprint=FINGERPRINT)[
        0
    ].complete


def test_sample_selection_uses_last_at_or_before_and_source_row_tie_break() -> None:
    samples = (sample(1, 10.0, 3.0), sample(2, 11.0, 3.1), sample(3, 11.0, 3.2))
    selected = last_sample_at_or_before(
        samples, reference_time_s=10.0, target_time_s=1.0, excluded_source_row=1
    )
    assert selected is not None and selected.source_row == 3
    assert (
        last_sample_at_or_before(
            samples, reference_time_s=10.0, target_time_s=0.0, excluded_source_row=1
        )
        is None
    )


def test_geometry_conversion_to_si_and_validation() -> None:
    converted = physical_inputs_si(
        active_volume=2.0,
        active_volume_unit=VolumeUnit.CM3,
        effective_area=3.0,
        effective_area_unit=AreaUnit.CM2,
        particle_radius=4.0,
        particle_radius_unit=LengthUnit.UM,
    )
    assert converted.active_volume_m3 == pytest.approx(2e-6, rel=1e-12, abs=0)
    assert converted.effective_area_m2 == pytest.approx(3e-4, rel=1e-12, abs=0)
    assert converted.particle_radius_m == pytest.approx(4e-6, rel=1e-12, abs=0)
    with pytest.raises(ValueError):
        physical_inputs_si(
            active_volume=0,
            active_volume_unit=VolumeUnit.M3,
            effective_area=1,
            effective_area_unit=AreaUnit.M2,
        )


def test_sampled_ocv_is_last_real_row_without_averaging() -> None:
    rest = segment(7, 10, 12, Step.REST, polarity=0)
    point = sampled_ocv(
        rest,
        (sample(10, 10, 3.0), sample(11, 11, 3.1), sample(12, 12, 3.5)),
        state_polarity=1,
    )
    assert point is not None
    assert point.source is OcvSource.REST_SAMPLE
    assert point.source_row == 12
    assert point.voltage_v == 3.5
    assert point.capacity_ah == pytest.approx(0.012, rel=1e-12, abs=0)


def test_log_time_fit_is_on_demand_and_allows_far_target() -> None:
    rest = segment(9, 1, 4, Step.REST, polarity=0)
    alpha, beta = 3.2, 0.05
    rows = tuple(
        sample(index, time, alpha + beta * math.log(time))
        for index, time in enumerate((1.0, 2.0, 4.0, 8.0), 1)
    )
    fit = fit_log_time(
        rest,
        rows,
        reference_time_s=0.0,
        fit_start_s=1.0,
        fit_end_s=8.0,
        target_time_s=800.0,
    )
    assert fit.alpha_v == pytest.approx(alpha, rel=1e-12, abs=0)
    assert fit.beta_v == pytest.approx(beta, rel=1e-12, abs=0)
    assert fit.target_voltage_v == pytest.approx(alpha + beta * math.log(800.0), rel=1e-12, abs=0)
    assert fit.extrapolation_ratio == 100.0
    assert fit.rmse_v < 1e-12
    point = extrapolated_ocv(rest, rows, fit, state_polarity=1)
    assert point.source is OcvSource.RELAXATION_EXTRAPOLATED
    assert point.source_row is None and point.target_time_s == 800.0


def test_log_fit_rejects_zero_time_and_non_identifiable_window() -> None:
    rest = segment(9, 1, 2, Step.REST, polarity=0)
    rows = (sample(1, 0.0, 3.0), sample(2, 1.0, 3.1))
    with pytest.raises(ValueError, match="distinct positive"):
        fit_log_time(
            rest,
            rows,
            reference_time_s=0.0,
            fit_start_s=0.5,
            fit_end_s=2.0,
            target_time_s=10.0,
        )
