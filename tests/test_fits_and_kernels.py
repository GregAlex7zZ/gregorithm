from __future__ import annotations

import math

import pytest

from gregorithm.gitt_ici import (
    CanonicalSample,
    IrCorrection,
    OcvPoint,
    OcvSource,
    PhysicalInputs,
    corrected_voltage,
    delta_e_tau,
    fit_sqrt_time,
    gitt_delta,
    gitt_va,
    ici_diffusion,
    ici_intermediates,
    ir_correction_offset,
    measure_resistance,
    ocv_difference,
)


def sample(row: int, time_s: float, current_a: float, voltage_v: float) -> CanonicalSample:
    return CanonicalSample(row, time_s, current_a, voltage_v, 1, 1, 0.0, 0.0)


def exact_fit(*, slope: float = 0.02, intercept: float = 3.0):
    rows = tuple(
        sample(index, time_s, 1.0, intercept + slope * math.sqrt(time_s))
        for index, time_s in enumerate((1.0, 4.0, 9.0, 16.0), 1)
    )
    return fit_sqrt_time(rows, reference_time_s=0.0, fit_start_s=1.0, fit_end_s=16.0)


def test_sqrt_time_ols_recovers_line_and_metadata() -> None:
    fit = exact_fit()
    assert fit.intercept_v == pytest.approx(3.0, rel=1e-12, abs=0)
    assert fit.slope_v_per_sqrt_s == pytest.approx(0.02, rel=1e-12, abs=0)
    assert fit.source_rows == (1, 2, 3, 4)
    assert fit.effective_start_s == 1.0 and fit.effective_end_s == 16.0
    assert fit.rmse_v < 1e-14 and fit.r_squared == pytest.approx(1.0, rel=1e-12, abs=0)
    assert fit.covariance is not None
    assert fit.slope_standard_error is not None


def test_sqrt_fit_correction_changes_only_intercept() -> None:
    raw = exact_fit()
    rows = tuple(
        sample(index, time_s, 1.0, 3.0 + 0.02 * math.sqrt(time_s))
        for index, time_s in enumerate((1.0, 4.0, 9.0, 16.0), 1)
    )
    corrected = fit_sqrt_time(
        rows,
        reference_time_s=0.0,
        fit_start_s=1.0,
        fit_end_s=16.0,
        voltage_offset_v=0.4,
    )
    assert corrected.intercept_v == pytest.approx(raw.intercept_v - 0.4, rel=1e-12, abs=0)
    assert corrected.slope_v_per_sqrt_s == pytest.approx(raw.slope_v_per_sqrt_s, rel=1e-12, abs=0)


def test_covariance_needs_residual_degrees_of_freedom() -> None:
    rows = (sample(1, 1.0, 1.0, 3.0), sample(2, 4.0, 1.0, 3.1))
    fit = fit_sqrt_time(rows, reference_time_s=0, fit_start_s=1, fit_end_s=4)
    assert fit.covariance is None
    assert fit.slope_standard_error is None


def test_resistance_and_ir_modes_preserve_signed_differences() -> None:
    reference = sample(1, 0, 0.1, 3.5)
    measured = sample(2, 1, -0.9, 3.3)
    resistance = measure_resistance(reference, measured)
    assert resistance.delta_voltage_v == pytest.approx(-0.2, rel=1e-12, abs=0)
    assert resistance.delta_current_a == pytest.approx(-1.0, rel=1e-12, abs=0)
    assert resistance.resistance_ohm == pytest.approx(0.2, rel=1e-12, abs=0)
    assert ir_correction_offset(IrCorrection.NONE, reference_voltage_v=3.5) == 0
    assert ir_correction_offset(
        IrCorrection.SELECTED_SAMPLE, reference_voltage_v=3.5, selected_voltage_v=3.3
    ) == pytest.approx(-0.2, rel=1e-12, abs=0)
    assert ir_correction_offset(
        IrCorrection.GITT_INTERCEPT, reference_voltage_v=3.5, gitt_intercept_v=3.25
    ) == pytest.approx(-0.25, rel=1e-12, abs=0)
    assert corrected_voltage(3.2, -0.2) == pytest.approx(3.4, rel=1e-12, abs=0)


def test_zero_current_difference_has_no_resistance() -> None:
    result = measure_resistance(sample(1, 0, 1, 3), sample(2, 1, 1, 4))
    assert result.delta_voltage_v == 1 and result.delta_current_a == 0
    assert result.resistance_ohm is None


def test_gitt_va_matches_analytic_formula() -> None:
    geometry = PhysicalInputs(2e-6, 4e-4)
    fit = exact_fit(slope=0.02)
    result = gitt_va(geometry, delta_u_v=0.1, tau_s=100, pulse_fit=fit, cc_valid=True)
    expected = 4 / math.pi * (2e-6 / 4e-4) ** 2 * (0.1 / (100 * 0.02)) ** 2
    assert result.diffusion_m2_s == pytest.approx(expected, rel=1e-12, abs=0)
    assert result.unavailable_reason is None
    assert result.standard_uncertainty_m2_s is not None


def test_ocv_difference_rejects_mixed_sources() -> None:
    sampled = OcvPoint(OcvSource.REST_SAMPLE, 3.0, 0.0, 1, 1)
    later = OcvPoint(OcvSource.REST_SAMPLE, 3.1, 0.1, 2, 2)
    fitted = OcvPoint(OcvSource.RELAXATION_EXTRAPOLATED, 3.2, 0.1, 2, None, 100)
    assert ocv_difference(sampled, later) == pytest.approx(0.1, rel=1e-12, abs=0)
    with pytest.raises(ValueError, match="must not be mixed"):
        ocv_difference(sampled, fitted)


def test_constant_current_requirement_never_uses_average_current() -> None:
    result = gitt_va(
        PhysicalInputs(1e-6, 1e-4),
        delta_u_v=0.1,
        tau_s=10,
        pulse_fit=exact_fit(),
        cc_valid=False,
    )
    assert result.diffusion_m2_s is None
    assert result.unavailable_reason == "constant_current_required"
    assert result.response_v_per_sqrt_s is not None


def test_gitt_delta_sample_vs_intercept_ratio_is_100_over_81() -> None:
    geometry = PhysicalInputs(1e-6, 1e-4)
    reference_v, c0, slope, tau = 3.0, 0.2, 0.01, 100.0
    pulse_end = reference_v + c0 + slope * math.sqrt(tau)
    sample_offset = c0 + slope * math.sqrt(1.0)
    intercept_offset = c0
    delta_sample = delta_e_tau(
        pulse_end_voltage_v=pulse_end,
        reference_voltage_v=reference_v,
        correction_offset_v=sample_offset,
    )
    delta_intercept = delta_e_tau(
        pulse_end_voltage_v=pulse_end,
        reference_voltage_v=reference_v,
        correction_offset_v=intercept_offset,
    )
    sampled = gitt_delta(
        geometry, delta_u_v=0.05, tau_s=tau, delta_e_tau_v=delta_sample, cc_valid=True
    )
    intercept = gitt_delta(
        geometry, delta_u_v=0.05, tau_s=tau, delta_e_tau_v=delta_intercept, cc_valid=True
    )
    assert sampled.diffusion_m2_s is not None and intercept.diffusion_m2_s is not None
    assert sampled.diffusion_m2_s / intercept.diffusion_m2_s == pytest.approx(
        100 / 81, rel=1e-12, abs=0
    )


def test_gitt_delta_coincides_with_va_under_ideal_relation() -> None:
    geometry = PhysicalInputs(1e-6, 2e-4)
    fit = exact_fit(slope=0.03)
    tau = 100.0
    va = gitt_va(geometry, delta_u_v=0.08, tau_s=tau, pulse_fit=fit, cc_valid=True)
    delta = gitt_delta(
        geometry,
        delta_u_v=0.08,
        tau_s=tau,
        delta_e_tau_v=fit.slope_v_per_sqrt_s * math.sqrt(tau),
        cc_valid=True,
    )
    assert delta.diffusion_m2_s == pytest.approx(va.diffusion_m2_s, rel=1e-12, abs=0)


def test_ici_intermediates_and_diffusion() -> None:
    fit = exact_fit(slope=-0.04, intercept=3.6)
    intermediate = ici_intermediates(fit, pre_voltage_v=3.5, pre_current_a=-2.0)
    assert intermediate.pseudo_ocv_v == pytest.approx(3.6, rel=1e-12, abs=0)
    assert intermediate.k_ohm_sqrt_s == pytest.approx(-0.02, rel=1e-12, abs=0)
    assert intermediate.signed_resistance_ohm == pytest.approx(0.05, rel=1e-12, abs=0)
    result = ici_diffusion(
        PhysicalInputs(2e-6, 4e-4),
        delta_u_v=-0.1,
        current_application_time_s=50,
        rest_fit=fit,
        cc_valid=True,
    )
    expected = 4 / math.pi * (2e-6 / 4e-4) ** 2 * (-0.1 / (50 * -0.04)) ** 2
    assert result.diffusion_m2_s == pytest.approx(expected, rel=1e-12, abs=0)


def test_zero_slope_keeps_fit_but_makes_diffusion_unavailable() -> None:
    result = ici_diffusion(
        PhysicalInputs(1e-6, 1e-4),
        delta_u_v=0.1,
        current_application_time_s=10,
        rest_fit=exact_fit(slope=0.0),
        cc_valid=True,
    )
    assert result.diffusion_m2_s is None
    assert result.unavailable_reason == "zero_or_nonfinite_response"


def test_uncertainty_is_propagated_only_when_available() -> None:
    two_rows = (
        sample(1, 1.0, 1.0, 3.02),
        sample(2, 4.0, 1.0, 3.04),
    )
    fit = fit_sqrt_time(two_rows, reference_time_s=0, fit_start_s=1, fit_end_s=4)
    without = gitt_va(
        PhysicalInputs(1e-6, 1e-4), delta_u_v=0.1, tau_s=10, pulse_fit=fit, cc_valid=True
    )
    assert without.standard_uncertainty_m2_s is None
    with_ocv = gitt_va(
        PhysicalInputs(1e-6, 1e-4),
        delta_u_v=0.1,
        tau_s=10,
        pulse_fit=fit,
        cc_valid=True,
        delta_u_standard_uncertainty_v=0.001,
    )
    assert with_ocv.standard_uncertainty_m2_s is not None
