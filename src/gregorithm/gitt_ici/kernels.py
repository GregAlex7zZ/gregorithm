"""Dimensionally explicit kernels for GITT_VA, GITT_DELTA and ICI."""

from __future__ import annotations

import math

from .contracts import (
    CanonicalSample,
    DiffusionMethod,
    DiffusionResult,
    IciIntermediates,
    IrCorrection,
    OcvPoint,
    PhysicalInputs,
    ResistanceMeasurement,
    SqrtTimeFit,
)


def ocv_difference(before: OcvPoint, after: OcvPoint) -> float:
    """Return a signed OCV difference only for a homogeneous source pair."""
    if before.source is not after.source:
        raise ValueError("OCV sources must not be mixed within one diffusion result")
    return after.voltage_v - before.voltage_v


def measure_resistance(
    reference: CanonicalSample, measured: CanonicalSample
) -> ResistanceMeasurement:
    """Measure R_on/R_off from paired real samples while retaining signed differences."""
    delta_v = measured.voltage_v - reference.voltage_v
    delta_i = measured.current_a - reference.current_a
    resistance = None if delta_i == 0 else abs(delta_v / delta_i)
    return ResistanceMeasurement(delta_v, delta_i, resistance)


def ir_correction_offset(
    mode: IrCorrection,
    *,
    reference_voltage_v: float,
    selected_voltage_v: float | None = None,
    gitt_intercept_v: float | None = None,
) -> float:
    """Return the single constant ``c`` to subtract from a pulse voltage."""
    if mode is IrCorrection.NONE:
        return 0.0
    if mode is IrCorrection.SELECTED_SAMPLE:
        if selected_voltage_v is None:
            raise ValueError("selected sample voltage is required")
        return selected_voltage_v - reference_voltage_v
    if gitt_intercept_v is None:
        raise ValueError("GITT intercept is required")
    return gitt_intercept_v - reference_voltage_v


def corrected_voltage(voltage_v: float, correction_offset_v: float) -> float:
    """Apply a constant IR correction exactly once, without absolute values."""
    return voltage_v - correction_offset_v


def delta_e_tau(
    *, pulse_end_voltage_v: float, reference_voltage_v: float, correction_offset_v: float
) -> float:
    """Return the operational GITT_DELTA response over the full observed pulse duration."""
    return pulse_end_voltage_v - reference_voltage_v - correction_offset_v


def _uncertainty(
    diffusion: float,
    *,
    delta_u: float,
    delta_u_std: float | None,
    response: float,
    response_std: float | None,
) -> float | None:
    """Propagate standard uncertainties of ΔU and the response to D (D ∝ ΔU²/response²)."""
    components: list[float] = []
    if delta_u_std is not None:
        if not math.isfinite(delta_u_std) or delta_u_std < 0:
            raise ValueError("delta_u_std must be finite and nonnegative")
        components.append((2 * delta_u_std / delta_u) ** 2)
    if response_std is not None:
        if not math.isfinite(response_std) or response_std < 0:
            raise ValueError("response_std must be finite and nonnegative")
        components.append((2 * response_std / response) ** 2)
    return None if not components else abs(diffusion) * math.sqrt(sum(components))


def _unavailable(
    method: DiffusionMethod,
    delta_u_v: float,
    tau_s: float,
    reason: str,
    *,
    response: float | None = None,
    delta_e: float | None = None,
) -> DiffusionResult:
    """Return an unavailable diffusion result that keeps its intermediates and reason."""
    return DiffusionResult(method, None, None, delta_u_v, tau_s, response, delta_e, reason)


def gitt_va(
    geometry: PhysicalInputs,
    *,
    delta_u_v: float,
    tau_s: float,
    pulse_fit: SqrtTimeFit,
    cc_valid: bool,
    delta_u_standard_uncertainty_v: float | None = None,
) -> DiffusionResult:
    """Compute GITT_VA only when the authoritative active segment is constant-current."""
    slope = pulse_fit.slope_v_per_sqrt_s
    if not cc_valid:
        return _unavailable(
            DiffusionMethod.GITT_VA, delta_u_v, tau_s, "constant_current_required", response=slope
        )
    ratio = geometry.volume_to_area_m
    if ratio is None:
        return _unavailable(
            DiffusionMethod.GITT_VA, delta_u_v, tau_s, "physical_input_missing", response=slope
        )
    if not math.isfinite(tau_s) or tau_s <= 0:
        return _unavailable(
            DiffusionMethod.GITT_VA, delta_u_v, tau_s, "invalid_pulse_duration", response=slope
        )
    if (
        not all(math.isfinite(value) for value in (delta_u_v, slope))
        or pulse_fit.slope_is_numerically_zero
    ):
        return _unavailable(
            DiffusionMethod.GITT_VA, delta_u_v, tau_s, "zero_or_nonfinite_response", response=slope
        )
    diffusion = 4 / math.pi * ratio**2 * (delta_u_v / (tau_s * slope)) ** 2
    uncertainty = (
        _uncertainty(
            diffusion,
            delta_u=delta_u_v,
            delta_u_std=delta_u_standard_uncertainty_v,
            response=slope,
            response_std=pulse_fit.slope_standard_error,
        )
        if delta_u_v != 0
        else None
    )
    return DiffusionResult(
        DiffusionMethod.GITT_VA, diffusion, uncertainty, delta_u_v, tau_s, slope, None, None
    )


def gitt_delta(
    geometry: PhysicalInputs,
    *,
    delta_u_v: float,
    tau_s: float,
    delta_e_tau_v: float,
    cc_valid: bool,
    delta_u_standard_uncertainty_v: float | None = None,
    delta_e_standard_uncertainty_v: float | None = None,
) -> DiffusionResult:
    """Compute the direct sampled GITT_DELTA estimate without requiring a sqrt-time fit."""
    if not cc_valid:
        return _unavailable(
            DiffusionMethod.GITT_DELTA,
            delta_u_v,
            tau_s,
            "constant_current_required",
            delta_e=delta_e_tau_v,
        )
    ratio = geometry.volume_to_area_m
    if ratio is None:
        return _unavailable(
            DiffusionMethod.GITT_DELTA,
            delta_u_v,
            tau_s,
            "physical_input_missing",
            delta_e=delta_e_tau_v,
        )
    if not math.isfinite(tau_s) or tau_s <= 0:
        return _unavailable(
            DiffusionMethod.GITT_DELTA,
            delta_u_v,
            tau_s,
            "invalid_pulse_duration",
            delta_e=delta_e_tau_v,
        )
    if not all(math.isfinite(value) for value in (delta_u_v, delta_e_tau_v)) or delta_e_tau_v == 0:
        return _unavailable(
            DiffusionMethod.GITT_DELTA,
            delta_u_v,
            tau_s,
            "zero_or_nonfinite_response",
            delta_e=delta_e_tau_v,
        )
    diffusion = 4 / (math.pi * tau_s) * ratio**2 * (delta_u_v / delta_e_tau_v) ** 2
    uncertainty = (
        _uncertainty(
            diffusion,
            delta_u=delta_u_v,
            delta_u_std=delta_u_standard_uncertainty_v,
            response=delta_e_tau_v,
            response_std=delta_e_standard_uncertainty_v,
        )
        if delta_u_v != 0
        else None
    )
    return DiffusionResult(
        DiffusionMethod.GITT_DELTA,
        diffusion,
        uncertainty,
        delta_u_v,
        tau_s,
        None,
        delta_e_tau_v,
        None,
    )


def ici_intermediates(
    rest_fit: SqrtTimeFit, *, pre_voltage_v: float, pre_current_a: float
) -> IciIntermediates:
    """Return pseudo-OCV, k and signed ICI resistance independently from D."""
    if pre_current_a == 0 or not math.isfinite(pre_current_a):
        return IciIntermediates(
            rest_fit.intercept_v, rest_fit.slope_v_per_sqrt_s, pre_current_a, None, None
        )
    return IciIntermediates(
        rest_fit.intercept_v,
        rest_fit.slope_v_per_sqrt_s,
        pre_current_a,
        -rest_fit.slope_v_per_sqrt_s / pre_current_a,
        (pre_voltage_v - rest_fit.intercept_v) / pre_current_a,
    )


def ici_diffusion(
    geometry: PhysicalInputs,
    *,
    delta_u_v: float,
    current_application_time_s: float,
    rest_fit: SqrtTimeFit,
    cc_valid: bool,
    delta_u_standard_uncertainty_v: float | None = None,
) -> DiffusionResult:
    """Compute ICI D using a homogeneous OCV source pair supplied by the caller."""
    slope = rest_fit.slope_v_per_sqrt_s
    if not cc_valid:
        return _unavailable(
            DiffusionMethod.ICI,
            delta_u_v,
            current_application_time_s,
            "constant_current_required",
            response=slope,
        )
    ratio = geometry.volume_to_area_m
    if ratio is None:
        return _unavailable(
            DiffusionMethod.ICI,
            delta_u_v,
            current_application_time_s,
            "physical_input_missing",
            response=slope,
        )
    if not math.isfinite(current_application_time_s) or current_application_time_s <= 0:
        return _unavailable(
            DiffusionMethod.ICI,
            delta_u_v,
            current_application_time_s,
            "invalid_current_application_time",
            response=slope,
        )
    if (
        not all(math.isfinite(value) for value in (delta_u_v, slope))
        or rest_fit.slope_is_numerically_zero
    ):
        return _unavailable(
            DiffusionMethod.ICI,
            delta_u_v,
            current_application_time_s,
            "zero_or_nonfinite_response",
            response=slope,
        )
    diffusion = 4 / math.pi * ratio**2 * (delta_u_v / (current_application_time_s * slope)) ** 2
    uncertainty = (
        _uncertainty(
            diffusion,
            delta_u=delta_u_v,
            delta_u_std=delta_u_standard_uncertainty_v,
            response=slope,
            response_std=rest_fit.slope_standard_error,
        )
        if delta_u_v != 0
        else None
    )
    return DiffusionResult(
        DiffusionMethod.ICI,
        diffusion,
        uncertainty,
        delta_u_v,
        current_application_time_s,
        slope,
        None,
        None,
    )
