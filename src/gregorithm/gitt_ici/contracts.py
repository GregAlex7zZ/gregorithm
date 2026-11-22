"""Typed contracts for GITT/ICI event discovery, physical inputs and OCV products."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class Protocol(StrEnum):
    """Explicit analysis protocol; it is never inferred from segment durations."""

    GITT = "gitt"
    ICI = "ici"


class OcvSource(StrEnum):
    """Mutually distinct OCV products approved by the specification."""

    REST_SAMPLE = "rest_sample"
    ICI_INTERCEPT = "ici_intercept"
    RELAXATION_EXTRAPOLATED = "relaxation_extrapolated"


class VolumeUnit(StrEnum):
    """Units accepted for the active-phase volume V_act."""

    M3 = "m3"
    CM3 = "cm3"
    MM3 = "mm3"


class AreaUnit(StrEnum):
    """Units accepted for the effective diffusion area A_eff."""

    M2 = "m2"
    CM2 = "cm2"


class LengthUnit(StrEnum):
    """Units accepted for the particle radius r_p."""

    M = "m"
    MM = "mm"
    UM = "um"


class IrCorrection(StrEnum):
    """Constant voltage correction applied to one pulse."""

    NONE = "none"
    SELECTED_SAMPLE = "selected_sample"
    GITT_INTERCEPT = "gitt_intercept"


class DiffusionMethod(StrEnum):
    """Diffusion methods that use the semi-infinite (sqrt-time) approximation."""

    GITT_VA = "gitt_va"
    GITT_DELTA = "gitt_delta"
    ICI = "ici"
    GITT_SPHERE_FULL = "gitt_sphere_full"


class DelacourtOffsetMode(StrEnum):
    """Approved treatment of the rapid voltage component in the spherical fit."""

    FIXED = "fixed"
    FREE = "free"


@dataclass(frozen=True, slots=True)
class PhysicalInputs:
    """Global physical inputs converted to SI for one analysis.

    The three values are independent direct operator inputs. ``active_volume_m3`` and
    ``effective_area_m2`` are needed by GITT_VA, GITT_DELTA and ICI; ``particle_radius_m`` is
    required only by the analytical Delacourt fit and is optional for window guidance. Any value
    may be None: methods that need a missing value report D as unavailable instead of failing.
    """

    active_volume_m3: float | None = None
    effective_area_m2: float | None = None
    particle_radius_m: float | None = None

    def __post_init__(self) -> None:
        """Require every supplied value to be finite and positive."""
        for label, value in (
            ("active volume", self.active_volume_m3),
            ("effective area", self.effective_area_m2),
            ("particle radius", self.particle_radius_m),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{label} must be finite and positive when supplied")

    @property
    def volume_to_area_m(self) -> float | None:
        """Return ``V_act/A_eff`` in metres, or None when either input is missing."""
        if self.active_volume_m3 is None or self.effective_area_m2 is None:
            return None
        return self.active_volume_m3 / self.effective_area_m2


@dataclass(frozen=True, slots=True)
class CanonicalSample:
    """A measured row from a canonical elaboration, identified by ``source_row``."""

    source_row: int
    time_s: float
    current_a: float
    voltage_v: float
    cycle: int
    half_cycle: int
    charge_capacity_ah: float
    discharge_capacity_ah: float


@dataclass(frozen=True, slots=True)
class CanonicalSegment:
    """A contiguous segment from ``indexes/segments.parquet``."""

    segment_id: int
    first_row: int
    last_row: int
    first_time_s: float
    last_time_s: float
    cycle: int
    half_cycle: int
    step: int
    polarity: int

    @property
    def duration_s(self) -> float:
        """Return the last-minus-first sample time of the segment in seconds."""
        return self.last_time_s - self.first_time_s


@dataclass(frozen=True, slots=True)
class GittIciEvent:
    """Association built above canonical segments without changing their segmentation.

    ``constant_current`` is False when gregorithm labelled the active segment CV or OTHER: the
    event keeps its computable intermediates, but no method requiring constant current produces D.
    The label is copied from the elaboration, never re-derived from the samples.
    """

    event_id: str
    protocol: Protocol
    elaboration_fingerprint: str
    cycle: int
    half_cycle: int
    polarity: int
    active_segment_id: int
    active_first_row: int
    active_last_row: int
    rest_before_segment_id: int | None
    rest_after_segment_id: int | None
    complete: bool
    constant_current: bool = True


@dataclass(frozen=True, slots=True)
class OcvPoint:
    """One OCV product with explicit source, represented state and provenance."""

    source: OcvSource
    voltage_v: float
    capacity_ah: float
    rest_segment_id: int
    source_row: int | None
    target_time_s: float | None = None


@dataclass(frozen=True, slots=True)
class LogTimeFit:
    """Fit ``alpha + beta*ln(t/1 s)`` over real positive relative times."""

    alpha_v: float
    beta_v: float
    target_time_s: float
    target_voltage_v: float
    fit_start_s: float
    fit_end_s: float
    sample_count: int
    distinct_time_count: int
    rmse_v: float
    r_squared: float | None
    extrapolation_ratio: float


@dataclass(frozen=True, slots=True)
class SqrtTimeFit:
    """OLS fit with free intercept in the ``sqrt(relative time)`` domain."""

    intercept_v: float
    slope_v_per_sqrt_s: float
    requested_start_s: float
    requested_end_s: float
    effective_start_s: float
    effective_end_s: float
    source_rows: tuple[int, ...]
    relative_times_s: tuple[float, ...]
    residuals_v: tuple[float, ...]
    rmse_v: float
    r_squared: float | None
    covariance: tuple[tuple[float, float], tuple[float, float]] | None

    @property
    def slope_standard_error(self) -> float | None:
        """Return the covariance-derived slope standard error when estimable."""
        if self.covariance is None:
            return None
        return math.sqrt(max(0.0, self.covariance[1][1]))

    @property
    def slope_is_numerically_zero(self) -> bool:
        """Whether the fitted slope is indistinguishable from zero at binary64 precision."""
        span = math.sqrt(self.effective_end_s) - math.sqrt(self.effective_start_s)
        voltage_scale = max(1.0, abs(self.intercept_v))
        tolerance = 32 * math.ulp(voltage_scale) / max(span, math.ulp(1.0))
        return abs(self.slope_v_per_sqrt_s) <= tolerance


@dataclass(frozen=True, slots=True)
class ResistanceMeasurement:
    """Signed voltage/current differences and their absolute resistance."""

    delta_voltage_v: float
    delta_current_a: float
    resistance_ohm: float | None


@dataclass(frozen=True, slots=True)
class DiffusionResult:
    """One diffusion result or a technically unavailable result with a stable reason.

    ``delta_u_v`` and ``tau_s`` are None only when the corresponding reference does not exist
    (for example the first ICI interruption without a previous pseudo-OCV); they are never
    replaced by a fabricated zero.
    """

    method: DiffusionMethod
    diffusion_m2_s: float | None
    standard_uncertainty_m2_s: float | None
    delta_u_v: float | None
    tau_s: float | None
    response_v_per_sqrt_s: float | None
    delta_e_tau_v: float | None
    unavailable_reason: str | None


@dataclass(frozen=True, slots=True)
class IciIntermediates:
    """ICI quantities independent of whether a diffusion coefficient is available."""

    pseudo_ocv_v: float
    slope_v_per_sqrt_s: float
    pre_interruption_current_a: float
    k_ohm_sqrt_s: float | None
    signed_resistance_ohm: float | None


@dataclass(frozen=True, slots=True)
class DelacourtParameters:
    """Parameters of the analytical spherical fit."""

    particle_radius_m: float
    pulse_duration_s: float
    initial_ocv_v: float
    final_ocv_v: float
    offset_mode: DelacourtOffsetMode = DelacourtOffsetMode.FIXED
    fixed_offset_v: float = 0.0
    log10_theta_min: float = -6.0
    log10_theta_max: float = 3.0
    series_terms: int = 20

    def __post_init__(self) -> None:
        """Validate finite parameters, positive radius/duration, ordered bounds and terms."""
        numeric = (
            self.particle_radius_m,
            self.pulse_duration_s,
            self.initial_ocv_v,
            self.final_ocv_v,
            self.fixed_offset_v,
            self.log10_theta_min,
            self.log10_theta_max,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("Delacourt parameters must be finite")
        if self.particle_radius_m <= 0 or self.pulse_duration_s <= 0:
            raise ValueError("particle radius and pulse duration must be positive")
        if self.log10_theta_max <= self.log10_theta_min:
            raise ValueError("theta bounds must be ordered")
        if self.series_terms < 1:
            raise ValueError("series_terms must be positive")


@dataclass(frozen=True, slots=True)
class DelacourtResult:
    """Analytical spherical fit result and its identifiability diagnostics."""

    diffusion_m2_s: float
    standard_uncertainty_m2_s: float | None
    log10_theta_tau: float
    theta_tau: float
    diffusion_time_s: float
    offset_v: float
    source_rows: tuple[int, ...]
    relative_times_s: tuple[float, ...]
    fitted_voltage_v: tuple[float, ...]
    residuals_v: tuple[float, ...]
    rmse_v: float
    r_squared: float | None
    objective_sse_v2: float
    at_search_boundary: bool
    identifiable: bool
    sensitivity_norm_v_per_log10_theta: float
    optimizer_evaluations: int
