"""Non-blocking statistical, physical and method-specific quality indicators.

Colors are advisory review aids. They never block a fit, a manual
selection, an acceptance or "Accept all", and they are distinct from technical unavailability.
All numeric thresholds in this module are the provisional presets of the specification or
explicitly documented provisional choices; they are configurable per analysis and are not
validated scientific standards.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from .contracts import DelacourtResult, DiffusionMethod, DiffusionResult, SqrtTimeFit

QUALITY_RULE_VERSION = "gitt-ici-quality-v2"
"""Version of the indicator rules, recorded with every assessment for traceability."""


class IndicatorColor(StrEnum):
    """Advisory indicator levels; N/D and N/A are neutral states, never a hidden green or red."""

    GREEN = "green"
    ORANGE = "orange"
    RED = "red"
    NOT_AVAILABLE = "not_available"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    """Analysis-global sqrt-time thresholds, configurable but never acceptance filters.

    Attributes:
        linearity_green_pct, linearity_orange_pct: provisional ``q_L`` limits (2 % / 5 %).
        stability_green_pct, stability_orange_pct: provisional ``q_S`` limits (5 % / 15 %).
        slope_min_t_ratio: a slope with ``|s| <= ratio * SE(s)`` is treated as indistinguishable
            from noise, so the relative metrics are not identifiable. Provisional value 2.
        neighbor_step_fraction: each window endpoint is perturbed by this fraction of the
            window's distinct times (at least one) to build the ``q_S`` neighbourhood. Provisional.
        minimum_neighbors: fewest valid neighbour windows needed to report ``q_S``.
    """

    linearity_green_pct: float = 2.0
    linearity_orange_pct: float = 5.0
    stability_green_pct: float = 5.0
    stability_orange_pct: float = 15.0
    slope_min_t_ratio: float = 2.0
    neighbor_step_fraction: float = 0.1
    minimum_neighbors: int = 2

    def __post_init__(self) -> None:
        """Reject non-finite, negative or unordered thresholds."""
        values = (
            self.linearity_green_pct,
            self.linearity_orange_pct,
            self.stability_green_pct,
            self.stability_orange_pct,
            self.slope_min_t_ratio,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("quality thresholds must be finite and nonnegative")
        if self.linearity_orange_pct < self.linearity_green_pct:
            raise ValueError("linearity thresholds must be ordered")
        if self.stability_orange_pct < self.stability_green_pct:
            raise ValueError("stability thresholds must be ordered")
        if not (0 < self.neighbor_step_fraction < 1):
            raise ValueError("neighbor_step_fraction must lie in (0, 1)")
        if self.minimum_neighbors < 1:
            raise ValueError("minimum_neighbors must be positive")


@dataclass(frozen=True, slots=True)
class DelacourtQualityThresholds:
    """Provisional global Delacourt thresholds; configurable and never acceptance filters.

    ``r_squared_*`` grade the overall fit fidelity; ``residual_autocorrelation_*`` grade the
    structure of the residuals through their lag-1 autocorrelation, which approaches one when the
    model misses a systematic shape of the transient.
    """

    r_squared_green: float = 0.99
    r_squared_orange: float = 0.95
    residual_autocorrelation_green: float = 0.5
    residual_autocorrelation_orange: float = 0.8

    def __post_init__(self) -> None:
        """Require ordered thresholds inside [0, 1]."""
        if not (0 <= self.r_squared_orange <= self.r_squared_green <= 1):
            raise ValueError("Delacourt R-squared thresholds must be ordered within [0, 1]")
        if not (
            0 <= self.residual_autocorrelation_green <= self.residual_autocorrelation_orange <= 1
        ):
            raise ValueError("Delacourt autocorrelation thresholds must be ordered within [0, 1]")


@dataclass(frozen=True, slots=True)
class QualityAssessment:
    """Advisory assessment of one result.

    For GITT_VA/ICI ``statistical`` and ``physical`` are the statistical and physical indicators.
    For Delacourt they hold fit fidelity and identifiability; ``overall`` is the worse of the two.
    For GITT_DELTA both are not applicable and ``overall`` reports the pertinent consistency check.
    ``needs_review`` marks an automatic result that did not reach green+green ("to review");
    it never blocks acceptance. ``metrics`` carries method-specific numbers for the technical panel.
    """

    method: DiffusionMethod
    statistical: IndicatorColor
    physical: IndicatorColor
    overall: IndicatorColor
    q_linearity_pct: float | None
    q_stability_pct: float | None
    reason: str | None = None
    needs_review: bool = False
    metrics: tuple[tuple[str, float], ...] = ()
    rule_version: str = QUALITY_RULE_VERSION


DEFAULT_QUALITY_THRESHOLDS = QualityThresholds()
DEFAULT_DELACOURT_THRESHOLDS = DelacourtQualityThresholds()

_RANK = {
    IndicatorColor.NOT_APPLICABLE: -1,
    IndicatorColor.NOT_AVAILABLE: 0,
    IndicatorColor.GREEN: 1,
    IndicatorColor.ORANGE: 2,
    IndicatorColor.RED: 3,
}


def _color(value: float, green: float, orange: float) -> IndicatorColor:
    """Grade a "smaller is better" metric against inclusive green/orange limits."""
    if value <= green:
        return IndicatorColor.GREEN
    if value <= orange:
        return IndicatorColor.ORANGE
    return IndicatorColor.RED


def worst_color(*colors: IndicatorColor) -> IndicatorColor:
    """Return the most severe color; neutral states (N/D, N/A) rank below green."""
    return max(colors, key=_RANK.__getitem__)


def slope_is_identifiable(fit: SqrtTimeFit, thresholds: QualityThresholds) -> bool:
    """Whether the fitted slope is distinguishable from zero and from its noise.

    A numerically zero slope is never identifiable. When a standard error exists, the slope must
    exceed ``slope_min_t_ratio`` standard errors. With only two distinct times there is no noise
    estimate and the slope is accepted as identifiable; the linearity metric is then unavailable.
    """
    if fit.slope_is_numerically_zero:
        return False
    error = fit.slope_standard_error
    return error is None or abs(fit.slope_v_per_sqrt_s) > thresholds.slope_min_t_ratio * error


def statistical_metrics(
    fit: SqrtTimeFit,
    neighbor_slopes: Sequence[float],
    thresholds: QualityThresholds = DEFAULT_QUALITY_THRESHOLDS,
) -> tuple[float | None, float | None, str | None]:
    """Return ``(q_L %, q_S %, reason)``, the relative sqrt-time fit quality metrics.

    ``q_L = RMSE / (|s| (sqrt(t1) - sqrt(t0)))`` needs residual degrees of freedom, i.e. at least
    three samples. ``q_S`` is the median relative slope change over the neighbour windows obtained
    by slightly perturbing the endpoints. Both are None, with a reason, when the slope is zero or
    indistinguishable from noise, so no false green is produced.
    """
    if not slope_is_identifiable(fit, thresholds):
        return None, None, "slope_not_identifiable"
    reason = None
    q_linearity = None
    span = abs(fit.slope_v_per_sqrt_s) * (
        math.sqrt(fit.effective_end_s) - math.sqrt(fit.effective_start_s)
    )
    if fit.covariance is None or span <= 0:
        reason = "linearity_not_estimable"
    else:
        q_linearity = 100 * fit.rmse_v / span
    finite = [value for value in neighbor_slopes if math.isfinite(value)]
    q_stability = None
    if len(finite) < thresholds.minimum_neighbors:
        reason = reason or "insufficient_neighbors"
    else:
        slope = fit.slope_v_per_sqrt_s
        q_stability = 100 * statistics.median(abs(value - slope) / abs(slope) for value in finite)
    return q_linearity, q_stability, reason


def physical_indicator(selected_end_s: float, useful_time_s: float | None) -> IndicatorColor:
    """Apply the approved green / +50 % orange / red physical-time rule; N/D when unknown."""
    if useful_time_s is None or not math.isfinite(useful_time_s) or useful_time_s <= 0:
        return IndicatorColor.NOT_AVAILABLE
    if selected_end_s <= useful_time_s:
        return IndicatorColor.GREEN
    if selected_end_s <= 1.5 * useful_time_s:
        return IndicatorColor.ORANGE
    return IndicatorColor.RED


def sqrt_fit_quality(
    method: DiffusionMethod,
    fit: SqrtTimeFit,
    *,
    neighbor_slopes: Sequence[float],
    useful_time_s: float | None,
    thresholds: QualityThresholds = DEFAULT_QUALITY_THRESHOLDS,
    automatic: bool = False,
    physical_reason: str | None = None,
) -> QualityAssessment:
    """Assess a GITT_VA or ICI sqrt-time window; colors remain advisory.

    The statistical color is the worse of the ``q_L`` and ``q_S`` colors and is N/D when either
    metric is not identifiable. ``needs_review`` is set for automatic selections whose statistical
    color is not green or whose physical color is orange/red; a physical N/D stays visible in
    ``physical`` and ``reason`` but does not by itself mark the event for review.
    """
    if method not in {DiffusionMethod.GITT_VA, DiffusionMethod.ICI}:
        raise ValueError("sqrt_fit_quality applies only to GITT_VA and ICI")
    q_linearity, q_stability, reason = statistical_metrics(fit, neighbor_slopes, thresholds)
    if q_linearity is None or q_stability is None:
        statistical = IndicatorColor.NOT_AVAILABLE
    else:
        statistical = worst_color(
            _color(q_linearity, thresholds.linearity_green_pct, thresholds.linearity_orange_pct),
            _color(q_stability, thresholds.stability_green_pct, thresholds.stability_orange_pct),
        )
    physical = physical_indicator(fit.effective_end_s, useful_time_s)
    if physical is IndicatorColor.NOT_AVAILABLE:
        reason = reason or physical_reason or "physical_check_unavailable"
    needs_review = automatic and (
        statistical is not IndicatorColor.GREEN
        or physical in {IndicatorColor.ORANGE, IndicatorColor.RED}
    )
    return QualityAssessment(
        method,
        statistical,
        physical,
        worst_color(statistical, physical),
        q_linearity,
        q_stability,
        reason,
        needs_review,
        (("neighbor_count", float(len(neighbor_slopes))),),
    )


def gitt_delta_quality(result: DiffusionResult, *, current_sign: int) -> QualityAssessment:
    """Assess GITT_DELTA with its pertinent checks only; sqrt-time linearity is N/A.

    The check is the sign consistency required by the method: under a current of sign ``s`` both
    the OCV step ``ΔU`` and the corrected response ``ΔE_τ`` move in the direction of ``s``. A zero
    ``ΔU`` yields a zero D, which is orange because it cannot be shown on a log axis.
    """
    if result.method is not DiffusionMethod.GITT_DELTA:
        raise ValueError("gitt_delta_quality applies only to GITT_DELTA results")
    if current_sign not in {-1, 1}:
        raise ValueError("current_sign must be -1 or 1")
    na = IndicatorColor.NOT_APPLICABLE
    if result.diffusion_m2_s is None or result.delta_e_tau_v is None or result.delta_u_v is None:
        reason = result.unavailable_reason or "technical_result_unavailable"
        return QualityAssessment(
            DiffusionMethod.GITT_DELTA, na, na, IndicatorColor.NOT_AVAILABLE, None, None, reason
        )
    detail: str | None
    if result.delta_u_v == 0:
        overall, detail = IndicatorColor.ORANGE, "zero_delta_u"
    elif (
        math.copysign(1, result.delta_u_v)
        == math.copysign(1, result.delta_e_tau_v)
        == float(current_sign)
    ):
        overall, detail = IndicatorColor.GREEN, None
    else:
        overall, detail = IndicatorColor.RED, "sign_inconsistency"
    return QualityAssessment(DiffusionMethod.GITT_DELTA, na, na, overall, None, None, detail)


def residual_autocorrelation(residuals: Sequence[float]) -> float | None:
    """Return the lag-1 autocorrelation of residuals, or None with fewer than three values."""
    if len(residuals) < 3:
        return None
    energy = math.fsum(value * value for value in residuals)
    if energy == 0:
        return 0.0
    return math.fsum(a * b for a, b in zip(residuals[1:], residuals[:-1], strict=True)) / energy


def delacourt_quality(
    result: DelacourtResult,
    thresholds: DelacourtQualityThresholds = DEFAULT_DELACOURT_THRESHOLDS,
) -> QualityAssessment:
    """Return one overall color equal to the worse of fit fidelity and identifiability.

    Fit fidelity is the worse of the R² color and the residual-structure color. Identifiability is
    red at a search boundary or without local identifiability, orange when D is identifiable but
    its statistical uncertainty is not estimable, green otherwise. No sqrt-time linearity is used.
    """
    r2_color = (
        IndicatorColor.NOT_AVAILABLE
        if result.r_squared is None
        else IndicatorColor.GREEN
        if result.r_squared >= thresholds.r_squared_green
        else IndicatorColor.ORANGE
        if result.r_squared >= thresholds.r_squared_orange
        else IndicatorColor.RED
    )
    correlation = residual_autocorrelation(result.residuals_v)
    structure_color = (
        IndicatorColor.NOT_AVAILABLE
        if correlation is None
        else _color(
            max(0.0, correlation),
            thresholds.residual_autocorrelation_green,
            thresholds.residual_autocorrelation_orange,
        )
    )
    fit_color = worst_color(r2_color, structure_color)
    if not result.identifiable or result.at_search_boundary:
        identity_color, reason = IndicatorColor.RED, "not_identifiable"
    elif result.standard_uncertainty_m2_s is None:
        identity_color, reason = IndicatorColor.ORANGE, "uncertainty_not_estimable"
    else:
        identity_color, reason = IndicatorColor.GREEN, None
    metrics = [("rmse_v", result.rmse_v), ("objective_sse_v2", result.objective_sse_v2)]
    if result.r_squared is not None:
        metrics.append(("r_squared", result.r_squared))
    if correlation is not None:
        metrics.append(("residual_lag1_autocorrelation", correlation))
    if result.standard_uncertainty_m2_s is not None and result.diffusion_m2_s > 0:
        metrics.append(
            ("relative_uncertainty", result.standard_uncertainty_m2_s / result.diffusion_m2_s)
        )
    return QualityAssessment(
        DiffusionMethod.GITT_SPHERE_FULL,
        fit_color,
        identity_color,
        worst_color(fit_color, identity_color),
        None,
        None,
        reason,
        False,
        tuple(metrics),
    )
