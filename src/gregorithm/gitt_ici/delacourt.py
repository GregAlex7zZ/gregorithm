"""Analytical spherical GITT model and Delacourt fitting in log10(theta_tau)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from scipy import special  # type: ignore[import-untyped]
from scipy.optimize import brentq, minimize_scalar  # type: ignore[import-untyped]

from .contracts import (
    CanonicalSample,
    DelacourtOffsetMode,
    DelacourtParameters,
    DelacourtResult,
)


@lru_cache(maxsize=8)
def spherical_eigenvalues(terms: int = 20) -> tuple[float, ...]:
    """Return the first positive roots of ``tan(lambda)=lambda``."""
    if terms < 1:
        raise ValueError("terms must be positive")
    roots = []
    epsilon = 1e-12
    for index in range(1, terms + 1):
        lower = index * math.pi + epsilon
        upper = (index + 0.5) * math.pi - epsilon
        roots.append(float(brentq(lambda value: math.tan(value) - value, lower, upper)))
    return tuple(roots)


SPHERICAL_SWITCH_THETA = 2e-2
"""Dimensionless time separating the short-time closed form from the eigenfunction series."""

_MINIMUM_SERIES_TERMS = 14
"""Fewest terms keeping the series truncation below 1e-17 at the switch (λ²θ ≥ 40)."""


def spherical_f(theta: float | np.ndarray, *, terms: int = 20) -> float | np.ndarray:
    """Evaluate the spherical constant-flux response ``f(θ)``, accurate to about 1e-15.

    ``f(θ) = 3θ + 1/5 - 2 Σ exp(-λ_j²θ)/λ_j²`` with ``tan λ_j = λ_j``. For ``θ ≥ 0.02`` the
    eigenfunction series is used with ``terms`` terms (20 as in the cited paper; at least 14 are
    always used so that the truncation is negligible at the switch). For ``θ < 0.02`` the series
    converges too slowly, so the exact short-time solution of the Laplace-domain response
    ``1/(s(√s coth√s - 1))`` is used: replacing ``coth√s`` by one gives
    ``f(θ) = e^θ (1 + erf √θ) - 1``, whose neglected image terms are of order ``exp(-1/θ)``
    (below 1e-20 at the switch). It expands as ``2√(θ/π) + θ + 4θ^{3/2}/(3√π) + ...`` and gives
    ``f(0) = 0`` exactly. Both branches agree to ~3e-16 at the switch, so no blending is needed.
    """
    values = np.asarray(theta, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("theta must be finite and nonnegative")
    flat = values.reshape(-1)
    result = np.empty_like(flat)
    short = flat < SPHERICAL_SWITCH_THETA
    if np.any(short):
        selected = flat[short]
        result[short] = np.expm1(selected) + np.exp(selected) * special.erf(np.sqrt(selected))
    if np.any(~short):
        roots = np.asarray(spherical_eigenvalues(max(terms, _MINIMUM_SERIES_TERMS)))
        squared = roots * roots
        selected = flat[~short]
        transient = np.exp(-np.outer(selected, squared)) / squared
        result[~short] = 3 * selected + 0.2 - 2 * np.sum(transient, axis=1)
    shaped = result.reshape(values.shape)
    if np.isscalar(theta):
        return float(shaped)
    return shaped


def delacourt_voltage(
    relative_time_s: float | np.ndarray,
    *,
    log10_theta_tau: float,
    pulse_duration_s: float,
    initial_ocv_v: float,
    final_ocv_v: float,
    offset_v: float,
    series_terms: int = 20,
) -> float | np.ndarray:
    """Evaluate the analytical spherical voltage model in its dimensionless parametrization."""
    times = np.asarray(relative_time_s, dtype=np.float64)
    if np.any(~np.isfinite(times)) or np.any(times < 0) or pulse_duration_s <= 0:
        raise ValueError("times must be finite/nonnegative and pulse duration positive")
    theta_tau = 10.0**log10_theta_tau
    response = spherical_f(theta_tau * times / pulse_duration_s, terms=series_terms)
    model = (
        initial_ocv_v
        + offset_v
        + (final_ocv_v - initial_ocv_v) * np.asarray(response) / (3 * theta_tau)
    )
    if np.isscalar(relative_time_s):
        return float(model)
    return np.asarray(model, dtype=np.float64)


def _selected_samples(
    samples: Sequence[CanonicalSample], reference_time_s: float, duration_s: float
) -> tuple[CanonicalSample, ...]:
    """Return the pulse samples with 0 < t <= duration, requiring three distinct times."""
    selected = tuple(
        sample for sample in samples if 0 < sample.time_s - reference_time_s <= duration_s
    )
    if len(selected) < 3:
        raise ValueError("Delacourt fit requires at least three pulse samples")
    relative = [sample.time_s - reference_time_s for sample in selected]
    if len(set(relative)) < 3:
        raise ValueError("Delacourt fit requires at least three distinct pulse times")
    return selected


def fit_delacourt(
    samples: Sequence[CanonicalSample],
    *,
    reference_time_s: float,
    parameters: DelacourtParameters,
    grid_points: int = 121,
) -> DelacourtResult:
    """Fit the full CC pulse after a log-grid exploration and bounded local refinements."""
    if grid_points < 9:
        raise ValueError("grid_points must be at least 9")
    selected = _selected_samples(samples, reference_time_s, parameters.pulse_duration_s)
    times = np.array([sample.time_s - reference_time_s for sample in selected], dtype=float)
    measured = np.array([sample.voltage_v for sample in selected], dtype=float)
    evaluations = 0

    def model_and_offset(log_theta: float) -> tuple[np.ndarray, float]:
        """Return the model at ``log_theta`` and its fixed or projected (free) offset."""
        non_offset = np.asarray(
            delacourt_voltage(
                times,
                log10_theta_tau=log_theta,
                pulse_duration_s=parameters.pulse_duration_s,
                initial_ocv_v=parameters.initial_ocv_v,
                final_ocv_v=parameters.final_ocv_v,
                offset_v=0.0,
                series_terms=parameters.series_terms,
            )
        )
        offset = (
            parameters.fixed_offset_v
            if parameters.offset_mode is DelacourtOffsetMode.FIXED
            else float(np.mean(measured - non_offset))
        )
        return non_offset + offset, offset

    def objective(log_theta: float) -> float:
        """Return the sum of squared residuals at ``log_theta`` and count the evaluation."""
        nonlocal evaluations
        evaluations += 1
        predicted, _ = model_and_offset(log_theta)
        residuals = measured - predicted
        return float(residuals @ residuals)

    grid = np.linspace(parameters.log10_theta_min, parameters.log10_theta_max, grid_points)
    scores = np.array([objective(float(value)) for value in grid])
    candidates: list[tuple[float, float]] = []
    for index in range(1, grid_points - 1):
        if scores[index] <= scores[index - 1] and scores[index] <= scores[index + 1]:
            candidates.append((float(grid[index - 1]), float(grid[index + 1])))
    best_grid = int(np.argmin(scores))
    if not candidates:
        lower = float(grid[max(0, best_grid - 1)])
        upper = float(grid[min(grid_points - 1, best_grid + 1)])
        candidates.append((lower, upper))
    refined: list[tuple[float, float]] = []
    for lower, upper in candidates:
        if upper <= lower:
            continue
        result = minimize_scalar(
            objective,
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 1e-11, "maxiter": 500},
        )
        refined.append((float(result.fun), float(result.x)))
    refined.extend(
        (
            (float(scores[0]), parameters.log10_theta_min),
            (float(scores[-1]), parameters.log10_theta_max),
        )
    )
    sse, optimum = min(refined)
    predicted, offset = model_and_offset(optimum)
    residuals = measured - predicted
    rmse = math.sqrt(sse / len(measured))
    total = float(np.sum((measured - np.mean(measured)) ** 2))
    r_squared = None if total == 0 else 1.0 - sse / total
    width = parameters.log10_theta_max - parameters.log10_theta_min
    boundary_tolerance = max(1e-7, width * 1e-6)
    at_boundary = (
        optimum - parameters.log10_theta_min <= boundary_tolerance
        or parameters.log10_theta_max - optimum <= boundary_tolerance
    )

    step = min(1e-4, width / 10_000)
    left = max(parameters.log10_theta_min, optimum - step)
    right = min(parameters.log10_theta_max, optimum + step)
    left_model, _ = model_and_offset(left)
    right_model, _ = model_and_offset(right)
    derivative = (right_model - left_model) / (right - left)
    sensitivity = float(np.linalg.norm(derivative))
    columns = [derivative]
    if parameters.offset_mode is DelacourtOffsetMode.FREE:
        columns.append(np.ones(len(measured)))
    jacobian = np.column_stack(columns)
    rank = int(np.linalg.matrix_rank(jacobian))
    parameter_count = len(columns)
    identifiable = not at_boundary and rank == parameter_count and sensitivity > math.ulp(1.0)
    uncertainty = None
    if identifiable and len(measured) > parameter_count:
        sigma_squared = sse / (len(measured) - parameter_count)
        covariance = sigma_squared * np.linalg.inv(jacobian.T @ jacobian)
        variance_log_theta = float(covariance[0, 0])
        if math.isfinite(variance_log_theta) and variance_log_theta >= 0:
            theta_tau = 10.0**optimum
            diffusion = theta_tau * parameters.particle_radius_m**2 / parameters.pulse_duration_s
            uncertainty = math.log(10) * diffusion * math.sqrt(variance_log_theta)

    theta_tau = 10.0**optimum
    diffusion = theta_tau * parameters.particle_radius_m**2 / parameters.pulse_duration_s
    return DelacourtResult(
        diffusion,
        uncertainty,
        optimum,
        theta_tau,
        parameters.particle_radius_m**2 / diffusion,
        offset,
        tuple(sample.source_row for sample in selected),
        tuple(float(value) for value in times),
        tuple(float(value) for value in predicted),
        tuple(float(value) for value in residuals),
        rmse,
        r_squared,
        sse,
        at_boundary,
        identifiable,
        sensitivity,
        evaluations,
    )
