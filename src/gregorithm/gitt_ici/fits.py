"""Shared unweighted OLS fit in the square-root-of-time domain."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .contracts import CanonicalSample, SqrtTimeFit


def fit_sqrt_time(
    samples: Sequence[CanonicalSample],
    *,
    reference_time_s: float,
    fit_start_s: float,
    fit_end_s: float,
    voltage_offset_v: float = 0.0,
) -> SqrtTimeFit:
    """Fit ``a + slope*sqrt(t)`` with inclusive bounds, no smoothing and uniform weights."""
    if not all(math.isfinite(value) for value in (reference_time_s, fit_start_s, fit_end_s)):
        raise ValueError("fit times must be finite")
    if fit_start_s < 0 or fit_end_s <= fit_start_s:
        raise ValueError("fit bounds must be nonnegative and ordered")
    selected = tuple(
        sample
        for sample in samples
        if fit_start_s <= sample.time_s - reference_time_s <= fit_end_s
        and sample.time_s > reference_time_s
    )
    times = np.array([sample.time_s - reference_time_s for sample in selected], dtype=float)
    if len(selected) < 2 or len(np.unique(times)) < 2:
        raise ValueError("sqrt-time fit requires at least two distinct positive times")
    voltage = np.array([sample.voltage_v - voltage_offset_v for sample in selected], dtype=float)
    x = np.sqrt(times)
    design = np.column_stack((np.ones(len(x)), x))
    coefficients, _, rank, _ = np.linalg.lstsq(design, voltage, rcond=None)
    if rank != 2 or not np.all(np.isfinite(coefficients)):
        raise ValueError("sqrt-time fit is not identifiable")
    fitted = design @ coefficients
    residuals = voltage - fitted
    sse = float(residuals @ residuals)
    rmse = math.sqrt(sse / len(selected))
    total = float(np.sum((voltage - np.mean(voltage)) ** 2))
    r_squared = None if total == 0 else 1.0 - sse / total
    covariance = None
    if len(selected) > 2:
        sigma_squared = sse / (len(selected) - 2)
        matrix = sigma_squared * np.linalg.inv(design.T @ design)
        if np.all(np.isfinite(matrix)):
            covariance = (
                (float(matrix[0, 0]), float(matrix[0, 1])),
                (float(matrix[1, 0]), float(matrix[1, 1])),
            )
    return SqrtTimeFit(
        float(coefficients[0]),
        float(coefficients[1]),
        fit_start_s,
        fit_end_s,
        float(np.min(times)),
        float(np.max(times)),
        tuple(sample.source_row for sample in selected),
        tuple(float(value) for value in times),
        tuple(float(value) for value in residuals),
        rmse,
        r_squared,
        covariance,
    )
