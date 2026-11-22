"""Sampled and finite-time extrapolated OCV products for canonical rest segments."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .contracts import CanonicalSample, CanonicalSegment, LogTimeFit, OcvPoint, OcvSource
from .events import samples_for_segment


@dataclass(frozen=True, slots=True)
class OcvSampleTime:
    """Sample time [s] of the sampled OCV of one rest, measured from the relaxation origin.

    The rule is the same as the global ``AnalysisSettings.ocv_sample_time_s``: the last real rest
    sample with ``t <= seconds``; nothing is interpolated. Use it in `OcvSampleSelections` to give
    one rest its own time instead of the global rule.
    """

    seconds: float

    def __post_init__(self) -> None:
        """Require a finite positive time."""
        if not (self.seconds > 0 and math.isfinite(self.seconds)):
            raise ValueError("an OCV sample time must be finite and positive")


@dataclass(frozen=True, slots=True)
class OcvTargetPlan:
    """One global relaxation target with explicit per-rest overrides.

    ``overrides`` replaces the extrapolation target of single rests; ``fit_start_overrides``
    replaces, for single rests, the start of the LOG_TIME_2P fit as a fraction of the observed rest
    (same meaning and range as ``AnalysisSettings.relaxation_fit_start_fraction``). Rests without an
    override use the global values.
    """

    global_target_s: float
    overrides: tuple[tuple[int, float], ...] = ()
    fit_start_overrides: tuple[tuple[int, float], ...] = ()

    def __post_init__(self) -> None:
        """Require a finite positive global target and unique, valid per-rest overrides."""
        if not math.isfinite(self.global_target_s) or self.global_target_s <= 0:
            raise ValueError("global OCV target must be finite and positive")
        rest_ids = [rest_id for rest_id, _ in self.overrides]
        if len(rest_ids) != len(set(rest_ids)):
            raise ValueError("OCV target overrides must have unique rest IDs")
        if any(
            rest_id < 0 or not math.isfinite(target) or target <= 0
            for rest_id, target in self.overrides
        ):
            raise ValueError("OCV target overrides must be positive and have nonnegative IDs")
        fit_ids = [rest_id for rest_id, _ in self.fit_start_overrides]
        if len(fit_ids) != len(set(fit_ids)):
            raise ValueError("OCV fit start overrides must have unique rest IDs")
        if any(
            rest_id < 0 or not (0 <= fraction < 1) for rest_id, fraction in self.fit_start_overrides
        ):
            raise ValueError("OCV fit start overrides must lie in [0, 1) with nonnegative IDs")

    def effective_target_s(self, rest_segment_id: int) -> float:
        """Return the override for ``rest_segment_id`` or the global target."""
        return dict(self.overrides).get(rest_segment_id, self.global_target_s)

    def effective_fit_start(self, rest_segment_id: int, default: float) -> float:
        """Return the fit start fraction of ``rest_segment_id``: its override or ``default``."""
        return dict(self.fit_start_overrides).get(rest_segment_id, default)

    def with_override(self, rest_segment_id: int, target_s: float) -> OcvTargetPlan:
        """Return a new plan with one per-rest target override added or replaced."""
        if rest_segment_id < 0 or not math.isfinite(target_s) or target_s <= 0:
            raise ValueError("OCV target override is invalid")
        values = dict(self.overrides)
        values[rest_segment_id] = target_s
        return OcvTargetPlan(
            self.global_target_s, tuple(sorted(values.items())), self.fit_start_overrides
        )

    def with_fit_start(self, rest_segment_id: int, fraction: float) -> OcvTargetPlan:
        """Return a new plan with one per-rest fit start fraction added or replaced."""
        if rest_segment_id < 0 or not (0 <= fraction < 1):
            raise ValueError("OCV fit start override is invalid")
        values = dict(self.fit_start_overrides)
        values[rest_segment_id] = fraction
        return OcvTargetPlan(self.global_target_s, self.overrides, tuple(sorted(values.items())))


def _capacity(sample: CanonicalSample, polarity: int) -> float:
    """Return the branch capacity of a sample (charge for positive polarity)."""
    return sample.charge_capacity_ah if polarity >= 0 else sample.discharge_capacity_ah


def sampled_ocv(
    rest: CanonicalSegment,
    samples: Sequence[CanonicalSample],
    *,
    state_polarity: int,
    source_row: int | None = None,
    time_s: float | None = None,
    reference_time_s: float | None = None,
) -> OcvPoint | None:
    """Return the last real sample of a rest without averaging or interpolation.

    ``source_row`` selects instead exactly that real sample of the rest (a click keeps the row
    identity); a row outside the rest returns None, never a nearby sample. ``time_s`` selects the
    last real rest sample with ``t - reference_time_s <= time_s`` (``reference_time_s`` defaults to
    the first rest sample); None when no sample lies within that time. ``source_row`` and
    ``time_s`` are mutually exclusive.
    """
    if source_row is not None and time_s is not None:
        raise ValueError("choose either a source row or a sample time, not both")
    rows = samples_for_segment(samples, rest)
    if not rows:
        return None
    if time_s is not None:
        origin = rows[0].time_s if reference_time_s is None else reference_time_s
        within = [row for row in rows if row.time_s - origin <= time_s]
        if not within:
            return None
        sample = within[-1]
    elif source_row is None:
        sample = rows[-1]
    else:
        match = next((row for row in rows if row.source_row == source_row), None)
        if match is None:
            return None
        sample = match
    return OcvPoint(
        OcvSource.REST_SAMPLE,
        sample.voltage_v,
        _capacity(sample, state_polarity),
        rest.segment_id,
        sample.source_row,
    )


def fit_log_time(
    rest: CanonicalSegment,
    samples: Sequence[CanonicalSample],
    *,
    reference_time_s: float,
    fit_start_s: float,
    fit_end_s: float,
    target_time_s: float,
) -> LogTimeFit:
    """Fit LOG_TIME_2P on request; target may be beyond observed data and is never clipped."""
    if not (0 < fit_start_s < fit_end_s and target_time_s > 0):
        raise ValueError("fit bounds and target must be positive and ordered")
    rows = samples_for_segment(samples, rest)
    relative = np.array([row.time_s - reference_time_s for row in rows], dtype=np.float64)
    voltage = np.array([row.voltage_v for row in rows], dtype=np.float64)
    mask = (relative > 0) & (relative >= fit_start_s) & (relative <= fit_end_s)
    x_time = relative[mask]
    y = voltage[mask]
    distinct = np.unique(x_time)
    if len(y) < 2 or len(distinct) < 2:
        raise ValueError("LOG_TIME_2P requires at least two distinct positive times")
    x = np.log(x_time)
    design = np.column_stack((np.ones(len(x)), x))
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != 2 or not np.all(np.isfinite(coefficients)):
        raise ValueError("LOG_TIME_2P design is not identifiable")
    fitted = design @ coefficients
    residual = y - fitted
    rmse = float(np.sqrt(np.mean(residual * residual)))
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = None if total == 0 else 1.0 - float(np.sum(residual * residual)) / total
    alpha, beta = (float(value) for value in coefficients)
    target = alpha + beta * math.log(target_time_s)
    observed = float(np.max(relative[relative > 0]))
    return LogTimeFit(
        alpha,
        beta,
        target_time_s,
        target,
        fit_start_s,
        fit_end_s,
        len(y),
        len(distinct),
        rmse,
        r_squared,
        target_time_s / observed,
    )


def extrapolated_ocv(
    rest: CanonicalSegment,
    samples: Sequence[CanonicalSample],
    fit: LogTimeFit,
    *,
    state_polarity: int,
) -> OcvPoint:
    """Create an extrapolated OCV point while keeping it distinct from sampled OCV."""
    rows = samples_for_segment(samples, rest)
    if not rows:
        raise ValueError("rest segment has no canonical samples")
    return OcvPoint(
        OcvSource.RELAXATION_EXTRAPOLATED,
        fit.target_voltage_v,
        _capacity(rows[-1], state_polarity),
        rest.segment_id,
        None,
        fit.target_time_s,
    )
