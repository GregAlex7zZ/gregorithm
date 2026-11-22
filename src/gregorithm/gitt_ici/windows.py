"""Manual, fixed, statistical and physically guided sqrt-time window selection.

Candidate windows are scored with prefix sums over the canonical samples, so every candidate fit
costs O(1) after an O(n) preparation and memory stays bounded by the number of candidate
endpoints. The selected window is always refitted with `fit_sqrt_time`, which is the authoritative
result. Selection never leaves the allowed (suggested or operator-edited) range, and in the
physically guided mode never leaves the useful time.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from .contracts import CanonicalSample, DiffusionMethod, Protocol, SqrtTimeFit
from .fits import fit_sqrt_time
from .quality import (
    DEFAULT_QUALITY_THRESHOLDS,
    IndicatorColor,
    QualityAssessment,
    QualityThresholds,
    physical_indicator,
    sqrt_fit_quality,
    worst_color,
)

WINDOW_RULE_VERSION = "gitt-ici-window-v3"
"""Version of the suggestion, neighbourhood and ranking rules recorded with each selection."""

ICI_GUIDANCE_UNAVAILABLE = "ici_useful_time_not_validated"
"""Stable reason: no validated ICI useful-time function exists, so ICI physics stays N/D."""


class WindowMode(StrEnum):
    """Window selection modes: manual and the three automatic variants."""

    MANUAL = "manual"
    AUTO_FIXED = "auto_fixed"
    AUTO_STATISTICAL = "auto_statistical"
    PHYSICALLY_GUIDED = "physically_guided"


class WindowRanking(StrEnum):
    """Order among statistically adequate (green or orange) candidate windows.

    ``LONGEST`` (default) prefers the longest, then best-sampled window wherever
    it starts; it can select a statistically green window outside the diffusive regime when the
    transient is curved, so such selections are flagged ``early_data_excluded`` for review.
    ``EARLY_ANCHORED`` is an experimental alternative: best color first, then the
    earliest start and, from that start, the longest window. The sqrt-time regime is the early part
    of the segment, but on canonical data this rule changes most selected windows, so
    it is not the default until it has been validated.
    """

    EARLY_ANCHORED = "early_anchored"
    LONGEST = "longest"


class InitialWindowRule(StrEnum):
    """Rules for the suggested initial window; only ``FULL_SEGMENT`` is used by default.

    ``LEADING_FRACTION`` suggests ``[first, first + fraction * span]`` of the protocol segment. It
    is experimental: no fraction has been validated, so it is never applied by default.
    """

    FULL_SEGMENT = "full_segment"
    LEADING_FRACTION = "leading_fraction"


@dataclass(frozen=True, slots=True)
class WindowBounds:
    """Inclusive relative-time bounds in seconds from the method's time origin."""

    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        """Require finite, nonnegative and strictly ordered bounds."""
        if not all(math.isfinite(value) for value in (self.start_s, self.end_s)):
            raise ValueError("window bounds must be finite")
        if self.start_s < 0 or self.end_s <= self.start_s:
            raise ValueError("window bounds must be nonnegative and ordered")


@dataclass(frozen=True, slots=True)
class WindowSelection:
    """A fitted window with its advisory quality and traceability data.

    ``search`` is the range actually searched (the allowed range, clipped to the useful time in
    the guided mode); ``reason`` explains a physical N/D or a guidance that could not be applied.
    ``early_data_excluded`` marks an automatic search whose window starts after the first available
    time of the searched range: the sqrt-time regime is early, so the selection needs review.
    It is advisory and never blocks acceptance.
    """

    mode: WindowMode
    protocol: Protocol
    allowed: WindowBounds
    selected: WindowBounds
    fit: SqrtTimeFit
    quality: QualityAssessment
    useful_time_s: float | None
    candidate_count: int
    search: WindowBounds | None = None
    reason: str | None = None
    rule_version: str = WINDOW_RULE_VERSION
    early_data_excluded: bool = False


@dataclass(frozen=True, slots=True)
class GuidedRefinement:
    """Outcome of the deterministic, bounded physically guided refinement.

    ``history`` lists ``(useful_time_s, diffusion_m2_s)`` used at each iteration. ``converged`` is
    False with a ``reason`` when the window did not stabilize or D became unusable.
    """

    selection: WindowSelection
    diffusion_m2_s: float | None
    iterations: int
    converged: bool
    reason: str | None
    history: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class _Series:
    """Positive-time samples sorted by time with centered prefix sums for O(1) window fits."""

    distinct: np.ndarray
    first: np.ndarray
    last: np.ndarray
    sums: tuple[np.ndarray, ...]
    voltage_scale: float


def _series(samples: Sequence[CanonicalSample], reference_time_s: float) -> _Series:
    """Prepare sorted relative times and centered prefix sums of x=sqrt(t) and voltage."""
    rows = sorted(
        (sample for sample in samples if sample.time_s > reference_time_s),
        key=lambda sample: (sample.time_s, sample.source_row),
    )
    times = np.array([row.time_s - reference_time_s for row in rows], dtype=np.float64)
    voltage = np.array([row.voltage_v for row in rows], dtype=np.float64)
    distinct, first = np.unique(times, return_index=True)
    last = np.append(first[1:] - 1, len(times) - 1) if len(times) else first
    x = np.sqrt(times)
    x_centered = x - (float(np.mean(x)) if len(x) else 0.0)
    y_centered = voltage - (float(np.mean(voltage)) if len(voltage) else 0.0)
    columns = (
        np.ones(len(x)),
        x_centered,
        y_centered,
        x_centered * x_centered,
        x_centered * y_centered,
        y_centered * y_centered,
    )
    sums = tuple(np.concatenate(([0.0], np.cumsum(column))) for column in columns)
    scale = max(1.0, float(np.max(np.abs(voltage)))) if len(voltage) else 1.0
    return _Series(distinct, first, last, sums, scale)


def _range_indices(series: _Series, bounds: WindowBounds) -> tuple[int, int]:
    """Map inclusive time bounds to inclusive distinct-time indices (``hi < lo`` when empty)."""
    lo = int(np.searchsorted(series.distinct, bounds.start_s, side="left"))
    hi = int(np.searchsorted(series.distinct, bounds.end_s, side="right")) - 1
    return lo, hi


def _window_stats(
    series: _Series, start: np.ndarray, end: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return slope, SSE, sample count and Sxx for inclusive distinct-index windows."""
    a = series.first[start]
    b = series.last[end] + 1
    n, sx, sy, sxx, sxy, syy = (column[b] - column[a] for column in series.sums)
    sxx_c = sxx - sx * sx / n
    sxy_c = sxy - sx * sy / n
    syy_c = syy - sy * sy / n
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(sxx_c > 0, sxy_c / sxx_c, np.nan)
    sse = np.maximum(syy_c - slope * sxy_c, 0.0)
    return slope, sse, n, sxx_c


def _neighbor_slopes(
    series: _Series,
    start: np.ndarray,
    end: np.ndarray,
    lo: int,
    hi: int,
    thresholds: QualityThresholds,
) -> np.ndarray:
    """Return a (4, m) array of neighbour slopes; NaN marks a neighbour outside the range."""
    step = np.maximum(1, np.rint(thresholds.neighbor_step_fraction * (end - start + 1))).astype(
        np.int64
    )
    pairs = ((start - step, end), (start + step, end), (start, end - step), (start, end + step))
    result = np.full((4, len(start)), np.nan)
    for row, (left, right) in enumerate(pairs):
        valid = (left >= lo) & (right <= hi) & (right - left >= 2)
        if np.any(valid):
            slope, _, _, _ = _window_stats(series, left[valid], right[valid])
            result[row, valid] = slope
    return result


def _candidate_scores(
    series: _Series,
    start: np.ndarray,
    end: np.ndarray,
    lo: int,
    hi: int,
    thresholds: QualityThresholds,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return statistical color rank, ``max(q_L, q_S)`` and slope for each candidate window."""
    slope, sse, n, sxx_c = _window_stats(series, start, end)
    root_start = np.sqrt(series.distinct[start])
    root_end = np.sqrt(series.distinct[end])
    with np.errstate(divide="ignore", invalid="ignore"):
        standard_error = np.sqrt(sse / np.maximum(n - 2, 1) / sxx_c)
        tolerance = (
            32 * np.spacing(series.voltage_scale) / np.maximum(root_end - root_start, 1e-300)
        )
        identifiable = (np.abs(slope) > tolerance) & (
            np.abs(slope) > thresholds.slope_min_t_ratio * standard_error
        )
        q_linearity = 100 * np.sqrt(sse / n) / (np.abs(slope) * (root_end - root_start))
        neighbors = _neighbor_slopes(series, start, end, lo, hi, thresholds)
        relative = np.abs(neighbors - slope) / np.abs(slope)
    valid_neighbors = np.sum(np.isfinite(relative), axis=0)
    q_stability = np.full(len(start), np.nan)
    enough = valid_neighbors >= thresholds.minimum_neighbors
    if np.any(enough):
        q_stability[enough] = 100 * np.nanmedian(relative[:, enough], axis=0)
    available = identifiable & enough & (n >= 3)

    def grade(values: np.ndarray, green: float, orange: float) -> np.ndarray:
        """Map metric values to 0 (green), 1 (orange) or 2 (red); NaN maps to red."""
        return np.where(values <= green, 0, np.where(values <= orange, 1, 2))

    rank = np.maximum(
        grade(q_linearity, thresholds.linearity_green_pct, thresholds.linearity_orange_pct),
        grade(q_stability, thresholds.stability_green_pct, thresholds.stability_orange_pct),
    )
    rank = np.where(available, rank, 3)
    worst = np.where(available, np.maximum(q_linearity, q_stability), np.inf)
    return rank, worst, slope


def suggest_initial_window(
    samples: Sequence[CanonicalSample],
    *,
    reference_time_s: float,
    protocol: Protocol,
    rule: InitialWindowRule = InitialWindowRule.FULL_SEGMENT,
    fraction: float | None = None,
) -> WindowBounds:
    """Suggest an initial window from the real duration of the protocol segment.

    The caller supplies the CC pulse for GITT (origin: last rest sample) or the following rest for
    ICI (origin: last sample under current). The approved default is the full observed span.
    ``LEADING_FRACTION`` with an explicit ``fraction`` in (0, 1] is a non-approved proposal kept for
    comparison; it ends at the last real time not after ``first + fraction * span``. No row
    percentage or universal duration is used. The suggestion remains editable by the operator.
    """
    _ = protocol
    times = sorted(
        {sample.time_s - reference_time_s for sample in samples if sample.time_s > reference_time_s}
    )
    if len(times) < 2:
        raise ValueError("a suggested window requires two distinct positive times")
    if rule is InitialWindowRule.FULL_SEGMENT:
        if fraction is not None:
            raise ValueError("fraction applies only to LEADING_FRACTION")
        return WindowBounds(times[0], times[-1])
    if fraction is None or not (0 < fraction <= 1):
        raise ValueError("LEADING_FRACTION needs an explicit fraction in (0, 1]")
    limit = times[0] + fraction * (times[-1] - times[0])
    eligible = [value for value in times if value <= limit]
    if len(eligible) < 2:
        raise ValueError("the leading fraction contains fewer than two distinct times")
    return WindowBounds(eligible[0], eligible[-1])


def physical_useful_time(
    *,
    protocol: Protocol,
    particle_radius_m: float,
    diffusion_m2_s: float,
    coefficient: float,
) -> float:
    """Return ``coefficient * r_p² / D`` for GITT; ICI has no validated useful-time function.

    The coefficient is explicit because the specification cites ``0.0032`` only as a theoretical
    reference to validate. For ICI the useful time depends on the current history; no
    protocol-specific function has been validated, so ICI raises instead of reusing the GITT rule.
    """
    if protocol is Protocol.ICI:
        raise ValueError("ICI physical guidance is not validated; the physical check is N/D")
    values = (particle_radius_m, diffusion_m2_s, coefficient)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("radius, diffusion and physical coefficient must be finite and positive")
    return coefficient * particle_radius_m**2 / diffusion_m2_s


def _method(protocol: Protocol) -> DiffusionMethod:
    """Return the sqrt-time method whose window the protocol-specific segment serves."""
    return DiffusionMethod.GITT_VA if protocol is Protocol.GITT else DiffusionMethod.ICI


def evaluate_window(
    samples: Sequence[CanonicalSample],
    *,
    reference_time_s: float,
    protocol: Protocol,
    allowed: WindowBounds,
    selected: WindowBounds,
    mode: WindowMode,
    useful_time_s: float | None = None,
    thresholds: QualityThresholds = DEFAULT_QUALITY_THRESHOLDS,
    physical_reason: str | None = None,
    candidate_count: int = 1,
    search: WindowBounds | None = None,
) -> WindowSelection:
    """Fit ``selected`` and assess it; neighbours are perturbations kept inside ``allowed``.

    Raises ValueError when the window is outside the allowed range or not fittable (fewer than two
    distinct positive times), which is technical unavailability, not a quality color.
    """
    if selected.start_s < allowed.start_s or selected.end_s > allowed.end_s:
        raise ValueError("window lies outside the allowed segment range")
    fit = fit_sqrt_time(
        samples,
        reference_time_s=reference_time_s,
        fit_start_s=selected.start_s,
        fit_end_s=selected.end_s,
    )
    series = _series(samples, reference_time_s)
    lo, hi = _range_indices(series, allowed)
    start, end = _range_indices(series, selected)
    neighbors = _neighbor_slopes(series, np.array([start]), np.array([end]), lo, hi, thresholds)[
        :, 0
    ]
    quality = sqrt_fit_quality(
        _method(protocol),
        fit,
        neighbor_slopes=[float(value) for value in neighbors if math.isfinite(value)],
        useful_time_s=useful_time_s,
        thresholds=thresholds,
        automatic=mode is not WindowMode.MANUAL,
        physical_reason=physical_reason,
    )
    searched = search or allowed
    search_lo, _ = _range_indices(series, searched)
    early_excluded = (
        mode in {WindowMode.AUTO_STATISTICAL, WindowMode.PHYSICALLY_GUIDED}
        and search_lo < len(series.distinct)
        and start > search_lo
    )
    if early_excluded:
        quality = replace(quality, needs_review=True)
    return WindowSelection(
        mode,
        protocol,
        allowed,
        selected,
        fit,
        quality,
        useful_time_s,
        candidate_count,
        search,
        quality.reason if quality.physical is IndicatorColor.NOT_AVAILABLE else None,
        early_data_excluded=early_excluded,
    )


def select_window(
    samples: Sequence[CanonicalSample],
    *,
    reference_time_s: float,
    protocol: Protocol,
    allowed: WindowBounds,
    mode: WindowMode,
    manual: WindowBounds | None = None,
    particle_radius_m: float | None = None,
    preliminary_diffusion_m2_s: float | None = None,
    physical_coefficient: float | None = None,
    thresholds: QualityThresholds = DEFAULT_QUALITY_THRESHOLDS,
    max_candidate_endpoints: int = 64,
    ranking: WindowRanking = WindowRanking.LONGEST,
) -> WindowSelection:
    """Select a fit window without leaving the allowed range or the useful time.

    MANUAL fits the operator's bounds (two distinct times suffice; unestimable metrics are N/D).
    AUTO_FIXED fits the allowed range unchanged. AUTO_STATISTICAL searches inside the allowed range
    for the best statistical window among candidates with at least five distinct times. The
    ranking first prefers the best statistical color; among green or orange windows ``ranking``
    decides (`WindowRanking`: longest anywhere by default, or earliest start then longest); among
    red/N-D windows it prefers the smallest error, then the most samples and the longest.
    PHYSICALLY_GUIDED (GITT only) needs ``r_p``, a preliminary D and the coefficient, and searches
    only up to the useful time. The physical indicator is N/D whenever no useful time is known.
    """
    useful = None
    search = allowed
    if mode is WindowMode.PHYSICALLY_GUIDED:
        if (
            particle_radius_m is None
            or preliminary_diffusion_m2_s is None
            or physical_coefficient is None
        ):
            raise ValueError("physical guidance requires radius, preliminary D and coefficient")
        useful = physical_useful_time(
            protocol=protocol,
            particle_radius_m=particle_radius_m,
            diffusion_m2_s=preliminary_diffusion_m2_s,
            coefficient=physical_coefficient,
        )
        if useful <= allowed.start_s:
            raise ValueError("the useful time ends before the allowed range starts")
        search = WindowBounds(allowed.start_s, min(allowed.end_s, useful))
    physical_reason = ICI_GUIDANCE_UNAVAILABLE if protocol is Protocol.ICI else None
    if mode is WindowMode.MANUAL:
        if manual is None:
            raise ValueError("manual mode requires explicit bounds")
        return evaluate_window(
            samples,
            reference_time_s=reference_time_s,
            protocol=protocol,
            allowed=allowed,
            selected=manual,
            mode=mode,
            thresholds=thresholds,
            physical_reason=physical_reason,
        )
    if mode is WindowMode.AUTO_FIXED:
        return evaluate_window(
            samples,
            reference_time_s=reference_time_s,
            protocol=protocol,
            allowed=allowed,
            selected=allowed,
            mode=mode,
            thresholds=thresholds,
            physical_reason=physical_reason,
            search=allowed,
        )
    if max_candidate_endpoints < 5:
        raise ValueError("max_candidate_endpoints must be at least five")
    series = _series(samples, reference_time_s)
    lo, hi = _range_indices(series, allowed)
    search_lo, search_hi = _range_indices(series, search)
    if search_hi - search_lo + 1 < 5:
        raise ValueError("automatic selection requires at least five distinct times")
    grid = np.arange(search_lo, search_hi + 1)
    if len(grid) > max_candidate_endpoints:
        grid = np.unique(np.rint(np.linspace(search_lo, search_hi, max_candidate_endpoints)))
        grid = grid.astype(np.int64)
    start_grid, end_grid = np.meshgrid(grid, grid, indexing="ij")
    keep = end_grid - start_grid >= 4
    start, end = start_grid[keep], end_grid[keep]
    rank, worst, _ = _candidate_scores(series, start, end, lo, hi, thresholds)
    duration = series.distinct[end] - series.distinct[start]
    distinct_count = end - start + 1
    adequate = rank <= 1
    # Lexicographic keys, least significant first; the start index is the final tie-break.
    if ranking is WindowRanking.EARLY_ANCHORED:
        keys = (
            np.where(adequate, worst, -duration),
            -distinct_count,
            np.where(adequate, -duration, worst),
            np.where(adequate, start, 0),
            rank,
        )
    else:
        keys = (
            start,
            np.where(adequate, worst, -duration),
            -distinct_count,
            np.where(adequate, -duration, worst),
            rank,
        )
    order = np.lexsort(keys)
    best = int(order[0])
    selected = WindowBounds(float(series.distinct[start[best]]), float(series.distinct[end[best]]))
    return evaluate_window(
        samples,
        reference_time_s=reference_time_s,
        protocol=protocol,
        allowed=allowed,
        selected=selected,
        mode=mode,
        useful_time_s=useful,
        thresholds=thresholds,
        physical_reason=physical_reason,
        candidate_count=len(start),
        search=search,
    )


def refine_physically_guided(
    samples: Sequence[CanonicalSample],
    *,
    reference_time_s: float,
    allowed: WindowBounds,
    particle_radius_m: float,
    initial_diffusion_m2_s: float,
    physical_coefficient: float,
    diffusion_from_fit: Callable[[SqrtTimeFit], float | None],
    thresholds: QualityThresholds = DEFAULT_QUALITY_THRESHOLDS,
    max_iterations: int = 5,
    max_candidate_endpoints: int = 64,
    ranking: WindowRanking = WindowRanking.LONGEST,
) -> GuidedRefinement:
    """Run the GITT physically guided search and re-check it after D is updated.

    Each iteration searches only up to the useful time of the current D, refits, and recomputes D
    with ``diffusion_from_fit``. It stops when the selected window no longer changes, when D becomes
    unusable, or after ``max_iterations`` (deterministic, reason recorded). The final physical color
    is re-evaluated against the useful time implied by the final D, never forced to green.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    diffusion = initial_diffusion_m2_s
    history: list[tuple[float, float]] = []
    previous: WindowBounds | None = None
    selection: WindowSelection | None = None
    reason: str | None = "guided_refinement_not_stabilized"
    converged = False
    final_diffusion: float | None = None
    for _ in range(max_iterations):
        selection = select_window(
            samples,
            reference_time_s=reference_time_s,
            protocol=Protocol.GITT,
            allowed=allowed,
            mode=WindowMode.PHYSICALLY_GUIDED,
            particle_radius_m=particle_radius_m,
            preliminary_diffusion_m2_s=diffusion,
            physical_coefficient=physical_coefficient,
            thresholds=thresholds,
            max_candidate_endpoints=max_candidate_endpoints,
            ranking=ranking,
        )
        assert selection.useful_time_s is not None
        history.append((selection.useful_time_s, diffusion))
        updated = diffusion_from_fit(selection.fit)
        final_diffusion = updated
        if updated is None or not math.isfinite(updated) or updated <= 0:
            reason = "diffusion_unavailable_after_fit"
            break
        if selection.selected == previous:
            converged, reason = True, None
            break
        previous = selection.selected
        diffusion = updated
    assert selection is not None
    if final_diffusion is not None and math.isfinite(final_diffusion) and final_diffusion > 0:
        useful = physical_useful_time(
            protocol=Protocol.GITT,
            particle_radius_m=particle_radius_m,
            diffusion_m2_s=final_diffusion,
            coefficient=physical_coefficient,
        )
        physical = physical_indicator(selection.fit.effective_end_s, useful)
        quality = selection.quality
        quality = replace(
            quality,
            physical=physical,
            overall=worst_color(quality.statistical, physical),
            needs_review=quality.statistical is not IndicatorColor.GREEN
            or physical in {IndicatorColor.ORANGE, IndicatorColor.RED}
            or selection.early_data_excluded,
        )
        selection = replace(selection, quality=quality, useful_time_s=useful)
    return GuidedRefinement(
        selection, final_diffusion, len(history), converged, reason, tuple(history)
    )
