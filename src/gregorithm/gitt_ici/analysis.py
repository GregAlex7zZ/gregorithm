"""Event-level GITT/ICI analysis and recalculation planning.

This module keeps the science in the library: it turns canonical events and samples into
`BatchTask` objects whose bodies select OCV sources, IR corrections and fit windows and call the
GITT_VA, GITT_DELTA, ICI and analytical Delacourt kernels. A GUI controller only chooses events,
variants, windows and settings; it never reimplements formulas in callbacks.

Inputs are exclusively canonical gregorithm samples and segments, identified by ``source_row`` and
segment ID. Nothing is re-segmented, re-classified or interpolated. Constant current is the
authoritative CC label of the elaborated segment (events are only built on ``Step.CC``).
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum

from gregorithm.contracts import JsonValue
from gregorithm.diagnostics import Diagnostic, DiagnosticCode, DiagnosticKind
from gregorithm.elaboration import Step
from gregorithm.identity import hash_parameters

from .batch import BatchTask, ComputeOutput, RecoverableAnalysisError
from .contracts import (
    CanonicalSample,
    CanonicalSegment,
    DelacourtOffsetMode,
    DelacourtParameters,
    DiffusionMethod,
    DiffusionResult,
    GittIciEvent,
    IrCorrection,
    LogTimeFit,
    OcvPoint,
    OcvSource,
    PhysicalInputs,
    Protocol,
    SqrtTimeFit,
)
from .delacourt import fit_delacourt
from .ir_drop import (
    ImpulseIrDrops,
    IrDropSide,
    IrSampleSelections,
    IrSampleTime,
    build_ir_drop,
    resolve_selection,
    select_ir_sample,
    side_samples,
)
from .kernels import (
    delta_e_tau,
    gitt_delta,
    gitt_va,
    ici_diffusion,
    ici_intermediates,
    ir_correction_offset,
    ocv_difference,
)
from .ocv import OcvSampleTime, OcvTargetPlan, extrapolated_ocv, fit_log_time, sampled_ocv
from .quality import (
    DEFAULT_DELACOURT_THRESHOLDS,
    DEFAULT_QUALITY_THRESHOLDS,
    DelacourtQualityThresholds,
    QualityThresholds,
    delacourt_quality,
    gitt_delta_quality,
)
from .review import EventCoordinates, ResultIdentity, ResultSnapshot, ResultState
from .windows import (
    ICI_GUIDANCE_UNAVAILABLE,
    InitialWindowRule,
    WindowBounds,
    WindowMode,
    WindowRanking,
    WindowSelection,
    evaluate_window,
    refine_physically_guided,
    select_window,
    suggest_initial_window,
)

OcvSampleSelections = Mapping[int, "int | OcvSampleTime"]
"""Per-rest sampled OCV choices: ``rest_segment_id -> source_row`` of a real sample of that rest, or
an `OcvSampleTime` replacing the global sample time for that rest only."""

SUGGESTED_OCV_TARGET_S = 10_800.0
"""Bibliographic suggestion for the relaxation target; editable, never a validity limit."""


class RecalculationScope(StrEnum):
    """Global recalculation of every impulse in scope or one opened impulse."""

    GLOBAL = "global"
    LOCAL = "local"


class WindowRole(StrEnum):
    """Segment whose sqrt-time window a request governs: the CC pulse or the following rest."""

    PULSE = "pulse"
    REST_AFTER = "rest_after"


_GITT_SOURCES = frozenset({OcvSource.REST_SAMPLE, OcvSource.RELAXATION_EXTRAPOLATED})
_DELACOURT_VARIANTS = frozenset(mode.value for mode in DelacourtOffsetMode)


@dataclass(frozen=True, slots=True)
class MethodVariant:
    """A selectable result series: method, OCV source and fit variant.

    GITT_VA, GITT_DELTA and Delacourt accept the sampled or relaxation-extrapolated OCV; ICI also
    accepts its default pseudo-OCV (intercept). Delacourt's variant is the offset mode
    (``"fixed"`` default, ``"free"``); other methods use ``"default"``.
    """

    method: DiffusionMethod
    ocv_source: OcvSource
    variant: str = "default"

    def __post_init__(self) -> None:
        """Reject combinations the specification does not define."""
        if self.method is DiffusionMethod.ICI:
            allowed_variant = self.variant == "default"
        elif self.method is DiffusionMethod.GITT_SPHERE_FULL:
            allowed_variant = self.variant in _DELACOURT_VARIANTS
        else:
            allowed_variant = self.variant == "default"
        if not allowed_variant:
            raise ValueError(f"variant {self.variant!r} is not defined for {self.method.value}")
        if self.method is not DiffusionMethod.ICI and self.ocv_source not in _GITT_SOURCES:
            raise ValueError(f"{self.method.value} uses sampled or extrapolated OCV only")

    def identity(self, event_id: str) -> ResultIdentity:
        """Return the logical result identity of this series for one event."""
        return ResultIdentity(event_id, self.method, self.variant, self.ocv_source)


def delacourt_variant(
    offset_mode: DelacourtOffsetMode = DelacourtOffsetMode.FIXED,
    ocv_source: OcvSource = OcvSource.REST_SAMPLE,
) -> MethodVariant:
    """Return the Delacourt series for an offset mode (fixed is the approved default)."""
    return MethodVariant(DiffusionMethod.GITT_SPHERE_FULL, ocv_source, offset_mode.value)


def default_method_variants(protocol: Protocol) -> tuple[MethodVariant, ...]:
    """Return the preselected series: GITT_VA+GITT_DELTA sampled for GITT, ICI pseudo for ICI.

    Preselected does not mean computable: missing inputs still leave D unavailable with a reason.
    ICI and Delacourt can be added to a GITT analysis explicitly.
    """
    if protocol is Protocol.GITT:
        return (
            MethodVariant(DiffusionMethod.GITT_VA, OcvSource.REST_SAMPLE),
            MethodVariant(DiffusionMethod.GITT_DELTA, OcvSource.REST_SAMPLE),
        )
    return (MethodVariant(DiffusionMethod.ICI, OcvSource.ICI_INTERCEPT),)


@dataclass(frozen=True, slots=True)
class AnalysisOptions:
    """Entry options of an analysis.

    ``auto_analysis_on_recognition`` is off by default: recognizing events computes nothing until
    the operator enables it or starts a calculation explicitly.
    """

    protocol: Protocol
    auto_analysis_on_recognition: bool = False
    variants: tuple[MethodVariant, ...] | None = None

    def selected_variants(self) -> tuple[MethodVariant, ...]:
        """Return the explicitly selected series or the protocol defaults."""
        return default_method_variants(self.protocol) if self.variants is None else self.variants


@dataclass(frozen=True, slots=True)
class FitWindowRequest:
    """How one sqrt-time window is chosen for one event and role.

    ``mode`` defaults to the first automatic pass, a statistical search inside the suggested range.
    For MANUAL, ``bounds`` are the operator's window; for automatic modes optional ``bounds``
    replace the suggested allowed range (operator-editable suggestion).
    """

    mode: WindowMode = WindowMode.AUTO_STATISTICAL
    bounds: WindowBounds | None = None

    def __post_init__(self) -> None:
        """Manual windows need explicit bounds."""
        if self.mode is WindowMode.MANUAL and self.bounds is None:
            raise ValueError("a manual window request needs bounds")

    def payload(self) -> JsonValue:
        """Return a JSON description used in fingerprints and technical details."""
        bounds: JsonValue = (
            None if self.bounds is None else [self.bounds.start_s, self.bounds.end_s]
        )
        return {"mode": self.mode.value, "bounds": bounds}


@dataclass(frozen=True, slots=True)
class PhysicalGuidance:
    """One D source chosen for a whole physically guided run.

    The source is a series (method, variant, OCV source); each event uses that series' estimate of
    the same event, never another event's or another method's value. ``physical_coefficient``
    multiplies ``r_p²/D``; the specification cites 0.0032 only as a reference to validate.
    """

    source: MethodVariant
    physical_coefficient: float
    max_iterations: int = 5

    def __post_init__(self) -> None:
        """Require a finite positive coefficient and a positive iteration bound."""
        if not math.isfinite(self.physical_coefficient) or self.physical_coefficient <= 0:
            raise ValueError("physical_coefficient must be finite and positive")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")


@dataclass(frozen=True, slots=True)
class GuidanceValue:
    """The per-event D used to guide a window, with provenance.

    ``provisional`` is True when the source result was not accepted; using it never changes its
    review state.
    """

    identity: ResultIdentity
    diffusion_m2_s: float
    state: ResultState
    calculation_fingerprint: str

    @property
    def provisional(self) -> bool:
        """Whether the guiding D comes from a not-yet-accepted result."""
        return self.state is not ResultState.ACCEPTED

    def payload(self) -> JsonValue:
        """Return the provenance as JSON for details and fingerprints."""
        return {
            "event_id": self.identity.event_id,
            "method": self.identity.method.value,
            "variant": self.identity.variant,
            "ocv_source": self.identity.ocv_source.value,
            "diffusion_m2_s": self.diffusion_m2_s,
            "state": self.state.value,
            "provisional": self.provisional,
            "calculation_fingerprint": self.calculation_fingerprint,
        }


def resolve_guidance(
    snapshot: ResultSnapshot, event_id: str, source: MethodVariant
) -> GuidanceValue | None:
    """Return the chosen source's D for ``event_id`` or None; there is no implicit fallback.

    Errors, rejected, stale and non-positive or missing values are not usable as guidance.
    """
    record = snapshot.current(source.identity(event_id))
    if (
        record is None
        or record.stale
        or record.state in {ResultState.ERROR, ResultState.REJECTED}
        or not record.log_plottable
    ):
        return None
    value = record.diffusion_m2_s
    assert value is not None
    return GuidanceValue(record.identity, value, record.state, record.calculation_fingerprint)


@dataclass(frozen=True, slots=True)
class AnalysisSettings:
    """Applied global settings of one analysis.

    ``geometry`` holds the operator's direct V_act, A_eff and r_p in SI, each optional and constant
    for every impulse. Without V_act/A_eff, GITT_VA/GITT_DELTA/ICI keep their intermediates but D
    is unavailable; Delacourt needs only r_p. ``ir_correction`` (default none) applies to
    GITT_DELTA. ``delacourt_ir_correction`` is the IR source of the Delacourt fixed offset; it
    defaults to the selected sample, i.e. the initial voltage
    jump measured at ``ir_sample_time_s`` (default +1 s from the last rest sample), which is
    also the pre-pulse IR drop target. ``post_pulse_sample_time_s`` is the independent post-pulse
    (R_off) target; None uses ``ir_sample_time_s``. It feeds no D result, so it has no dependency.
    ``relaxation_fit_start_fraction`` is the "last 80 %" time preset of the LOG_TIME_2P fit (single
    rests can override it through ``OcvTargetPlan.fit_start_overrides``). ``ocv_sample_time_s`` is
    the global time of the sampled OCV, from the relaxation origin: the last real rest sample with
    ``t <= ocv_sample_time_s``; None (default) keeps the last sample of the rest.
    """

    geometry: PhysicalInputs | None = None
    ir_correction: IrCorrection = IrCorrection.NONE
    ir_sample_time_s: float = 1.0
    ocv_targets: OcvTargetPlan | None = None
    relaxation_fit_start_fraction: float = 0.2
    thresholds: QualityThresholds = DEFAULT_QUALITY_THRESHOLDS
    delacourt_thresholds: DelacourtQualityThresholds = DEFAULT_DELACOURT_THRESHOLDS
    log10_theta_min: float = -6.0
    log10_theta_max: float = 3.0
    max_candidate_endpoints: int = 64
    delacourt_ir_correction: IrCorrection = IrCorrection.SELECTED_SAMPLE
    window_ranking: WindowRanking = WindowRanking.LONGEST
    initial_window_rule: InitialWindowRule = InitialWindowRule.FULL_SEGMENT
    initial_window_fraction: float | None = None
    post_pulse_sample_time_s: float | None = None
    ocv_sample_time_s: float | None = None

    def __post_init__(self) -> None:
        """Validate scalar settings."""
        if not math.isfinite(self.ir_sample_time_s) or self.ir_sample_time_s <= 0:
            raise ValueError("ir_sample_time_s must be finite and positive")
        post = self.post_pulse_sample_time_s
        if post is not None and (not math.isfinite(post) or post <= 0):
            raise ValueError("post_pulse_sample_time_s must be finite and positive")
        if not (0 <= self.relaxation_fit_start_fraction < 1):
            raise ValueError("relaxation_fit_start_fraction must lie in [0, 1)")
        ocv_time = self.ocv_sample_time_s
        if ocv_time is not None and (not math.isfinite(ocv_time) or ocv_time <= 0):
            raise ValueError("ocv_sample_time_s must be finite and positive")

    def dependency_fingerprints(self) -> dict[str, str]:
        """Return named fingerprints used for targeted invalidation after "Apply".

        Quality thresholds are intentionally absent: changing them does not alter data, fits or
        acceptance; they only enter cache keys of automatic selections.
        """
        geometry = self.geometry
        return {
            "geometry": hash_parameters(
                {
                    "active_volume_m3": None if geometry is None else geometry.active_volume_m3,
                    "effective_area_m2": None if geometry is None else geometry.effective_area_m2,
                }
            ),
            "particle_radius": hash_parameters(
                {"particle_radius_m": None if geometry is None else geometry.particle_radius_m}
            ),
            "ir": hash_parameters(
                {"mode": self.ir_correction.value, "sample_time_s": self.ir_sample_time_s}
            ),
            "delacourt_ir": hash_parameters(
                {
                    "mode": self.delacourt_ir_correction.value,
                    "sample_time_s": self.ir_sample_time_s,
                }
            ),
            "relaxation_fit": hash_parameters(
                {"start_fraction": self.relaxation_fit_start_fraction}
            ),
            "window_rules": hash_parameters(
                {
                    "ranking": self.window_ranking.value,
                    "initial_rule": self.initial_window_rule.value,
                    "initial_fraction": self.initial_window_fraction,
                }
            ),
            "delacourt_search": hash_parameters(
                {"log10_min": self.log10_theta_min, "log10_max": self.log10_theta_max}
            ),
        }


def ir_sample_dependencies(
    samples: IrSampleSelections | None, event_ids: Sequence[str]
) -> dict[str, str]:
    """Return the ``ir_sample:<event_id>`` fingerprints of the pre-pulse IR drop samples.

    Only series whose selected-sample IR correction uses the pre-pulse sample carry this
    dependency (GITT_DELTA with that correction, fixed-offset Delacourt with it); passing the
    result to `ResultSnapshot.invalidate` after a manual change marks exactly those stale. A
    post-pulse sample feeds no D result and has no dependency.
    """
    samples = samples or {}

    def payload(selection: int | IrSampleTime | None) -> dict[str, JsonValue]:
        """Row selections keep their earlier fingerprint; a local time has its own."""
        if isinstance(selection, IrSampleTime):
            return {"time": selection.seconds}
        return {"row": selection}

    return {
        f"ir_sample:{event_id}": hash_parameters(
            payload(samples.get((event_id, IrDropSide.PRE_PULSE)))
        )
        for event_id in event_ids
    }


def ocv_target_dependencies(plan: OcvTargetPlan, rest_ids: Sequence[int]) -> dict[str, str]:
    """Return ``{"ocv_target:<rest_id>": fingerprint}`` of the targets applied to rests.

    A per-rest fit start override enters the same fingerprint, so changing it marks exactly the
    results using that rest's extrapolated OCV stale; rests without one keep their earlier
    fingerprint (the global fraction is the ``relaxation_fit`` dependency).
    """
    fit_starts = dict(plan.fit_start_overrides)

    def payload(rest_id: int) -> dict[str, JsonValue]:
        """Target, plus the fit start fraction only when this rest overrides it."""
        out: dict[str, JsonValue] = {"target_s": plan.effective_target_s(rest_id)}
        if rest_id in fit_starts:
            out["fit_start_fraction"] = fit_starts[rest_id]
        return out

    return {f"ocv_target:{rest_id}": hash_parameters(payload(rest_id)) for rest_id in rest_ids}


@dataclass(frozen=True, slots=True)
class EventData:
    """Canonical samples of one event and its rests, selected by row identity only.

    ``before_reference`` is the last sample under current that precedes the rest before the pulse;
    it is the time origin of that rest's relaxation or ICI fit and is None when unavailable.
    """

    event: GittIciEvent
    active_segment: CanonicalSegment
    active: tuple[CanonicalSample, ...]
    rest_before_segment: CanonicalSegment | None
    rest_before: tuple[CanonicalSample, ...]
    rest_after_segment: CanonicalSegment | None
    rest_after: tuple[CanonicalSample, ...]
    before_reference: CanonicalSample | None = None


def build_event_data(
    events: Sequence[GittIciEvent],
    segments: Sequence[CanonicalSegment],
    samples: Sequence[CanonicalSample],
) -> tuple[EventData, ...]:
    """Attach canonical samples to events by ``source_row`` ranges of their segments.

    Samples are indexed once (sorted by ``source_row``) and sliced with binary search, so the cost
    is O((n + events) log n). The caller limits ``samples`` to the analysis scope.
    """
    ordered = sorted(samples, key=lambda sample: sample.source_row)
    rows = [sample.source_row for sample in ordered]
    by_id = {segment.segment_id: segment for segment in segments}
    starts = sorted((segment.first_row, segment.segment_id) for segment in segments)
    first_rows = [first for first, _ in starts]

    def rows_of(segment: CanonicalSegment | None) -> tuple[CanonicalSample, ...]:
        """Return the samples whose source rows lie inside ``segment``."""
        if segment is None:
            return ()
        lo = bisect.bisect_left(rows, segment.first_row)
        hi = bisect.bisect_right(rows, segment.last_row)
        return tuple(ordered[lo:hi])

    def segment_of_row(row: int) -> CanonicalSegment | None:
        """Return the canonical segment containing ``row``, if any."""
        position = bisect.bisect_right(first_rows, row) - 1
        if position < 0:
            return None
        segment = by_id[starts[position][1]]
        return segment if segment.first_row <= row <= segment.last_row else None

    result = []
    for event in events:
        active = by_id[event.active_segment_id]
        before = (
            None if event.rest_before_segment_id is None else by_id[event.rest_before_segment_id]
        )
        after = None if event.rest_after_segment_id is None else by_id[event.rest_after_segment_id]
        reference = None
        if before is not None:
            owner = segment_of_row(before.first_row - 1)
            if owner is not None and owner.step != int(Step.REST):
                position = bisect.bisect_left(rows, before.first_row - 1)
                if position < len(rows) and rows[position] == before.first_row - 1:
                    reference = ordered[position]
        result.append(
            EventData(
                event,
                active,
                rows_of(active),
                before,
                rows_of(before),
                after,
                rows_of(after),
                reference,
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class AnalysisPlan:
    """Tasks to run and diagnostics for events that were not analysed."""

    tasks: tuple[BatchTask, ...]
    skipped: tuple[Diagnostic, ...] = ()


@contextmanager
def _recoverable(reason: str) -> Iterator[None]:
    """Turn a technical ValueError into an event-local recoverable error with a stable reason."""
    try:
        yield
    except ValueError as error:
        raise RecoverableAnalysisError(f"{reason}: {error}") from error


def _require(condition: bool, reason: str) -> None:
    """Raise a recoverable error with ``reason`` when ``condition`` is false."""
    if not condition:
        raise RecoverableAnalysisError(reason)


def _capacity(sample: CanonicalSample, polarity: int) -> float:
    """Return the branch capacity of a sample (charge for positive polarity)."""
    return sample.charge_capacity_ah if polarity >= 0 else sample.discharge_capacity_ah


def _coordinates(data: EventData) -> EventCoordinates:
    """Place D at the canonical capacity at the start of the CC segment."""
    event = data.event
    return EventCoordinates(
        event.cycle,
        event.half_cycle,
        event.polarity,
        _capacity(data.active[0], event.polarity),
        _capacity(data.active[-1], event.polarity),
    )


def _fit_payload(fit: SqrtTimeFit) -> JsonValue:
    """Return the sqrt-time statistics shown in the technical panel and tooltips."""
    return {
        "intercept_v": fit.intercept_v,
        "slope_v_per_sqrt_s": fit.slope_v_per_sqrt_s,
        "slope_standard_error": fit.slope_standard_error,
        "requested_s": [fit.requested_start_s, fit.requested_end_s],
        "effective_s": [fit.effective_start_s, fit.effective_end_s],
        "sample_count": len(fit.source_rows),
        "first_row": fit.source_rows[0],
        "last_row": fit.source_rows[-1],
        "rmse_v": fit.rmse_v,
        "r_squared": fit.r_squared,
    }


def _selection_payload(selection: WindowSelection, requested: FitWindowRequest) -> JsonValue:
    """Return window provenance: requested and effective mode, ranges, colors and rule version."""
    quality = selection.quality
    return {
        "requested": requested.payload(),
        "mode": selection.mode.value,
        "allowed_s": [selection.allowed.start_s, selection.allowed.end_s],
        "selected_s": [selection.selected.start_s, selection.selected.end_s],
        "search_s": None
        if selection.search is None
        else [selection.search.start_s, selection.search.end_s],
        "useful_time_s": selection.useful_time_s,
        "candidate_count": selection.candidate_count,
        "reason": selection.reason,
        "rule_version": selection.rule_version,
        "statistical": quality.statistical.value,
        "physical": quality.physical.value,
        "q_linearity_pct": quality.q_linearity_pct,
        "q_stability_pct": quality.q_stability_pct,
        "needs_review": quality.needs_review,
        "early_data_excluded": selection.early_data_excluded,
        "fit": _fit_payload(selection.fit),
    }


@dataclass(slots=True)
class _Context:
    """Inputs shared by the task bodies of one event and series."""

    data: EventData
    variant: MethodVariant
    settings: AnalysisSettings
    windows: Mapping[tuple[str, WindowRole], FitWindowRequest]
    previous: EventData | None
    guidance: PhysicalGuidance | None
    guidance_value: GuidanceValue | None
    cached: JsonValue
    details: dict[str, JsonValue] = field(default_factory=dict)
    cache_value: dict[str, JsonValue] = field(default_factory=dict)
    ir_samples: IrSampleSelections = field(default_factory=dict)
    ocv_samples: OcvSampleSelections = field(default_factory=dict)

    def window(self, event_id: str, role: WindowRole) -> FitWindowRequest:
        """Return the window request of an event/role or the default first automatic pass."""
        return self.windows.get((event_id, role), FitWindowRequest())


def _pulse_reference(data: EventData) -> CanonicalSample:
    """Return the GITT time origin: the last canonical sample of the rest before the pulse."""
    _require(bool(data.rest_before), "rest_before_missing")
    return data.rest_before[-1]


def _ocv_point(
    ctx: _Context,
    segment: CanonicalSegment | None,
    rows: tuple[CanonicalSample, ...],
    reference: CanonicalSample | None,
    label: str,
) -> OcvPoint:
    """Return the sampled or on-demand extrapolated OCV of one rest."""
    _require(segment is not None and bool(rows), f"{label}_rest_missing")
    assert segment is not None
    polarity = ctx.data.event.polarity
    if ctx.variant.ocv_source is OcvSource.REST_SAMPLE:
        manual = ctx.ocv_samples.get(segment.segment_id)
        point, sample_time, why = _sampled_point(
            segment, rows, reference, polarity, manual, ctx.settings.ocv_sample_time_s
        )
        missing = why == "ocv_sample_invalid" and manual is None
        _require(point is not None, f"{label}_ocv_missing" if missing else f"{label}_{why}")
        assert point is not None
        ctx.details[f"ocv_{label}"] = {
            "source": point.source.value,
            "voltage_v": point.voltage_v,
            "capacity_ah": point.capacity_ah,
            "rest_segment_id": point.rest_segment_id,
            "source_row": point.source_row,
            "manual_sample": manual is not None,
            "sample_time_s": sample_time,
        }
        return point
    plan = ctx.settings.ocv_targets
    _require(plan is not None, "ocv_target_required")
    _require(reference is not None, f"{label}_relaxation_reference_missing")
    assert plan is not None and reference is not None
    target = plan.effective_target_s(segment.segment_id)
    with _recoverable(f"{label}_relaxation_fit_unavailable"):
        fit, point = _extrapolate(segment, rows, reference, ctx.settings, target, polarity)
    ctx.details[f"ocv_{label}"] = {
        "source": point.source.value,
        "voltage_v": point.voltage_v,
        "capacity_ah": point.capacity_ah,
        "rest_segment_id": point.rest_segment_id,
        "target_time_s": target,
        "fit_s": [fit.fit_start_s, fit.fit_end_s],
        "alpha_v": fit.alpha_v,
        "beta_v": fit.beta_v,
        "rmse_v": fit.rmse_v,
        "r_squared": fit.r_squared,
        "extrapolation_ratio": fit.extrapolation_ratio,
        "warning": DiagnosticCode.GITT_ICI_EXTRAPOLATION_WARNING.value
        if fit.extrapolation_ratio > 1
        else None,
    }
    return point


def _extrapolate(
    segment: CanonicalSegment,
    rows: tuple[CanonicalSample, ...],
    reference: CanonicalSample,
    settings: AnalysisSettings,
    target_s: float,
    polarity: int,
) -> tuple[LogTimeFit, OcvPoint]:
    """LOG_TIME_2P fit of one rest over its last ``1 - fit start fraction`` of time (the rest's
    override in ``settings.ocv_targets`` or the global fraction), extrapolated to ``target_s``;
    shared by D results and `measure_rest_ocv`."""
    duration = rows[-1].time_s - reference.time_s
    fraction = settings.relaxation_fit_start_fraction
    if settings.ocv_targets is not None:
        fraction = settings.ocv_targets.effective_fit_start(segment.segment_id, fraction)
    fit = fit_log_time(
        segment,
        rows,
        reference_time_s=reference.time_s,
        fit_start_s=max(fraction * duration, 1e-300),
        fit_end_s=duration,
        target_time_s=target_s,
    )
    return fit, extrapolated_ocv(segment, rows, fit, state_polarity=polarity)


@dataclass(frozen=True, slots=True)
class RestOcv:
    """Sampled and extrapolated OCV of one rest, measured independently of any D result.

    ``reference`` is the time origin of the relaxation fit (last sample under current before the
    rest), ``samples`` the rest rows (for plots). Sampled and extrapolated products are always
    separate objects: the extrapolation never replaces the sampled point. A missing product has a
    stable reason (``rest_empty``, ``ocv_sample_invalid``, ``relaxation_reference_missing``,
    ``relaxation_fit_unavailable: ...``, ``not_requested``, ``ocv_no_sample_within_time``).
    ``manual_sample`` is True for a per-rest choice (exact row or local time); ``sample_time_s`` is
    the sample time that chose the sampled point (local or global), None for the last sample or an
    exact row.
    """

    rest_segment_id: int
    polarity: int
    reference: CanonicalSample | None
    samples: tuple[CanonicalSample, ...]
    sampled: OcvPoint | None
    sampled_reason: str | None
    manual_sample: bool
    target_time_s: float | None
    fit: LogTimeFit | None
    extrapolated: OcvPoint | None
    extrapolated_reason: str | None
    sample_time_s: float | None = None


def measure_rest_ocv(
    data: Sequence[EventData],
    settings: AnalysisSettings,
    *,
    samples: OcvSampleSelections | None = None,
    extrapolate: bool = True,
) -> tuple[RestOcv, ...]:
    """Measure every distinct rest of ``data`` once, in order of appearance.

    The sampled point is the last real rest sample, the last one within the global
    ``settings.ocv_sample_time_s``, or the per-rest choice of ``samples`` (exact row or local
    `OcvSampleTime`); the extrapolated point uses the same LOG_TIME_2P fit, fit interval and
    per-rest target and fit start as the D results, so both always agree. Nothing is filled in when
    a product is unavailable.
    """
    samples = samples or {}
    plan = settings.ocv_targets
    seen: set[int] = set()
    out: list[RestOcv] = []
    for item in data:
        polarity = item.event.polarity
        for segment, rows, reference in (
            (item.rest_before_segment, item.rest_before, item.before_reference),
            (item.rest_after_segment, item.rest_after, item.active[-1] if item.active else None),
        ):
            if segment is None or segment.segment_id in seen:
                continue
            seen.add(segment.segment_id)
            manual = samples.get(segment.segment_id)
            sampled, sample_time, sampled_reason = _sampled_point(
                segment, rows, reference, polarity, manual, settings.ocv_sample_time_s
            )
            target = None if plan is None else plan.effective_target_s(segment.segment_id)
            fit = point = None
            reason: str | None = None
            if not extrapolate or target is None:
                reason = "not_requested"
            elif reference is None:
                reason = "relaxation_reference_missing"
            elif not rows:
                reason = "rest_empty"
            else:
                try:
                    fit, point = _extrapolate(segment, rows, reference, settings, target, polarity)
                except ValueError as error:
                    reason = f"relaxation_fit_unavailable: {error}"
            out.append(
                RestOcv(
                    segment.segment_id,
                    polarity,
                    reference,
                    rows,
                    sampled,
                    sampled_reason,
                    manual is not None,
                    target,
                    fit,
                    point,
                    reason,
                    sample_time,
                )
            )
    return tuple(out)


def _sampled_point(
    segment: CanonicalSegment,
    rows: tuple[CanonicalSample, ...],
    reference: CanonicalSample | None,
    polarity: int,
    selection: int | OcvSampleTime | None,
    default_time_s: float | None,
) -> tuple[OcvPoint | None, float | None, str | None]:
    """Resolve one rest's sampled OCV: ``(point, sample time used or None, reason)``.

    An exact row wins, then a local `OcvSampleTime`, then the global time, then the last sample.
    Times are measured from the relaxation origin ``reference`` (the first rest sample when it is
    missing). Reasons: ``rest_empty``, ``ocv_sample_invalid`` (row outside the rest),
    ``ocv_no_sample_within_time`` (no real sample within the chosen time).
    """
    if not rows:
        return None, None, "rest_empty"
    if isinstance(selection, OcvSampleTime):
        time_s: float | None = selection.seconds
    elif selection is None:
        time_s = default_time_s
    else:
        time_s = None
    origin = None if reference is None else reference.time_s
    if time_s is not None:
        point = sampled_ocv(
            segment, rows, state_polarity=polarity, time_s=time_s, reference_time_s=origin
        )
        return point, time_s, None if point is not None else "ocv_no_sample_within_time"
    row = selection if isinstance(selection, int) else None
    point = sampled_ocv(segment, rows, state_polarity=polarity, source_row=row)
    return point, None, None if point is not None else "ocv_sample_invalid"


def ocv_sample_dependencies(
    samples: OcvSampleSelections | None,
    rest_ids: Sequence[int],
    default_time_s: float | None = None,
) -> dict[str, str]:
    """Return the ``ocv_sample:<rest_id>`` fingerprints of the sampled OCV choices.

    ``default_time_s`` is the global ``AnalysisSettings.ocv_sample_time_s``. Only series with the
    sampled OCV source carry this dependency, so a manual row, a local time or a new global time
    marks exactly the results using those rests' sampled OCV stale. Rows and the default "last
    sample" keep their earlier fingerprints.
    """
    samples = samples or {}

    def payload(selection: int | OcvSampleTime | None) -> dict[str, JsonValue]:
        """Exact row, local time, global time or last sample."""
        if isinstance(selection, OcvSampleTime):
            return {"time": selection.seconds}
        if selection is None and default_time_s is not None:
            return {"time": default_time_s}
        return {"row": selection}

    return {
        f"ocv_sample:{rest_id}": hash_parameters(payload(samples.get(rest_id)))
        for rest_id in rest_ids
    }


def _ocv_pair(ctx: _Context) -> tuple[OcvPoint, OcvPoint]:
    """Return the homogeneous before/after OCV pair of the same pulse."""
    data = ctx.data
    before = _ocv_point(
        ctx, data.rest_before_segment, data.rest_before, data.before_reference, "before"
    )
    after = _ocv_point(ctx, data.rest_after_segment, data.rest_after, data.active[-1], "after")
    return before, after


def _sqrt_selection(
    ctx: _Context,
    *,
    samples: tuple[CanonicalSample, ...],
    reference_time_s: float,
    protocol: Protocol,
    event_id: str,
    role: WindowRole,
    diffusion_from_fit: Callable[[SqrtTimeFit], float | None] | None,
    cacheable: bool,
) -> WindowSelection:
    """Choose a sqrt-time window following the request, cache and guidance rules.

    A physically guided request falls back to the statistical search inside the allowed range when
    guidance is impossible (no r_p, no D of the chosen source for this event, or ICI), recording
    the reason; it never borrows another event's or method's D.
    """
    request = ctx.window(event_id, role)
    settings = ctx.settings
    with _recoverable(f"{role.value}_window_unavailable"):
        suggested = suggest_initial_window(
            samples,
            reference_time_s=reference_time_s,
            protocol=protocol,
            rule=settings.initial_window_rule,
            fraction=settings.initial_window_fraction,
        )
        allowed = (
            request.bounds
            if request.mode is not WindowMode.MANUAL and request.bounds is not None
            else suggested
        )
        if request.mode in {WindowMode.MANUAL, WindowMode.AUTO_FIXED}:
            return select_window(
                samples,
                reference_time_s=reference_time_s,
                protocol=protocol,
                allowed=allowed,
                mode=request.mode,
                manual=request.bounds,
                thresholds=settings.thresholds,
            )
        reason = None
        if request.mode is WindowMode.PHYSICALLY_GUIDED:
            radius = None if settings.geometry is None else settings.geometry.particle_radius_m
            if protocol is Protocol.ICI:
                reason = ICI_GUIDANCE_UNAVAILABLE
            elif ctx.guidance is None:
                reason = "guidance_source_not_selected"
            elif radius is None:
                reason = "particle_radius_missing"
            elif ctx.guidance_value is None:
                reason = "guidance_diffusion_unavailable"
            elif diffusion_from_fit is None:
                reason = "guidance_not_applicable"
            else:
                refinement = refine_physically_guided(
                    samples,
                    reference_time_s=reference_time_s,
                    allowed=allowed,
                    particle_radius_m=radius,
                    initial_diffusion_m2_s=ctx.guidance_value.diffusion_m2_s,
                    physical_coefficient=ctx.guidance.physical_coefficient,
                    diffusion_from_fit=diffusion_from_fit,
                    thresholds=settings.thresholds,
                    max_iterations=ctx.guidance.max_iterations,
                    max_candidate_endpoints=settings.max_candidate_endpoints,
                    ranking=settings.window_ranking,
                )
                ctx.details["guidance"] = {
                    "source": ctx.guidance_value.payload(),
                    "physical_coefficient": ctx.guidance.physical_coefficient,
                    "iterations": refinement.iterations,
                    "converged": refinement.converged,
                    "reason": refinement.reason,
                    "history": [[useful, value] for useful, value in refinement.history],
                }
                return refinement.selection
            ctx.details["guidance"] = {"applied": False, "reason": reason}
        cached = _cached_bounds(ctx.cached, role, allowed) if cacheable else None
        if cached is not None:
            bounds, count = cached
            selection = evaluate_window(
                samples,
                reference_time_s=reference_time_s,
                protocol=protocol,
                allowed=allowed,
                selected=bounds,
                mode=WindowMode.AUTO_STATISTICAL,
                thresholds=settings.thresholds,
                physical_reason=ICI_GUIDANCE_UNAVAILABLE if protocol is Protocol.ICI else None,
                candidate_count=count,
                search=allowed,
            )
        else:
            selection = select_window(
                samples,
                reference_time_s=reference_time_s,
                protocol=protocol,
                allowed=allowed,
                mode=WindowMode.AUTO_STATISTICAL,
                thresholds=settings.thresholds,
                max_candidate_endpoints=settings.max_candidate_endpoints,
                ranking=settings.window_ranking,
            )
        if cacheable:
            ctx.cache_value[role.value] = [
                selection.selected.start_s,
                selection.selected.end_s,
                selection.candidate_count,
            ]
        if reason is not None:
            selection = replace(selection, reason=reason)
        return selection


def _cached_bounds(
    cached: JsonValue, role: WindowRole, allowed: WindowBounds
) -> tuple[WindowBounds, int] | None:
    """Return valid cached automatic window bounds, treating anything malformed as a miss."""
    entry = cached.get(role.value) if isinstance(cached, dict) else None
    if not isinstance(entry, list) or len(entry) != 3:
        return None
    start, end, count = entry
    if not (
        isinstance(start, float)
        and isinstance(end, float)
        and isinstance(count, int)
        and allowed.start_s <= start < end <= allowed.end_s
    ):
        return None
    return WindowBounds(start, end), count


def _va_diffusion(
    geometry: PhysicalInputs, delta_u: float, tau: float, *, cc_valid: bool
) -> Callable[[SqrtTimeFit], float | None]:
    """Return the GITT_VA D of a refitted pulse window, used by the guided re-check."""

    def diffusion_from_fit(fit: SqrtTimeFit) -> float | None:
        """Recompute GITT_VA D from ``fit`` with this series' OCV step and pulse duration."""
        return gitt_va(
            geometry, delta_u_v=delta_u, tau_s=tau, pulse_fit=fit, cc_valid=cc_valid
        ).diffusion_m2_s

    return diffusion_from_fit


def _unavailable(
    method: DiffusionMethod,
    delta_u_v: float | None,
    tau_s: float | None,
    reason: str,
    *,
    response: float | None = None,
    delta_e: float | None = None,
) -> DiffusionResult:
    """Return a technically unavailable D that keeps its computable intermediates."""
    return DiffusionResult(method, None, None, delta_u_v, tau_s, response, delta_e, reason)


def _ir_offset(
    ctx: _Context,
    reference: CanonicalSample,
    diffusion_from_fit: Callable[[SqrtTimeFit], float | None] | None = None,
    *,
    mode: IrCorrection,
    label: str = "ir",
) -> float:
    """Return the single constant IR correction ``c`` of the pulse; applied once only.

    The selected-sample mode uses the pre-pulse IR drop sample (`measure_ir_drops`): the last
    canonical pulse sample with ``t <= ir_sample_time_s``, or the exact manual row of
    ``ir_samples``; its row, relative time and voltage are recorded. Without such a sample a
    recoverable error ``<label>_sample_missing`` (``<label>_sample_invalid`` for a manual row
    outside the pulse) is raised: no correction is invented and no other mode is used.
    The intercept mode uses the same pulse window rule as GITT_VA, including a guided refinement
    driven by ``diffusion_from_fit``, so ``a_on`` is the intercept of the GITT pulse fit.
    """
    settings = ctx.settings
    selected_voltage = None
    intercept = None
    record: dict[str, JsonValue] = {"mode": mode.value}
    if mode is IrCorrection.SELECTED_SAMPLE:
        target, manual_row, local = resolve_selection(
            ctx.ir_samples.get((ctx.data.event.event_id, IrDropSide.PRE_PULSE)),
            settings.ir_sample_time_s,
        )
        sample, reason = select_ir_sample(reference, ctx.data.active, target, manual_row)
        invalid = reason == "manual_sample_invalid"
        _require(sample is not None, f"{label}_sample_{'invalid' if invalid else 'missing'}")
        assert sample is not None
        selected_voltage = sample.voltage_v
        record.update(
            {
                "target_time_s": target,
                "sample_row": sample.source_row,
                "sample_time_s": sample.time_s - reference.time_s,
                "sample_voltage_v": sample.voltage_v,
                "manual_sample": manual_row is not None,
                "local_time": local,
            }
        )
    elif mode is IrCorrection.GITT_INTERCEPT:
        selection = _sqrt_selection(
            ctx,
            samples=ctx.data.active,
            reference_time_s=reference.time_s,
            protocol=Protocol.GITT,
            event_id=ctx.data.event.event_id,
            role=WindowRole.PULSE,
            diffusion_from_fit=diffusion_from_fit,
            cacheable=False,
        )
        intercept = selection.fit.intercept_v
        ctx.details[f"{label}_window"] = _selection_payload(
            selection, ctx.window(ctx.data.event.event_id, WindowRole.PULSE)
        )
    offset = ir_correction_offset(
        mode,
        reference_voltage_v=reference.voltage_v,
        selected_voltage_v=selected_voltage,
        gitt_intercept_v=intercept,
    )
    record.update({"reference_row": reference.source_row, "offset_v": offset})
    ctx.details[label] = record
    return offset


def _analyze_gitt_va(ctx: _Context) -> ComputeOutput:
    """Compute GITT_VA: OCV step of the pulse over ``τ`` times the pulse sqrt-time slope."""
    data = ctx.data
    reference = _pulse_reference(data)
    tau = data.active[-1].time_s - reference.time_s
    before, after = _ocv_pair(ctx)
    delta_u = ocv_difference(before, after)
    geometry = ctx.settings.geometry or PhysicalInputs()
    cc_valid = data.event.constant_current
    selection = _sqrt_selection(
        ctx,
        samples=data.active,
        reference_time_s=reference.time_s,
        protocol=Protocol.GITT,
        event_id=data.event.event_id,
        role=WindowRole.PULSE,
        diffusion_from_fit=_va_diffusion(geometry, delta_u, tau, cc_valid=cc_valid),
        cacheable=True,
    )
    result = gitt_va(
        geometry, delta_u_v=delta_u, tau_s=tau, pulse_fit=selection.fit, cc_valid=cc_valid
    )
    ctx.details.update({"tau_s": tau, "reference_row": reference.source_row, "delta_u_v": delta_u})
    ctx.details["window"] = _selection_payload(
        selection, ctx.window(data.event.event_id, WindowRole.PULSE)
    )
    return ComputeOutput(result, selection.quality)


def _analyze_gitt_delta(ctx: _Context) -> ComputeOutput:
    """Compute GITT_DELTA from ``ΔE_τ = E(τ) - V0 - c`` without a mandatory sqrt-time fit."""
    data = ctx.data
    reference = _pulse_reference(data)
    end = data.active[-1]
    tau = end.time_s - reference.time_s
    before, after = _ocv_pair(ctx)
    delta_u = ocv_difference(before, after)
    geometry = ctx.settings.geometry or PhysicalInputs()
    cc_valid = data.event.constant_current
    offset = _ir_offset(
        ctx,
        reference,
        _va_diffusion(geometry, delta_u, tau, cc_valid=cc_valid),
        mode=ctx.settings.ir_correction,
    )
    delta_e = delta_e_tau(
        pulse_end_voltage_v=end.voltage_v,
        reference_voltage_v=reference.voltage_v,
        correction_offset_v=offset,
    )
    result = gitt_delta(
        geometry, delta_u_v=delta_u, tau_s=tau, delta_e_tau_v=delta_e, cc_valid=cc_valid
    )
    ctx.details.update(
        {
            "tau_s": tau,
            "reference_row": reference.source_row,
            "pulse_end_row": end.source_row,
            "delta_u_v": delta_u,
            "delta_e_tau_v": delta_e,
        }
    )
    sign = 1 if data.event.polarity >= 0 else -1
    return ComputeOutput(result, gitt_delta_quality(result, current_sign=sign))


def _rest_fit_selection(
    ctx: _Context, data: EventData, *, cacheable: bool
) -> tuple[WindowSelection, CanonicalSample]:
    """Return the ICI window on the rest after the pulse (origin: last sample under current)."""
    _require(bool(data.rest_after), "rest_after_missing")
    reference = data.active[-1]
    selection = _sqrt_selection(
        ctx,
        samples=data.rest_after,
        reference_time_s=reference.time_s,
        protocol=Protocol.ICI,
        event_id=data.event.event_id,
        role=WindowRole.REST_AFTER,
        diffusion_from_fit=None,
        cacheable=cacheable,
    )
    return selection, reference


def _analyze_ici(ctx: _Context) -> ComputeOutput:
    """Compute ICI with the pseudo-OCV of consecutive interruptions or a same-pulse OCV pair.

    Pseudo-OCV (default): ``ΔU = a_off,j - a_off,j-1`` where interruption j-1 is the rest directly
    before pulse j in the same branch, and ``Δt_I`` is the current time between the references,
    i.e. the duration of pulse j. Sampled/extrapolated OCV: ICI applied to GITT on the same pulse
    with ``Δt_I = τ``. Missing pairs leave D unavailable while intermediates remain.
    """
    data = ctx.data
    selection, pre = _rest_fit_selection(ctx, data, cacheable=True)
    fit = selection.fit
    intermediates = ici_intermediates(fit, pre_voltage_v=pre.voltage_v, pre_current_a=pre.current_a)
    ctx.details["window"] = _selection_payload(
        selection, ctx.window(data.event.event_id, WindowRole.REST_AFTER)
    )
    ctx.details["intermediates"] = {
        "pseudo_ocv_v": intermediates.pseudo_ocv_v,
        "slope_v_per_sqrt_s": intermediates.slope_v_per_sqrt_s,
        "pre_interruption_current_a": intermediates.pre_interruption_current_a,
        "pre_interruption_row": pre.source_row,
        "k_ohm_sqrt_s": intermediates.k_ohm_sqrt_s,
        "signed_resistance_ohm": intermediates.signed_resistance_ohm,
    }
    tau = data.active[-1].time_s - data.rest_before[-1].time_s if data.rest_before else None
    delta_u: float | None = None
    delta_u_std = None
    reason = None
    if ctx.variant.ocv_source is OcvSource.ICI_INTERCEPT:
        previous = ctx.previous
        if previous is None:
            reason = "ici_previous_interruption_missing"
        else:
            prev_selection, prev_reference = _rest_fit_selection(ctx, previous, cacheable=False)
            delta_u = fit.intercept_v - prev_selection.fit.intercept_v
            if fit.covariance is not None and prev_selection.fit.covariance is not None:
                delta_u_std = math.sqrt(
                    max(0.0, fit.covariance[0][0]) + max(0.0, prev_selection.fit.covariance[0][0])
                )
            ctx.details["previous_interruption"] = {
                "event_id": previous.event.event_id,
                "rest_segment_id": previous.event.rest_after_segment_id,
                "reference_row": prev_reference.source_row,
                "window": _selection_payload(
                    prev_selection, ctx.window(previous.event.event_id, WindowRole.REST_AFTER)
                ),
            }
    else:
        before, after = _ocv_pair(ctx)
        delta_u = ocv_difference(before, after)
    geometry = ctx.settings.geometry or PhysicalInputs()
    if reason is None and tau is None:
        reason = "rest_before_missing"
    ctx.details["delta_u_v"] = delta_u
    if reason is not None:
        result = _unavailable(
            DiffusionMethod.ICI, delta_u, tau, reason, response=fit.slope_v_per_sqrt_s
        )
    else:
        assert tau is not None and delta_u is not None
        result = ici_diffusion(
            geometry,
            delta_u_v=delta_u,
            current_application_time_s=tau,
            rest_fit=fit,
            cc_valid=ctx.data.event.constant_current,
            delta_u_standard_uncertainty_v=delta_u_std,
        )
    ctx.details["current_application_time_s"] = tau
    return ComputeOutput(result, selection.quality)


def _analyze_delacourt(ctx: _Context) -> ComputeOutput:
    """Fit the analytical spherical model on the whole CC pulse.

    The fixed offset is ``η = (V0 - U0) + c_IR`` on the raw voltage, with ``c_IR`` from
    ``delacourt_ir_correction`` (default: the initial voltage jump at the selected sample). ``V0``
    is the last rest sample (IR reference), ``U0`` the chosen OCV of the same rest; the baseline
    term converts the correction to the ``U0`` model origin. With a free offset no IR constant is
    computed or subtracted: the estimated offset absorbs it, so nothing is applied twice.
    """
    data = ctx.data
    geometry = ctx.settings.geometry or PhysicalInputs()
    radius = geometry.particle_radius_m
    _require(radius is not None, "particle_radius_required")
    _require(data.event.constant_current, "constant_current_required")
    assert radius is not None
    reference = _pulse_reference(data)
    tau = data.active[-1].time_s - reference.time_s
    before, after = _ocv_pair(ctx)
    mode = DelacourtOffsetMode(ctx.variant.variant)
    offset = 0.0
    if mode is DelacourtOffsetMode.FIXED:
        guided = _va_diffusion(geometry, ocv_difference(before, after), tau, cc_valid=True)
        correction = _ir_offset(
            ctx,
            reference,
            guided,
            mode=ctx.settings.delacourt_ir_correction,
            label="delacourt_ir",
        )
        baseline = reference.voltage_v - before.voltage_v
        offset = baseline + correction
        ctx.details["fixed_offset"] = {
            "v0_v": reference.voltage_v,
            "u0_v": before.voltage_v,
            "baseline_v": baseline,
            "ir_correction_v": correction,
            "total_offset_v": offset,
        }
    with _recoverable("delacourt_fit_unavailable"):
        parameters = DelacourtParameters(
            radius,
            tau,
            before.voltage_v,
            after.voltage_v,
            mode,
            offset,
            ctx.settings.log10_theta_min,
            ctx.settings.log10_theta_max,
        )
        result = fit_delacourt(
            data.active, reference_time_s=reference.time_s, parameters=parameters
        )
    ctx.details.update(
        {
            "tau_s": tau,
            "reference_row": reference.source_row,
            "particle_radius_m": radius,
            "offset_mode": mode.value,
            "fixed_offset_v": offset if mode is DelacourtOffsetMode.FIXED else None,
            "offset_v": result.offset_v,
            "log10_theta_tau": result.log10_theta_tau,
            "diffusion_time_s": result.diffusion_time_s,
            "rmse_v": result.rmse_v,
            "r_squared": result.r_squared,
            "at_search_boundary": result.at_search_boundary,
            "identifiable": result.identifiable,
            "sensitivity": result.sensitivity_norm_v_per_log10_theta,
            "optimizer_evaluations": result.optimizer_evaluations,
            "sample_count": len(result.source_rows),
        }
    )
    return ComputeOutput(result, delacourt_quality(result, ctx.settings.delacourt_thresholds))


_ANALYZERS: dict[DiffusionMethod, Callable[[_Context], ComputeOutput]] = {
    DiffusionMethod.GITT_VA: _analyze_gitt_va,
    DiffusionMethod.GITT_DELTA: _analyze_gitt_delta,
    DiffusionMethod.ICI: _analyze_ici,
    DiffusionMethod.GITT_SPHERE_FULL: _analyze_delacourt,
}


def measure_ir_drops(
    data: EventData,
    settings: AnalysisSettings,
    *,
    samples: IrSampleSelections | None = None,
) -> ImpulseIrDrops:
    """Measure the pre-pulse (R_on) and post-pulse (R_off) IR drops of one impulse.

    Pre-pulse: last rest sample -> pulse sample at ``ir_sample_time_s``; post-pulse: last sample
    under current -> following-rest sample at ``post_pulse_sample_time_s`` (default the same
    time). ``samples`` replaces the measured sample of a side by an exact real row. This is the
    single source of the IR drops: the "selected sample" IR correction uses the same pre-pulse
    sample. Missing samples or a zero current step give an `IrDrop` without R and with a reason;
    nothing is invented. Incomplete events are measured on the samples they have.
    """
    samples = samples or {}
    event = data.event
    post_time = settings.post_pulse_sample_time_s or settings.ir_sample_time_s
    drops = []
    for side, target in (
        (IrDropSide.PRE_PULSE, settings.ir_sample_time_s),
        (IrDropSide.POST_PULSE, post_time),
    ):
        reference, candidates, reason = side_samples(
            data.rest_before, data.active, data.rest_after, side
        )
        drops.append(
            build_ir_drop(
                event.event_id,
                side,
                target,
                samples.get((event.event_id, side)),
                reference,
                candidates,
                reason,
                event.polarity,
            )
        )
    return ImpulseIrDrops(event.event_id, drops[0], drops[1])


def analyze_event(
    data: EventData,
    variant: MethodVariant,
    settings: AnalysisSettings,
    *,
    windows: Mapping[tuple[str, WindowRole], FitWindowRequest] | None = None,
    previous: EventData | None = None,
    guidance: PhysicalGuidance | None = None,
    guidance_value: GuidanceValue | None = None,
    cached: JsonValue = None,
    ir_samples: IrSampleSelections | None = None,
    ocv_samples: OcvSampleSelections | None = None,
) -> ComputeOutput:
    """Compute one series for one event; technical failures raise `RecoverableAnalysisError`.

    Missing physical inputs or pairs produce a result without D but with intermediates and a
    reason; impossible fits or missing context raise a recoverable error (error record).
    ``ir_samples`` holds manual IR drop samples; only the pre-pulse one can affect a result
    (selected-sample IR correction). The IR drops are not copied into the record:
    `measure_ir_drops` is their only source, so a record never carries a stale copy.
    """
    _require(data.event.complete, "incomplete_event")
    ctx = _Context(
        data,
        variant,
        settings,
        windows or {},
        previous,
        guidance,
        guidance_value,
        cached,
        ir_samples=dict(ir_samples or {}),
        ocv_samples=dict(ocv_samples or {}),
    )
    with _recoverable("analysis_unavailable"):
        output = _ANALYZERS[variant.method](ctx)
    details: dict[str, JsonValue] = {
        "method": variant.method.value,
        "variant": variant.variant,
        "ocv_source": variant.ocv_source.value,
        "event_id": data.event.event_id,
        "generation": data.event.elaboration_fingerprint,
        "active_segment_id": data.event.active_segment_id,
        "active_rows": [data.event.active_first_row, data.event.active_last_row],
        "rest_before_segment_id": data.event.rest_before_segment_id,
        "rest_after_segment_id": data.event.rest_after_segment_id,
        "constant_current": data.event.constant_current,
        **ctx.details,
    }
    return ComputeOutput(
        output.result,
        output.quality,
        details,
        _coordinates(data),
        dict(ctx.cache_value) or None,
    )


def _previous_interruptions(data: Sequence[EventData]) -> dict[str, EventData]:
    """Map each event to the event whose following rest is its preceding rest, same branch."""
    by_rest_after = {
        item.event.rest_after_segment_id: item
        for item in data
        if item.event.rest_after_segment_id is not None
    }
    result = {}
    for item in data:
        rest = item.event.rest_before_segment_id
        candidate = None if rest is None else by_rest_after.get(rest)
        if (
            candidate is not None
            and candidate.event.half_cycle == item.event.half_cycle
            and candidate.event.polarity == item.event.polarity
        ):
            result[item.event.event_id] = candidate
    return result


def plan_analysis(
    data: Sequence[EventData],
    *,
    variants: Sequence[MethodVariant],
    settings: AnalysisSettings,
    scope: RecalculationScope = RecalculationScope.GLOBAL,
    event_id: str | None = None,
    windows: Mapping[tuple[str, WindowRole], FitWindowRequest] | None = None,
    guidance: PhysicalGuidance | None = None,
    guidance_snapshot: ResultSnapshot | None = None,
    ir_samples: IrSampleSelections | None = None,
    ocv_samples: OcvSampleSelections | None = None,
) -> AnalysisPlan:
    """Build the tasks of a global or local recalculation.

    GLOBAL covers every event in ``data``; LOCAL covers only ``event_id`` but all the selected
    ``variants`` (the checked methods). Variants not listed are not touched, so their current
    results are preserved. Incomplete events produce a diagnostic and no task. With
    ``guidance``, each event resolves the D of the single chosen source for that same event.
    ``ir_samples`` are manual IR drop samples; a manual pre-pulse sample becomes a dependency only
    of the series whose selected-sample IR correction uses it, so changing it invalidates those.
    ``ocv_samples`` are manual sampled OCV points per rest, a dependency of sampled-OCV series only.
    """
    if scope is RecalculationScope.LOCAL and event_id is None:
        raise ValueError("a local recalculation needs an event_id")
    if len(set(variants)) != len(variants):
        raise ValueError("variants must be unique")
    windows = dict(windows or {})
    ir_samples = dict(ir_samples or {})
    ocv_samples = dict(ocv_samples or {})
    previous_of = _previous_interruptions(data)
    settings_dependencies = settings.dependency_fingerprints()
    tasks: list[BatchTask] = []
    skipped: list[Diagnostic] = []
    selected = [
        item
        for item in data
        if scope is RecalculationScope.GLOBAL or item.event.event_id == event_id
    ]
    if scope is RecalculationScope.LOCAL and not selected:
        raise KeyError(event_id)
    for item in selected:
        event = item.event
        if not event.complete:
            skipped.append(
                Diagnostic(
                    DiagnosticCode.GITT_ICI_INCOMPLETE_EVENT,
                    DiagnosticKind.UNAVAILABLE,
                    "The impulse is incomplete in the canonical scope and is not analysed.",
                    operation="gitt_ici",
                    cycle=event.cycle,
                    half_cycle=event.half_cycle,
                    event_id=event.event_id,
                )
            )
            continue
        previous = previous_of.get(event.event_id)
        for variant in variants:
            value = (
                None
                if guidance is None or guidance_snapshot is None
                else resolve_guidance(guidance_snapshot, event.event_id, guidance.source)
            )
            tasks.append(
                _task(
                    item,
                    variant,
                    settings,
                    settings_dependencies,
                    windows,
                    previous,
                    guidance,
                    value,
                    ir_samples,
                    ocv_samples,
                )
            )
    return AnalysisPlan(tuple(tasks), tuple(skipped))


def _task(
    data: EventData,
    variant: MethodVariant,
    settings: AnalysisSettings,
    settings_dependencies: Mapping[str, str],
    windows: Mapping[tuple[str, WindowRole], FitWindowRequest],
    previous: EventData | None,
    guidance: PhysicalGuidance | None,
    guidance_value: GuidanceValue | None,
    ir_samples: IrSampleSelections | None = None,
    ocv_samples: OcvSampleSelections | None = None,
) -> BatchTask:
    """Build one task with exactly the dependencies its result is computed from."""
    event = data.event
    method = variant.method
    dependencies: dict[str, str] = {
        "generation": event.elaboration_fingerprint,
        "event": event.event_id,
    }
    if method is not DiffusionMethod.GITT_SPHERE_FULL:
        dependencies["geometry"] = settings_dependencies["geometry"]
    if method is DiffusionMethod.GITT_DELTA:
        dependencies["ir"] = settings_dependencies["ir"]
    fixed_delacourt = (
        method is DiffusionMethod.GITT_SPHERE_FULL
        and variant.variant == DelacourtOffsetMode.FIXED.value
    )
    if method is DiffusionMethod.GITT_SPHERE_FULL:
        dependencies["particle_radius"] = settings_dependencies["particle_radius"]
        dependencies["delacourt_search"] = settings_dependencies["delacourt_search"]
        if fixed_delacourt:
            dependencies["delacourt_ir"] = settings_dependencies["delacourt_ir"]
    ir_samples = ir_samples or {}
    sample_ir = (
        method is DiffusionMethod.GITT_DELTA
        and settings.ir_correction is IrCorrection.SELECTED_SAMPLE
    ) or (fixed_delacourt and settings.delacourt_ir_correction is IrCorrection.SELECTED_SAMPLE)
    if sample_ir:
        dependencies.update(ir_sample_dependencies(ir_samples, [event.event_id]))
    intercept_ir = (
        method is DiffusionMethod.GITT_DELTA
        and settings.ir_correction is IrCorrection.GITT_INTERCEPT
    ) or (fixed_delacourt and settings.delacourt_ir_correction is IrCorrection.GITT_INTERCEPT)
    roles: list[tuple[str, WindowRole]] = []
    if method is DiffusionMethod.GITT_VA or intercept_ir:
        roles.append((event.event_id, WindowRole.PULSE))
    if method is DiffusionMethod.ICI:
        roles.append((event.event_id, WindowRole.REST_AFTER))
        if variant.ocv_source is OcvSource.ICI_INTERCEPT and previous is not None:
            roles.append((previous.event.event_id, WindowRole.REST_AFTER))
    if roles:
        dependencies["window_rules"] = settings_dependencies["window_rules"]
    for owner, role in roles:
        request = windows.get((owner, role), FitWindowRequest())
        dependencies[f"window:{role.value}:{owner}"] = hash_parameters(
            {"request": request.payload()}
        )
    if method is DiffusionMethod.GITT_VA:
        if guidance is not None:
            dependencies["particle_radius"] = settings_dependencies["particle_radius"]
        dependencies[f"guidance:{event.event_id}"] = hash_parameters(
            {
                "guidance": None
                if guidance is None
                else {
                    "source": guidance.source.method.value,
                    "variant": guidance.source.variant,
                    "ocv_source": guidance.source.ocv_source.value,
                    "coefficient": guidance.physical_coefficient,
                    "max_iterations": guidance.max_iterations,
                    "value": None if guidance_value is None else guidance_value.payload(),
                }
            }
        )
    if variant.ocv_source is OcvSource.REST_SAMPLE:
        rests = [
            rest
            for rest in (event.rest_before_segment_id, event.rest_after_segment_id)
            if rest is not None
        ]
        dependencies.update(ocv_sample_dependencies(ocv_samples, rests, settings.ocv_sample_time_s))
    if variant.ocv_source is OcvSource.RELAXATION_EXTRAPOLATED:
        dependencies["relaxation_fit"] = settings_dependencies["relaxation_fit"]
        plan = settings.ocv_targets
        rests = [
            rest
            for rest in (event.rest_before_segment_id, event.rest_after_segment_id)
            if rest is not None
        ]
        if plan is not None:
            dependencies.update(ocv_target_dependencies(plan, rests))
    parameters: JsonValue = {
        "variant": [method.value, variant.variant, variant.ocv_source.value],
        "thresholds": hash_parameters(
            {
                "sqrt": dict(asdict(settings.thresholds)),
                "delacourt": dict(asdict(settings.delacourt_thresholds)),
                "max_candidate_endpoints": settings.max_candidate_endpoints,
            }
        ),
    }

    def compute(cached: JsonValue) -> ComputeOutput:
        """Run the event analysis, reusing a cached automatic window when valid."""
        return analyze_event(
            data,
            variant,
            settings,
            windows=windows,
            previous=previous,
            guidance=guidance,
            guidance_value=guidance_value,
            cached=cached,
            ir_samples=ir_samples,
            ocv_samples=ocv_samples,
        )

    return BatchTask(
        variant.identity(event.event_id),
        tuple(sorted(dependencies.items())),
        parameters,
        compute,
    )
