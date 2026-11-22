"""Pre-pulse and post-pulse IR drops of GITT/ICI impulses.

An IR drop is a two-sample measurement on real canonical rows, never a fit:

* pre-pulse (``R_on``): reference = last sample of the rest before the pulse, measured sample =
  last pulse sample with ``t <= T_on`` after the reference;
* post-pulse (``R_off``): reference = last sample under current, measured sample = last sample of
  the following rest with ``t <= T_off`` after the reference.

``R = |ΔV / ΔI|`` with signed ``ΔV`` and ``ΔI`` retained; ``ΔI = 0`` leaves R absent. A caller may
replace, for one side of one impulse, the measured sample by an exact ``source_row`` of the same
segment (a click keeps the row identity) or the sample time by a local `IrSampleTime` (the
same "last sample with t <= T" rule with that impulse's own T); no interpolation, nearest choice or
fallback is ever applied.
The pre-pulse sample is also the sample of the "selected sample" IR correction of GITT_DELTA and
of the Delacourt fixed offset, so the measurement and the correction never diverge.

These types and the selection helper are used by `gregorithm.gitt_ici.analysis.measure_ir_drops`
and by the IR correction of the analysis; they do not activate any correction by themselves.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from .contracts import CanonicalSample
from .events import last_sample_at_or_before
from .kernels import measure_resistance


class IrDropSide(StrEnum):
    """Side of an impulse an IR drop is measured on."""

    PRE_PULSE = "pre_pulse"
    POST_PULSE = "post_pulse"


@dataclass(frozen=True, slots=True)
class IrSampleTime:
    """Local sample time [s] of one side of one impulse, replacing the global time for that side."""

    seconds: float

    def __post_init__(self) -> None:
        """Require a finite positive time."""
        if not (self.seconds > 0 and self.seconds < float("inf")):
            raise ValueError("a local IR sample time must be finite and positive")


IrSampleSelections = Mapping[tuple[str, IrDropSide], "int | IrSampleTime"]
"""Per impulse and side: an exact ``source_row`` of a real sample of that side, or a local time."""


def resolve_selection(
    selection: int | IrSampleTime | None, default_target_s: float
) -> tuple[float, int | None, bool]:
    """Return ``(target time, exact row or None, local time used)`` of one side's selection."""
    if isinstance(selection, IrSampleTime):
        return selection.seconds, None, True
    return default_target_s, selection, False


@dataclass(frozen=True, slots=True)
class IrDropSample:
    """One real canonical sample used by an IR drop, with its time from the side's reference."""

    source_row: int
    time_s: float
    relative_time_s: float
    voltage_v: float
    current_a: float
    capacity_ah: float


@dataclass(frozen=True, slots=True)
class IrDrop:
    """The IR drop of one side of one impulse.

    ``reason`` is None when R is available and otherwise a stable code: ``rest_before_missing``,
    ``pulse_samples_missing``, ``rest_after_missing``, ``no_sample_within_target``,
    ``manual_sample_invalid`` or ``zero_current_step``. ``reference`` and ``measured`` are kept
    whenever they exist, so an unavailable R still shows which samples were examined.
    """

    event_id: str
    side: IrDropSide
    target_time_s: float
    manual: bool
    reference: IrDropSample | None
    measured: IrDropSample | None
    delta_voltage_v: float | None
    delta_current_a: float | None
    resistance_ohm: float | None
    reason: str | None
    local_time: bool = False

    @property
    def available(self) -> bool:
        """True when the resistance could be measured."""
        return self.resistance_ohm is not None


@dataclass(frozen=True, slots=True)
class ImpulseIrDrops:
    """Both IR drops of one impulse."""

    event_id: str
    pre_pulse: IrDrop
    post_pulse: IrDrop

    def side(self, side: IrDropSide) -> IrDrop:
        """Return the IR drop of ``side``."""
        return self.pre_pulse if side is IrDropSide.PRE_PULSE else self.post_pulse


def side_samples(
    rest_before: tuple[CanonicalSample, ...],
    active: tuple[CanonicalSample, ...],
    rest_after: tuple[CanonicalSample, ...],
    side: IrDropSide,
) -> tuple[CanonicalSample | None, tuple[CanonicalSample, ...], str]:
    """Return ``(reference, candidate samples, reason)`` of a side; the reason applies when the
    reference or the candidates are missing."""
    if side is IrDropSide.PRE_PULSE:
        if not rest_before:
            return None, active, "rest_before_missing"
        return rest_before[-1], active, "pulse_samples_missing"
    if not active:
        return None, rest_after, "pulse_samples_missing"
    return active[-1], rest_after, "rest_after_missing"


def select_ir_sample(
    reference: CanonicalSample,
    candidates: tuple[CanonicalSample, ...],
    target_time_s: float,
    manual_row: int | None,
) -> tuple[CanonicalSample | None, str | None]:
    """Return ``(measured sample, reason)`` for a side.

    Automatic: the last candidate with ``0 <= t <= target`` (ties by source row), the reference
    row excluded. Manual: exactly the candidate with ``manual_row``, which must follow the
    reference in time; otherwise ``manual_sample_invalid`` and no sample.
    """
    if manual_row is not None:
        sample = next((s for s in candidates if s.source_row == manual_row), None)
        if (
            sample is None
            or sample.source_row == reference.source_row
            or sample.time_s < reference.time_s
        ):
            return None, "manual_sample_invalid"
        return sample, None
    sample = last_sample_at_or_before(
        candidates,
        reference_time_s=reference.time_s,
        target_time_s=target_time_s,
        excluded_source_row=reference.source_row,
    )
    return sample, None if sample is not None else "no_sample_within_target"


def build_ir_drop(
    event_id: str,
    side: IrDropSide,
    target_time_s: float,
    selection: int | IrSampleTime | None,
    reference: CanonicalSample | None,
    candidates: tuple[CanonicalSample, ...],
    missing_reason: str,
    polarity: int,
) -> IrDrop:
    """Measure one side with the global ``target_time_s`` unless ``selection`` replaces it.

    See `IrDrop` for the reasons of an absent value.
    """
    target_time_s, manual_row, local = resolve_selection(selection, target_time_s)

    def view(sample: CanonicalSample, origin: CanonicalSample) -> IrDropSample:
        """Typed view of a sample with its time from the reference."""
        capacity = sample.charge_capacity_ah if polarity >= 0 else sample.discharge_capacity_ah
        return IrDropSample(
            sample.source_row,
            sample.time_s,
            sample.time_s - origin.time_s,
            sample.voltage_v,
            sample.current_a,
            capacity,
        )

    manual = manual_row is not None
    if reference is None or not candidates:
        return IrDrop(
            event_id,
            side,
            target_time_s,
            manual,
            None if reference is None else view(reference, reference),
            None,
            None,
            None,
            None,
            missing_reason,
            local,
        )
    sample, reason = select_ir_sample(reference, candidates, target_time_s, manual_row)
    if sample is None:
        return IrDrop(
            event_id,
            side,
            target_time_s,
            manual,
            view(reference, reference),
            None,
            None,
            None,
            None,
            reason,
            local,
        )
    value = measure_resistance(reference, sample)
    return IrDrop(
        event_id,
        side,
        target_time_s,
        manual,
        view(reference, reference),
        view(sample, reference),
        value.delta_voltage_v,
        value.delta_current_a,
        value.resistance_ohm,
        None if value.resistance_ohm is not None else "zero_current_step",
        local,
    )
