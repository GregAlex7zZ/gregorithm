"""Canonical GITT/ICI event associations and sample selection."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence

from gregorithm.elaboration import Step

from .contracts import CanonicalSample, CanonicalSegment, GittIciEvent, Protocol


def _event_id(protocol: Protocol, fingerprint: str, active: CanonicalSegment) -> str:
    """Hash protocol, generation and active-segment identity into a stable event ID."""
    payload = json.dumps(
        [protocol.value, fingerprint, active.segment_id, active.first_row, active.last_row],
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(b"gregorithm-gitt-ici-event-v1\0" + payload).hexdigest()


def associate_events(
    segments: Sequence[CanonicalSegment],
    *,
    protocol: Protocol,
    elaboration_fingerprint: str,
    complete_only: bool = True,
) -> tuple[GittIciEvent, ...]:
    """Associate canonical active/rest segments for an explicitly selected protocol.

    GITT associates ``rest before -> active -> rest after``. ICI associates an active segment with
    its following rest interruption. Active segments are every non-rest step labelled by
    gregorithm; CC ones are marked ``constant_current`` and CV/OTHER ones are kept with
    ``constant_current=False`` so that their intermediates stay available. Event identities of
    CC segments are unchanged. No raw rows are classified or reordered here.
    """
    if len(elaboration_fingerprint) != 64 or any(
        value not in "0123456789abcdef" for value in elaboration_fingerprint
    ):
        raise ValueError("elaboration_fingerprint must be lowercase SHA-256 hex")
    ordered = sorted(segments, key=lambda item: (item.first_row, item.segment_id))
    events: list[GittIciEvent] = []
    for index, active in enumerate(ordered):
        if active.step == int(Step.REST):
            continue
        before = ordered[index - 1] if index else None
        after = ordered[index + 1] if index + 1 < len(ordered) else None
        if before is not None and (
            before.step != int(Step.REST) or before.last_row + 1 != active.first_row
        ):
            before = None
        if after is not None and (
            after.step != int(Step.REST)
            or active.last_row + 1 != after.first_row
            or after.half_cycle != active.half_cycle
        ):
            after = None
        if before is not None and before.half_cycle not in {0, active.half_cycle}:
            before = None
        complete = after is not None and (protocol is Protocol.ICI or before is not None)
        if complete_only and not complete:
            continue
        events.append(
            GittIciEvent(
                _event_id(protocol, elaboration_fingerprint, active),
                protocol,
                elaboration_fingerprint,
                active.cycle,
                active.half_cycle,
                active.polarity,
                active.segment_id,
                active.first_row,
                active.last_row,
                None if before is None else before.segment_id,
                None if after is None else after.segment_id,
                complete,
                active.step == int(Step.CC),
            )
        )
    return tuple(events)


def last_sample_at_or_before(
    samples: Iterable[CanonicalSample],
    *,
    reference_time_s: float,
    target_time_s: float,
    excluded_source_row: int | None = None,
) -> CanonicalSample | None:
    """Select the last real row at/before target, breaking equal-time ties by source row."""
    if target_time_s < 0:
        raise ValueError("target_time_s must not be negative")
    eligible = (
        sample
        for sample in samples
        if sample.source_row != excluded_source_row
        and sample.time_s >= reference_time_s
        and sample.time_s - reference_time_s <= target_time_s
    )
    return max(eligible, key=lambda sample: (sample.time_s, sample.source_row), default=None)


def samples_for_segment(
    samples: Iterable[CanonicalSample], segment: CanonicalSegment
) -> tuple[CanonicalSample, ...]:
    """Return measured samples by canonical row identity, preserving source order."""
    return tuple(
        sorted(
            (
                sample
                for sample in samples
                if segment.first_row <= sample.source_row <= segment.last_row
            ),
            key=lambda sample: sample.source_row,
        )
    )
