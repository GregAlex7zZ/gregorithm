"""DC internal resistance (DCIR): pulse detection, review and measurement.

Workflow:

1. `detect_dcir` scans the step segments of an elaboration result and publishes Pending
   candidate events (``tables/dcir_events.jsonl``). A candidate is a confirmed CC segment
   whose duration matches `pulse_s` within tolerance, optionally preceded and/or followed by a
   rest whose duration matches `rest_before_s` / `rest_after_s`. A nominal rest of 0 s means
   "no requirement". Durations are measured from the first sample of a segment to the first
   sample of the next segment.
2. The user reviews each candidate with `review_event` (Accepted or Rejected). Acceptance is
   a scientific decision and is never automatic; a candidate without a preceding sample
   cannot be accepted.
3. `measure_automatic_from_generation` measures an Accepted automatic event; manual events
   are created with `create_manual_event` from four values I0, V0, I1, V1.
4. `publish_dcir_events` stores reviewed events and measurements immutably.

Automatic measurement uses the real sample immediately before the pulse as reference (I_ref,
V_ref), the current of the third pulse sample (I_3, or the real pulse sample nearest to an
optional `current_time_s`) and the voltage of the real pulse sample nearest to
`requested_time_s` after the reference sample (default 10 s; ties take the later sample,
requests beyond the pulse use its last sample with a warning). The time origin is the
reference sample V0 because it is the last recorded instant before the current step; with a
sampling interval dt the first pulse sample is therefore at about +dt, not at 0.
Then ``R = |(V_meas - V_ref) / (I_3 - I_ref)|`` in ohm; signed ΔV and ΔI are kept. A zero
current difference makes R unavailable. An optional state of charge is estimated as
``f = Q1 / (Q1 + Q2 - Q_pulse)`` from the flanking half-cycles (``100 f`` for charge,
``100 (1 - f)`` for discharge) and is null whenever this mapping is ambiguous.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from .contracts import (
    AnalysisDescriptor,
    AvailabilityDescriptor,
    DependencyRef,
    JsonValue,
    OperationId,
    OutputDescriptor,
    ParameterDescriptor,
    ParameterGroup,
    ParameterLevel,
    ResultRef,
    WarningDescriptor,
)
from .diagnostics import Diagnostic, DiagnosticCode, DiagnosticKind
from .identity import canonical_parameter_bytes, hash_dependencies, hash_parameters
from .progress import CancelToken, ProgressEvent, ProgressSink, check_cancelled, report_progress
from .storage import GenerationManifest, ResultStore, open_result


class ReviewState(StrEnum):
    """Review state of a DCIR event: pending, accepted or rejected."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class EventSource(StrEnum):
    """Origin of a DCIR event: automatic detection or manual entry."""

    AUTO = "auto"
    MANUAL = "manual"


class PulsePolarity(StrEnum):
    """Pulse current sign: charge (positive) or discharge (negative).

    `BOTH` is only used as a detection filter meaning "either polarity".
    """

    BOTH = "both"
    CHARGE = "charge"
    DISCHARGE = "discharge"


class ValueProvenance(StrEnum):
    """Whether a manual DCIR value was typed or taken from a real sample."""

    TYPED = "typed"
    SAMPLE = "sample"


@dataclass(frozen=True, slots=True)
class DcirDetectionParameters:
    """Pulse-detection criteria.

    Attributes:
        rest_before_s: Nominal rest before the pulse in s (0 disables the check). Default 300.
        pulse_s: Nominal pulse duration in s (> 0). Default 30.
        rest_after_s: Nominal rest after the pulse in s (0 disables the check). Default 300.
        rest_before_tolerance_pct, pulse_tolerance_pct, rest_after_tolerance_pct: Inclusive
            relative tolerances in percent. Default 20 (e.g. 30 s accepts 24-36 s).
        polarity: `PulsePolarity` filter. Default `BOTH`.

    Defaults are starting points, not universal criteria: choose values matching the experimental
    protocol.
    """

    rest_before_s: float = 300.0
    pulse_s: float = 30.0
    rest_after_s: float = 300.0
    rest_before_tolerance_pct: float = 20.0
    pulse_tolerance_pct: float = 20.0
    rest_after_tolerance_pct: float = 20.0
    polarity: PulsePolarity = PulsePolarity.BOTH

    def validate(self) -> None:
        """Raise ValueError for negative values or a zero pulse."""
        values = (
            self.rest_before_s,
            self.pulse_s,
            self.rest_after_s,
            self.rest_before_tolerance_pct,
            self.pulse_tolerance_pct,
            self.rest_after_tolerance_pct,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("DCIR durations and tolerances must be finite and nonnegative")
        if self.pulse_s <= 0:
            raise ValueError("pulse_s must be positive")

    def as_dict(self) -> dict[str, Any]:
        """Return the parameters as a JSON-compatible dictionary."""
        return {
            "rest_before_s": self.rest_before_s,
            "pulse_s": self.pulse_s,
            "rest_after_s": self.rest_after_s,
            "rest_before_tolerance_pct": self.rest_before_tolerance_pct,
            "pulse_tolerance_pct": self.pulse_tolerance_pct,
            "rest_after_tolerance_pct": self.rest_after_tolerance_pct,
            "polarity": self.polarity.value,
        }


@dataclass(frozen=True, slots=True)
class SegmentRecord:
    """One contiguous step segment, as stored in ``indexes/segments.parquet``.

    `end_time_s` is the first time of the following segment (or the last time for a final rest),
    and `step` uses `Step` codes; `polarity` is the current sign.
    """

    segment_id: int
    first_row: int
    last_row: int
    first_time_s: float
    end_time_s: float | None
    cycle: int
    half_cycle: int
    step: int
    polarity: int

    @property
    def duration_s(self) -> float | None:
        """Segment duration in s, or None when it cannot be determined."""
        return None if self.end_time_s is None else self.end_time_s - self.first_time_s


@dataclass(frozen=True, slots=True)
class SampleRef:
    """A real measured sample identified by its canonical source row.

    Optional `cycle`, `half_cycle` and `elaboration_fingerprint` record where it was read from.
    """

    source_row: int
    time_s: float
    current_a: float
    voltage_v: float
    cycle: int | None = None
    half_cycle: int | None = None
    elaboration_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class DcirEvent:
    """A DCIR candidate or reviewed event.

    Attributes:
        event_id: Stable identifier (derived from the elaboration and pulse rows).
        source: `EventSource`.
        state: `ReviewState`.
        acceptance_order: Order of acceptance (Accepted events only).
        event_time_s: Pulse start time in s.
        cycle, half_cycle: Cycle context of the pulse.
        polarity: Pulse `PulsePolarity`.
        pulse_segment_id, pulse_first_row, pulse_last_row, pulse_duration_s: Pulse segment.
        rest_before_*, rest_after_*: Neighbouring rest segments, when present.
        pre_pulse_row: Source row of the reference sample before the pulse.
        correction_start_row, correction_end_row: Manual cycle-correction span, if any.

    Records loaded from ``dcir_events.jsonl`` can be rebuilt with ``DcirEvent(**{**row, "source":
    EventSource(row["source"]), ...})`` converting the enum fields.
    """

    event_id: str
    source: EventSource
    state: ReviewState
    acceptance_order: int | None
    event_time_s: float | None
    cycle: int | None
    half_cycle: int | None
    polarity: PulsePolarity | None
    pulse_segment_id: int | None = None
    pulse_first_row: int | None = None
    pulse_last_row: int | None = None
    pulse_duration_s: float | None = None
    rest_before_segment_id: int | None = None
    rest_before_duration_s: float | None = None
    rest_after_segment_id: int | None = None
    rest_after_duration_s: float | None = None
    pre_pulse_row: int | None = None
    correction_start_row: int | None = None
    correction_end_row: int | None = None


@dataclass(frozen=True, slots=True)
class DcirMeasurement:
    """Measured DCIR of one event.

    Attributes:
        event: The measured `DcirEvent`.
        display_label: Optional label such as ``"DCIR1"`` (see `event_labels`).
        requested_time_s, actual_time_s, time_deviation_s: Requested and actual voltage sampling
            time after the reference sample V0, and their difference (automatic events).
        i_ref_a, v_ref_v: Reference current and voltage.
        i_meas_a, v_meas_v: Measurement current and voltage.
        delta_i_a, delta_v_v: Signed differences (None if not representable).
        resistance_ohm: ``|delta_v / delta_i|`` in ohm, or None when unavailable.
        soc_pct: Optional capacity-based state of charge in percent.
        diagnostics: Warnings and reasons for unavailable values.
        reference_sample, current_sample, voltage_sample: Real samples used (automatic events).
        manual_values: The four manual values I0, V0, I1, V1 (manual events).
        soc_context: Capacities used for the SOC estimate.
        boundary_samples: Two samples bounding a manual cycle-correction span.
        current_requested_time_s: Requested current sampling time after V0 (automatic events);
            None means the third pulse sample was used.
    """

    event: DcirEvent
    display_label: str | None
    requested_time_s: float | None
    actual_time_s: float | None
    time_deviation_s: float | None
    i_ref_a: float
    v_ref_v: float
    i_meas_a: float
    v_meas_v: float
    delta_i_a: float | None
    delta_v_v: float | None
    resistance_ohm: float | None
    soc_pct: float | None
    diagnostics: tuple[Diagnostic, ...] = ()
    reference_sample: SampleRef | None = None
    current_sample: SampleRef | None = None
    voltage_sample: SampleRef | None = None
    manual_values: tuple[ManualValue, ManualValue, ManualValue, ManualValue] | None = None
    soc_context: SocContext | None = None
    boundary_samples: tuple[SampleRef, SampleRef] | None = None
    current_requested_time_s: float | None = None


@dataclass(frozen=True, slots=True)
class ManualValue:
    """A manually entered DCIR value, typed or taken from a real sample.

    With ``provenance=SAMPLE`` the `sample` must be given and `value` must equal its current or
    voltage; typed values carry no sample.
    """

    value: float
    provenance: ValueProvenance = ValueProvenance.TYPED
    sample: SampleRef | None = None

    def validate(self) -> None:
        """Raise ValueError for non-finite values or inconsistent provenance."""
        if not math.isfinite(self.value):
            raise ValueError("Manual DCIR values must be finite")
        if self.provenance is ValueProvenance.SAMPLE:
            if self.sample is None:
                raise ValueError("Sample provenance requires a real sample reference")
        elif self.sample is not None:
            raise ValueError("Typed values cannot imply a sample reference")


@dataclass(frozen=True, slots=True)
class SocContext:
    """Capacities used for the optional SOC estimate.

    `q1_ah` and `q2_ah` are the capacities of the half-cycles before and after the pulse (same
    direction), `q_pulse_ah` the capacity of the pulse half-cycle and `direction` the flanking
    direction that determines whether SOC is ``100 f`` or ``100 (1 - f)``.
    """

    q1_ah: float
    q2_ah: float
    q_pulse_ah: float
    direction: PulsePolarity
    q1_half_cycle: int | None = None
    q2_half_cycle: int | None = None
    pulse_half_cycle: int | None = None
    elaboration_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class DcirPublicationRequest:
    """Parameters of `publish_dcir_events`.

    Attributes:
        caller_root, sample_id, job_id, generation_id: Destination in the result store.
        elaboration_fingerprint: Fingerprint of the elaboration the events belong to.
        events: Events to publish (unique IDs).
        operation: `OperationId.DETECT_DCIR` or `OperationId.MEASURE_DCIR`.
        parameters: JSON-compatible parameters to record (e.g. requested time).
        measurements: Measurements; for `MEASURE_DCIR` exactly one per Accepted event.
        elaboration_path: Elaboration directory; required when measurements reference samples,
            which are then re-verified against the stored data.
    """

    caller_root: Path
    sample_id: str
    job_id: str
    generation_id: str
    elaboration_fingerprint: str
    events: tuple[DcirEvent, ...]
    operation: OperationId
    parameters: dict[str, Any]
    measurements: tuple[DcirMeasurement, ...] = ()
    elaboration_path: Path | None = None

    def validate(self) -> None:
        """Check operation, IDs and event/measurement consistency."""
        if self.operation not in (OperationId.DETECT_DCIR, OperationId.MEASURE_DCIR):
            raise ValueError("DCIR publication operation must be detect_dcir or measure_dcir")
        if len(self.elaboration_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.elaboration_fingerprint
        ):
            raise ValueError("elaboration_fingerprint must be lowercase SHA-256 hex")
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("Published DCIR event IDs must be unique")
        measurement_ids = [value.event.event_id for value in self.measurements]
        if len(measurement_ids) != len(set(measurement_ids)):
            raise ValueError("Published DCIR measurement event IDs must be unique")
        events_by_id = {event.event_id: event for event in self.events}
        if any(value.event.event_id not in events_by_id for value in self.measurements):
            raise ValueError("Every measurement must reference a published event")
        if any(events_by_id[value.event.event_id] != value.event for value in self.measurements):
            raise ValueError("Measurement event record must equal the published event record")
        if self.operation is OperationId.MEASURE_DCIR:
            accepted_ids = {
                event.event_id for event in self.events if event.state is ReviewState.ACCEPTED
            }
            if set(measurement_ids) != accepted_ids:
                raise ValueError("Every Accepted event requires exactly one complete measurement")
            if any(
                value.event.source is EventSource.MANUAL and value.manual_values is None
                for value in self.measurements
            ):
                raise ValueError("Accepted manual measurements require all four manual values")
        needs_m2 = any(
            value.event.source is EventSource.AUTO
            or (
                value.manual_values is not None
                and any(item.sample is not None for item in value.manual_values)
            )
            or value.boundary_samples is not None
            for value in self.measurements
        )
        if needs_m2 and self.elaboration_path is None:
            raise ValueError("Scientific DCIR measurements require an immutable elaboration result")


@dataclass(frozen=True, slots=True)
class DcirResult:
    """Published DCIR result.

    Attributes:
        ref: Reference to the DCIR generation.
        events: Number of event records.
        fingerprint: Fingerprint of the published content.
        accepted_fingerprint: Fingerprint of the Accepted events only (see
            `accepted_event_fingerprint`).
    """

    ref: ResultRef
    events: int
    fingerprint: str
    accepted_fingerprint: str


@dataclass(frozen=True, slots=True)
class DcirDetectionRequest:
    """Parameters of `detect_dcir`.

    Attributes:
        elaboration_path, elaboration_fingerprint: Elaboration result to scan.
        caller_root, sample_id, job_id, generation_id: Destination in the result store.
        parameters: `DcirDetectionParameters`.
        existing_events: Previously reviewed events; Accepted ones are kept and never
            duplicated when detection is repeated.
        batch_size: Rows per streaming batch.
    """

    elaboration_path: Path
    elaboration_fingerprint: str
    caller_root: Path
    sample_id: str
    job_id: str
    generation_id: str
    parameters: DcirDetectionParameters = DcirDetectionParameters()
    existing_events: tuple[DcirEvent, ...] = ()
    batch_size: int = 65_536

    def validate(self) -> None:
        """Validate parameters, batch size and fingerprint."""
        self.parameters.validate()
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len(self.elaboration_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.elaboration_fingerprint
        ):
            raise ValueError("elaboration_fingerprint must be lowercase SHA-256 hex")


def _stable_id(domain: bytes, values: dict[str, Any]) -> str:
    """Return a SHA-256 identifier of a domain and canonical values."""
    return hashlib.sha256(domain + b"\0" + canonical_parameter_bytes(values)).hexdigest()


def _matches(value: float | None, nominal: float, tolerance_pct: float) -> bool:
    """Return True if `value` lies in the inclusive tolerance band around `nominal`."""
    if value is None:
        return False
    lower = nominal * (1.0 - tolerance_pct / 100.0)
    upper = nominal * (1.0 + tolerance_pct / 100.0)
    return lower <= value <= upper


def detect_candidates(
    segments: tuple[SegmentRecord, ...],
    parameters: DcirDetectionParameters | None = None,
    *,
    elaboration_fingerprint: str = "",
    existing_events: tuple[DcirEvent, ...] = (),
) -> tuple[DcirEvent, ...]:
    """Find DCIR candidates in an ordered sequence of segments (pure).

    Returns Accepted events from `existing_events` followed by new Pending candidates. Each CC
    segment is evaluated once; a rest may serve as rest-after of one pulse and rest-before of the
    next. A CC segment at the very end of the data is not eligible (its duration is unknown).
    """
    parameters = parameters or DcirDetectionParameters()
    parameters.validate()
    accepted = {
        event.event_id: event for event in existing_events if event.state is ReviewState.ACCEPTED
    }
    output: list[DcirEvent] = list(accepted.values())
    for index, pulse in enumerate(segments):
        if pulse.step != 1 or pulse.duration_s is None:
            continue
        polarity = PulsePolarity.CHARGE if pulse.polarity > 0 else PulsePolarity.DISCHARGE
        if pulse.polarity == 0 or (
            parameters.polarity is not PulsePolarity.BOTH and parameters.polarity is not polarity
        ):
            continue
        if not _matches(pulse.duration_s, parameters.pulse_s, parameters.pulse_tolerance_pct):
            continue
        before = segments[index - 1] if index else None
        after = segments[index + 1] if index + 1 < len(segments) else None
        if parameters.rest_before_s > 0 and (
            before is None
            or before.step != 0
            or not _matches(
                before.duration_s,
                parameters.rest_before_s,
                parameters.rest_before_tolerance_pct,
            )
        ):
            continue
        if parameters.rest_after_s > 0 and (
            after is None
            or after.step != 0
            or not _matches(
                after.duration_s,
                parameters.rest_after_s,
                parameters.rest_after_tolerance_pct,
            )
        ):
            continue
        event_id = _stable_id(
            b"gregorithm-dcir-auto-event-v1",
            {
                "elaboration": elaboration_fingerprint,
                "segment_id": pulse.segment_id,
                "first_row": pulse.first_row,
                "last_row": pulse.last_row,
            },
        )
        if event_id in accepted:
            continue
        output.append(
            DcirEvent(
                event_id=event_id,
                source=EventSource.AUTO,
                state=ReviewState.PENDING,
                acceptance_order=None,
                event_time_s=pulse.first_time_s,
                cycle=pulse.cycle,
                half_cycle=pulse.half_cycle,
                polarity=polarity,
                pulse_segment_id=pulse.segment_id,
                pulse_first_row=pulse.first_row,
                pulse_last_row=pulse.last_row,
                pulse_duration_s=pulse.duration_s,
                rest_before_segment_id=before.segment_id if before and before.step == 0 else None,
                rest_before_duration_s=before.duration_s if before and before.step == 0 else None,
                rest_after_segment_id=after.segment_id if after and after.step == 0 else None,
                rest_after_duration_s=after.duration_s if after and after.step == 0 else None,
                pre_pulse_row=before.last_row if before is not None else None,
            )
        )
    return tuple(output)


def _iter_segment_dicts(path: Path, batch_size: int) -> Iterator[dict[str, Any]]:
    """Yield segment index rows as dictionaries, one batch at a time."""
    parquet = pq.ParquetFile(path)
    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            yield from pa.Table.from_batches((batch,)).to_pylist()
    finally:
        parquet.close()


def _segment_records(path: Path, batch_size: int) -> Iterator[SegmentRecord]:
    """Yield `SegmentRecord` objects with end times from the next segment."""
    source = _iter_segment_dicts(path, batch_size)
    current = next(source, None)
    while current is not None:
        following = next(source, None)
        end_time = (
            float(following["first_time_s"])
            if following is not None
            else float(current["last_time_s"])
            if int(current["step"]) == 0
            else None
        )
        yield SegmentRecord(
            int(current["segment_id"]),
            int(current["first_row"]),
            int(current["last_row"]),
            float(current["first_time_s"]),
            end_time,
            int(current["cycle"]),
            int(current["half_cycle"]),
            int(current["step"]),
            int(current["polarity"]),
        )
        current = following


def detect_dcir(
    request: DcirDetectionRequest,
    *,
    progress: ProgressSink | None = None,
    cancel: CancelToken | None = None,
    fault_hook: Any | None = None,
) -> DcirResult:
    """Detect DCIR pulse candidates in an elaboration result and publish them.

    Args:
        request: `DcirDetectionRequest`.
        progress: Optional progress sink (unit ``segments``).
        cancel: Optional cancellation token.
        fault_hook: Testing hook; leave it None.

    Returns:
        `DcirResult`; the candidates are JSON lines in
        ``result.ref.path / "tables" / "dcir_events.jsonl"``.

    Only the compact segment index is scanned, never the full point table.
    """
    request.validate()
    root = request.elaboration_path.resolve(strict=True)
    info = open_result(root)
    if info.operation is not OperationId.INGEST_AND_ELABORATE:
        raise ValueError("DCIR detection requires an immutable elaboration result")
    manifest = GenerationManifest.from_dict(
        json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    )
    if manifest.content_hash != request.elaboration_fingerprint:
        raise ValueError("elaboration_fingerprint does not match the elaboration manifest")
    segment_path = root / "indexes" / "segments.parquet"
    total = pq.ParquetFile(segment_path).metadata.num_rows
    report_progress(
        progress, ProgressEvent(OperationId.DETECT_DCIR, "scan-segments", 0, total, "segments")
    )
    store = ResultStore(request.caller_root)
    staging = store.begin_job(request.sample_id, request.job_id, request.generation_id)
    accepted = tuple(
        event for event in request.existing_events if event.state is ReviewState.ACCEPTED
    )
    accepted_ids = {event.event_id for event in accepted}
    events_path = staging.path / "tables" / "dcir_events.jsonl"
    events_path.parent.mkdir(parents=True)
    digest = hashlib.sha256(b"gregorithm-dcir-detection-events-v1\0")
    event_count = 0
    try:
        with events_path.open("wb") as output:
            for event in accepted:
                encoded = _event_json_line(event)
                output.write(encoded)
                digest.update(encoded)
                event_count += 1
            iterator = _segment_records(segment_path, request.batch_size)
            previous: SegmentRecord | None = None
            current = next(iterator, None)
            completed = 0
            while current is not None:
                check_cancelled(cancel, OperationId.DETECT_DCIR.value)
                following = next(iterator, None)
                window = tuple(item for item in (previous, current, following) if item is not None)
                for candidate in detect_candidates(
                    window,
                    request.parameters,
                    elaboration_fingerprint=request.elaboration_fingerprint,
                ):
                    if (
                        candidate.pulse_segment_id == current.segment_id
                        and candidate.event_id not in accepted_ids
                    ):
                        encoded = _event_json_line(candidate)
                        output.write(encoded)
                        digest.update(encoded)
                        event_count += 1
                completed += 1
                report_progress(
                    progress,
                    ProgressEvent(
                        OperationId.DETECT_DCIR,
                        "scan-segments",
                        completed,
                        total,
                        "segments",
                    ),
                )
                previous, current = current, following
            output.flush()
            os.fsync(output.fileno())
        check_cancelled(cancel, OperationId.DETECT_DCIR.value)
        parameters = request.parameters.as_dict()
        fingerprint = hashlib.sha256(
            b"gregorithm-dcir-detect-result-v1\0"
            + request.elaboration_fingerprint.encode()
            + b"\0"
            + canonical_parameter_bytes(parameters)
            + b"\0"
            + digest.digest()
        ).hexdigest()
        accepted_fingerprint = accepted_event_fingerprint(accepted)
        staging.write_bytes(
            "reports/dcir.json",
            (
                json.dumps(
                    {
                        "schema_version": 1,
                        "event_records": event_count,
                        "accepted_records": len(accepted),
                        "segment_records_scanned": completed,
                        "original_source_scans": 0,
                        "accepted_event_fingerprint": accepted_fingerprint,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode(),
        )
        dependency = DependencyRef("elaboration", request.elaboration_fingerprint)
        manifest = staging.build_manifest(
            operation=OperationId.DETECT_DCIR,
            parameter_hash=hash_parameters(parameters),
            dependency_hash=hash_dependencies((dependency,)),
            dependencies=(dependency,),
            content_hash=fingerprint,
            row_counts={"dcir_events": event_count, "segments_scanned": completed},
            logical_schemas={
                "tables/dcir_events.jsonl": "gregorithm.dcir_events.v1",
                "reports/dcir.json": "gregorithm.dcir_report.v1",
            },
        )
        ref = store.publish(staging, manifest, cancel=cancel, fault_hook=fault_hook)
        return DcirResult(ref, event_count, fingerprint, accepted_fingerprint)
    except Exception:
        staging.close()
        raise


def review_event(
    event: DcirEvent, state: ReviewState, acceptance_order: int | None = None
) -> DcirEvent:
    """Return a copy of `event` with a new review state.

    Accepting requires a non-negative `acceptance_order` and, for automatic events, a real pre-
    pulse reference sample (otherwise ValueError with `DCIR_MISSING_PRE_PULSE_REFERENCE`). Manual
    events cannot be set back to Pending. Rejecting clears the acceptance order.
    """
    if event.source is EventSource.MANUAL and state is ReviewState.PENDING:
        raise ValueError("Manual events are Accepted on creation")
    if state is ReviewState.ACCEPTED:
        if event.source is EventSource.AUTO and event.pre_pulse_row is None:
            raise ValueError(DiagnosticCode.DCIR_MISSING_PRE_PULSE_REFERENCE.value)
        if acceptance_order is None or acceptance_order < 0:
            raise ValueError("Accepted events require a nonnegative acceptance_order")
    else:
        acceptance_order = None
    return replace(event, state=state, acceptance_order=acceptance_order)


def event_labels(events: tuple[DcirEvent, ...]) -> dict[str, str]:
    """Return chronological display labels ``DCIR1``, ``DCIR2``, ... for Accepted events.

    Labels are for display only and do not affect any fingerprint.
    """
    chronological = sorted(
        (
            event
            for event in events
            if event.state is ReviewState.ACCEPTED and event.event_time_s is not None
        ),
        key=lambda event: (event.event_time_s, event.acceptance_order or 0),
    )
    return {event.event_id: f"DCIR{index}" for index, event in enumerate(chronological, 1)}


def _representable_fraction(value: Fraction) -> float | None:
    """Round an exact rational to a finite float, or return None."""
    try:
        result = float(value)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(result) or (result == 0.0 and value != 0):
        return None
    return result


def _stable_resistance(
    i0: float, v0: float, i1: float, v1: float, event_id: str
) -> tuple[float | None, float | None, float | None, Diagnostic | None]:
    """Compute signed deltas and resistance exactly, with diagnostics."""
    exact_di = Fraction.from_float(i1) - Fraction.from_float(i0)
    exact_dv = Fraction.from_float(v1) - Fraction.from_float(v0)
    delta_i = _representable_fraction(exact_di)
    delta_v = _representable_fraction(exact_dv)
    if exact_di == 0:
        return (
            delta_i,
            delta_v,
            None,
            Diagnostic(
                DiagnosticCode.DCIR_ZERO_CURRENT_DIFFERENCE,
                DiagnosticKind.UNAVAILABLE,
                "Zero current difference.",
                operation=OperationId.MEASURE_DCIR.value,
                event_id=event_id,
            ),
        )
    resistance = _representable_fraction(abs(exact_dv / exact_di))
    if delta_i is None or delta_v is None or resistance is None:
        fields = [
            name
            for name, value in (
                ("delta_i_a", delta_i),
                ("delta_v_v", delta_v),
                ("resistance_ohm", resistance),
            )
            if value is None
        ]
        return (
            delta_i,
            delta_v,
            resistance,
            Diagnostic(
                DiagnosticCode.NUMERIC_RESULT_UNREPRESENTABLE,
                DiagnosticKind.UNAVAILABLE,
                "Binary64 cannot represent: " + ", ".join(fields) + ".",
                operation=OperationId.MEASURE_DCIR.value,
                event_id=event_id,
            ),
        )
    return delta_i, delta_v, resistance, None


def _soc(context: SocContext | None, event_id: str) -> tuple[float | None, Diagnostic | None]:
    """Compute the optional SOC from a context, or return None with a diagnostic."""
    if context is None:
        return None, Diagnostic(
            DiagnosticCode.DCIR_SOC_UNAVAILABLE,
            DiagnosticKind.UNAVAILABLE,
            "Capacity context is unavailable or ambiguous.",
            operation=OperationId.MEASURE_DCIR.value,
            event_id=event_id,
        )
    if not all(
        math.isfinite(value) for value in (context.q1_ah, context.q2_ah, context.q_pulse_ah)
    ):
        return None, Diagnostic(
            DiagnosticCode.DCIR_SOC_UNAVAILABLE,
            DiagnosticKind.UNAVAILABLE,
            "Capacity SOC denominator is unavailable.",
            operation=OperationId.MEASURE_DCIR.value,
            event_id=event_id,
        )
    q1, q2, pulse = map(Fraction.from_float, (context.q1_ah, context.q2_ah, context.q_pulse_ah))
    denominator = q1 + q2 - pulse
    if denominator == 0:
        return None, Diagnostic(
            DiagnosticCode.DCIR_SOC_UNAVAILABLE,
            DiagnosticKind.UNAVAILABLE,
            "Capacity SOC denominator is unavailable.",
            operation=OperationId.MEASURE_DCIR.value,
            event_id=event_id,
        )
    fraction = q1 / denominator
    exact = 100 * (1 - fraction if context.direction is PulsePolarity.DISCHARGE else fraction)
    value = _representable_fraction(exact)
    if value is None:
        return None, Diagnostic(
            DiagnosticCode.DCIR_SOC_UNAVAILABLE,
            DiagnosticKind.UNAVAILABLE,
            "Capacity SOC result is not finite.",
            operation=OperationId.MEASURE_DCIR.value,
            event_id=event_id,
        )
    return value, None


def _check_current_time(current_time_s: float | None) -> None:
    """Raise ValueError unless the optional current sampling time is finite and nonnegative."""
    if current_time_s is not None and (not math.isfinite(current_time_s) or current_time_s < 0):
        raise ValueError("current_time_s must be finite and nonnegative")


def _nearest_sample(samples: tuple[SampleRef, ...], target: float) -> SampleRef:
    """Return the real sample nearest to `target` (absolute time); ties choose the later one."""
    return min(samples, key=lambda sample: (abs(sample.time_s - target), -sample.time_s))


def measure_automatic(
    event: DcirEvent,
    previous: SampleRef,
    pulse_samples: tuple[SampleRef, ...],
    *,
    requested_time_s: float = 10.0,
    current_time_s: float | None = None,
    label: str | None = None,
    soc_context: SocContext | None = None,
) -> DcirMeasurement:
    """Measure DCIR of an Accepted automatic event from supplied samples (pure).

    `previous` must be the recorded pre-pulse sample and `pulse_samples` the pulse samples in
    time order (at least three; first and last must match the event). The current is the third
    pulse sample, or, when `current_time_s` is given, the supplied pulse sample nearest to that
    time after V0 (same tie and beyond-the-pulse rules as the voltage). Prefer
    `measure_automatic_from_generation`, which reads and verifies the samples itself.
    """
    if event.source is not EventSource.AUTO or event.state is not ReviewState.ACCEPTED:
        raise ValueError("Automatic measurement requires an Accepted automatic event")
    if not math.isfinite(requested_time_s) or requested_time_s < 0:
        raise ValueError("requested_time_s must be finite and nonnegative")
    _check_current_time(current_time_s)
    if len(pulse_samples) < 3:
        raise ValueError("A confirmed CC pulse must have at least three real samples")
    if event.pre_pulse_row != previous.source_row:
        raise ValueError("Automatic baseline is not the recorded preceding real sample")
    if (
        event.pulse_first_row != pulse_samples[0].source_row
        or event.pulse_last_row != pulse_samples[-1].source_row
    ):
        raise ValueError("Automatic pulse samples do not match the recorded CC portion")
    origin = previous.time_s  # time zero is the reference sample V0, the last one before the step
    target = origin + requested_time_s
    selected = _nearest_sample(pulse_samples, target)
    actual = selected.time_s - origin
    diagnostics: list[Diagnostic] = []
    if max(requested_time_s, current_time_s or 0.0) > pulse_samples[-1].time_s - origin:
        diagnostics.append(
            Diagnostic(
                DiagnosticCode.DCIR_TIME_AFTER_LAST_SAMPLE,
                DiagnosticKind.WARNING,
                "Requested time exceeds the observed pulse; the last real sample was used.",
                operation=OperationId.MEASURE_DCIR.value,
                event_id=event.event_id,
            )
        )
    current_sample = (
        pulse_samples[2]
        if current_time_s is None
        else _nearest_sample(pulse_samples, origin + current_time_s)
    )
    i3 = current_sample.current_a
    delta_i, delta_v, resistance, numeric_diagnostic = _stable_resistance(
        previous.current_a,
        previous.voltage_v,
        i3,
        selected.voltage_v,
        event.event_id,
    )
    if numeric_diagnostic is not None:
        diagnostics.append(numeric_diagnostic)
    soc, soc_diagnostic = _soc(soc_context, event.event_id)
    if soc_diagnostic is not None:
        diagnostics.append(soc_diagnostic)
    return DcirMeasurement(
        event,
        label,
        requested_time_s,
        actual,
        abs(requested_time_s - actual),
        previous.current_a,
        previous.voltage_v,
        i3,
        selected.voltage_v,
        delta_i,
        delta_v,
        resistance,
        soc,
        tuple(diagnostics),
        previous,
        current_sample,
        selected,
        soc_context=soc_context,
        current_requested_time_s=current_time_s,
    )


def create_manual_event(
    i0: ManualValue,
    v0: ManualValue,
    i1: ManualValue,
    v1: ManualValue,
    *,
    acceptance_order: int,
    event_time_s: float | None = None,
    boundary_samples: tuple[SampleRef, SampleRef] | None = None,
    nonce: str = "",
) -> DcirMeasurement:
    """Create an Accepted manual DCIR event and its measurement.

    Args:
        i0, v0: Reference current (A) and voltage (V).
        i1, v1: Measurement current (A) and voltage (V).
        acceptance_order: Non-negative acceptance order.
        event_time_s: Optional event time; taken from `v1` when it comes from a sample.
        boundary_samples: Optional two distinct samples bounding a cycle-correction span.
        nonce: Extra text to make the identifier unique for otherwise identical entries.

    Returns:
        `DcirMeasurement` with ``R = |(V1 - V0) / (I1 - I0)|``. Invalid boundaries are reported as
        a `DCIR_MANUAL_CORRECTION_BOUNDARY_INVALID` diagnostic and not used for correction.
    """
    for value in (i0, v0, i1, v1):
        value.validate()
    for value in (i0, i1):
        if value.sample is not None and value.value != value.sample.current_a:
            raise ValueError("Sample-backed current must equal the real sample current")
    for value in (v0, v1):
        if value.sample is not None and value.value != value.sample.voltage_v:
            raise ValueError("Sample-backed voltage must equal the real sample voltage")
    if acceptance_order < 0:
        raise ValueError("acceptance_order must be nonnegative")
    anchor = v1.sample if v1.provenance is ValueProvenance.SAMPLE else None
    if anchor is not None:
        event_time_s = anchor.time_s
    if event_time_s is not None and not math.isfinite(event_time_s):
        raise ValueError("event_time_s must be finite when supplied")
    start_row = end_row = None
    diagnostics: list[Diagnostic] = []
    if boundary_samples is not None:
        first, second = boundary_samples
        if first.source_row == second.source_row or first.time_s == second.time_s:
            diagnostics.append(
                Diagnostic(
                    DiagnosticCode.DCIR_MANUAL_CORRECTION_BOUNDARY_INVALID,
                    DiagnosticKind.ERROR,
                    "Manual correction boundaries must be two distinct real samples.",
                    operation=OperationId.MEASURE_DCIR.value,
                )
            )
        else:
            earlier, later = sorted(boundary_samples, key=lambda sample: sample.time_s)
            start_row, end_row = earlier.source_row, later.source_row
    event_id = _stable_id(
        b"gregorithm-dcir-manual-event-v1",
        {
            "acceptance_order": acceptance_order,
            "nonce": nonce,
            "event_time_s": event_time_s,
        },
    )
    event = DcirEvent(
        event_id,
        EventSource.MANUAL,
        ReviewState.ACCEPTED,
        acceptance_order,
        event_time_s,
        anchor.cycle if anchor else None,
        anchor.half_cycle if anchor else None,
        None,
        correction_start_row=start_row,
        correction_end_row=end_row,
    )
    delta_i, delta_v, resistance, numeric_diagnostic = _stable_resistance(
        i0.value, v0.value, i1.value, v1.value, event_id
    )
    if numeric_diagnostic is not None:
        diagnostics.insert(
            0,
            numeric_diagnostic,
        )
    return DcirMeasurement(
        event,
        None,
        None,
        None,
        None,
        i0.value,
        v0.value,
        i1.value,
        v1.value,
        delta_i,
        delta_v,
        resistance,
        None,
        tuple(diagnostics),
        manual_values=(i0, v0, i1, v1),
        boundary_samples=boundary_samples,
    )


_POINT_COLUMNS = ("source_row", "time_s", "current_a", "voltage_v", "cycle", "half_cycle", "step")


def _validated_m2_root(path: Path, fingerprint: str) -> Path:
    """Check that a path is an elaboration result with the given fingerprint."""
    root = path.resolve(strict=True)
    info = open_result(root)
    if info.operation is not OperationId.INGEST_AND_ELABORATE:
        raise ValueError("DCIR measurement requires an immutable elaboration result")
    manifest = GenerationManifest.from_dict(
        json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    )
    if manifest.content_hash != fingerprint:
        raise ValueError("elaboration_fingerprint does not match the elaboration manifest")
    return root


def _iter_m2_samples(
    root: Path, first_row: int, last_row: int, fingerprint: str
) -> Iterator[SampleRef]:
    """Yield elaborated samples whose source rows lie in an inclusive range.

    Rows are yielded in stored order. The range is selected with Arrow compute before any
    conversion to Python objects: a row group holds up to `PARQUET_ROW_GROUP_SIZE` rows, and
    converting it whole for a pulse of a few samples (or for a single reference row) made DCIR
    measurement and publication take seconds per event while holding the GIL, which froze the
    GUI's progress animation. The selected rows and values are identical to a full scan.
    """
    parquet = pq.ParquetFile(root / "points" / "elaborated.parquet")
    try:
        column_index = parquet.schema_arrow.get_field_index("source_row")
        for row_group in range(parquet.num_row_groups):
            metadata = parquet.metadata.row_group(row_group)
            statistics = metadata.column(column_index).statistics
            if statistics is not None and (
                int(statistics.max) < first_row or int(statistics.min) > last_row
            ):
                continue
            table = parquet.read_row_group(row_group, columns=list(_POINT_COLUMNS))
            rows = table.column("source_row")
            in_range = pc.and_(pc.greater_equal(rows, first_row), pc.less_equal(rows, last_row))
            for row in table.filter(in_range).to_pylist():
                source_row = int(row["source_row"])
                if first_row <= source_row <= last_row:
                    yield SampleRef(
                        source_row,
                        float(row["time_s"]),
                        float(row["current_a"]),
                        float(row["voltage_v"]),
                        int(row["cycle"]),
                        int(row["half_cycle"]),
                        fingerprint,
                    )
    finally:
        parquet.close()


def _m2_sample(root: Path, source_row: int, fingerprint: str) -> SampleRef:
    """Return the elaborated sample of one source row."""
    rows = tuple(_iter_m2_samples(root, source_row, source_row, fingerprint))
    if len(rows) != 1:
        raise ValueError(f"Elaboration source row {source_row} is missing or duplicated")
    return rows[0]


def _derive_soc_context(root: Path, fingerprint: str, event: DcirEvent) -> SocContext | None:
    """Derive the SOC capacities from the half-cycle and segment indexes."""
    if event.half_cycle is None or event.pulse_segment_id is None or event.half_cycle < 1:
        return None
    q1_half = event.half_cycle - 1
    pulse_half = event.half_cycle
    q2_half = event.half_cycle + 1
    half_rows = pq.read_table(
        root / "indexes" / "half_cycles.parquet",
        filters=[("half_cycle", "in", [q1_half, pulse_half, q2_half])],
    ).to_pylist()
    halves = {int(row["half_cycle"]): row for row in half_rows}
    if set(halves) != {q1_half, pulse_half, q2_half}:
        return None
    segment_rows = pq.read_table(
        root / "indexes" / "segments.parquet",
        filters=[("half_cycle", "in", [q1_half, pulse_half, q2_half])],
    ).to_pylist()

    def active_polarities(half_cycle: int) -> set[int]:
        """Return the active current signs of a half-cycle."""
        return {
            int(row["polarity"])
            for row in segment_rows
            if int(row["half_cycle"]) == half_cycle
            and int(row["step"]) != 0
            and int(row["polarity"]) != 0
        }

    q1_polarities = active_polarities(q1_half)
    q2_polarities = active_polarities(q2_half)
    pulse_segments = [
        row
        for row in segment_rows
        if int(row["half_cycle"]) == pulse_half and int(row["step"]) != 0
    ]
    if (
        len(q1_polarities) != 1
        or q1_polarities != q2_polarities
        or len(pulse_segments) != 1
        or int(pulse_segments[0]["segment_id"]) != event.pulse_segment_id
    ):
        return None
    flank_sign = next(iter(q1_polarities))
    pulse_sign = int(pulse_segments[0]["polarity"])
    if pulse_sign == 0:
        return None
    flank_field = "charge_capacity_ah" if flank_sign > 0 else "discharge_capacity_ah"
    pulse_field = "charge_capacity_ah" if pulse_sign > 0 else "discharge_capacity_ah"
    q1 = halves[q1_half][flank_field]
    q2 = halves[q2_half][flank_field]
    q_pulse = halves[pulse_half][pulse_field]
    if any(value is None or not math.isfinite(float(value)) for value in (q1, q2, q_pulse)):
        return None
    return SocContext(
        float(q1),
        float(q2),
        float(q_pulse),
        PulsePolarity.CHARGE if flank_sign > 0 else PulsePolarity.DISCHARGE,
        q1_half,
        q2_half,
        pulse_half,
        fingerprint,
    )


def _same_sample(actual: SampleRef, supplied: SampleRef) -> bool:
    """Return True if a supplied sample equals the stored sample."""
    return (
        actual.source_row == supplied.source_row
        and actual.time_s == supplied.time_s
        and actual.current_a == supplied.current_a
        and actual.voltage_v == supplied.voltage_v
        and actual.cycle == supplied.cycle
        and actual.half_cycle == supplied.half_cycle
        and supplied.elaboration_fingerprint in (None, actual.elaboration_fingerprint)
    )


def measure_automatic_from_generation(
    elaboration_path: Path,
    elaboration_fingerprint: str,
    event: DcirEvent,
    *,
    requested_time_s: float = 10.0,
    current_time_s: float | None = None,
    label: str | None = None,
    soc_context: SocContext | None = None,
) -> DcirMeasurement:
    """Measure DCIR of an Accepted automatic event.

    Reads the reference sample, the pulse samples and (when unambiguous) the SOC context from the
    elaboration result and verifies them against the event record.

    Args:
        elaboration_path, elaboration_fingerprint: Elaboration result of the event.
        event: Accepted automatic `DcirEvent`.
        requested_time_s: Voltage sampling time after the reference sample V0, in s (default 10).
        current_time_s: Optional current sampling time after V0, in s. None (default) uses the
            third pulse sample; otherwise the real pulse sample nearest to it is used.
        label: Optional display label.
        soc_context: Optional SOC context; it must match the derived one.

    Returns:
        `DcirMeasurement`.
    """
    if event.source is not EventSource.AUTO or event.state is not ReviewState.ACCEPTED:
        raise ValueError("Automatic measurement requires an Accepted automatic event")
    if event.pre_pulse_row is None or event.pulse_first_row is None or event.pulse_last_row is None:
        raise ValueError("Automatic event has incomplete elaboration row provenance")
    if not math.isfinite(requested_time_s) or requested_time_s < 0:
        raise ValueError("requested_time_s must be finite and nonnegative")
    _check_current_time(current_time_s)
    root = _validated_m2_root(elaboration_path, elaboration_fingerprint)
    segments = pq.read_table(
        root / "indexes" / "segments.parquet",
        filters=[("segment_id", "=", event.pulse_segment_id)],
    ).to_pylist()
    if len(segments) != 1:
        raise ValueError("Recorded CC portion is absent from the elaboration result")
    segment = segments[0]
    expected = (
        int(segment["first_row"]),
        int(segment["last_row"]),
        int(segment["cycle"]),
        int(segment["half_cycle"]),
        int(segment["step"]),
    )
    if expected != (
        event.pulse_first_row,
        event.pulse_last_row,
        event.cycle,
        event.half_cycle,
        1,
    ):
        raise ValueError("DCIR event does not identify the recorded CC portion")
    derived_soc_context = _derive_soc_context(root, elaboration_fingerprint, event)
    if soc_context is not None and soc_context != derived_soc_context:
        raise ValueError(
            "Supplied SOC context does not match the elaboration half-cycle and capacity context"
        )
    previous = _m2_sample(root, event.pre_pulse_row, elaboration_fingerprint)
    first_three: list[SampleRef] = []
    selected: SampleRef | None = None
    current: SampleRef | None = None
    last: SampleRef | None = None
    origin = previous.time_s  # time zero is the reference sample V0, the last one before the step
    target = origin + requested_time_s
    current_target = None if current_time_s is None else origin + current_time_s
    count = 0
    for sample in _iter_m2_samples(
        root, event.pulse_first_row, event.pulse_last_row, elaboration_fingerprint
    ):
        count += 1
        if len(first_three) < 3:
            first_three.append(sample)
        if selected is None or (abs(sample.time_s - target), -sample.time_s) < (
            abs(selected.time_s - target),
            -selected.time_s,
        ):
            selected = sample
        if current_target is not None and (
            current is None
            or (abs(sample.time_s - current_target), -sample.time_s)
            < (abs(current.time_s - current_target), -current.time_s)
        ):
            current = sample
        last = sample
    if count < 3 or selected is None or last is None:
        raise ValueError("Recorded CC portion has fewer than three real samples")
    if (
        first_three[0].source_row != event.pulse_first_row
        or last.source_row != event.pulse_last_row
    ):
        raise ValueError("Pulse membership is incomplete or out of order")
    # The pure helper is reused only after genuine elaborated samples have been selected.
    compact = (first_three[0], first_three[1], first_three[2], last)
    result = measure_automatic(
        event,
        previous,
        compact,
        requested_time_s=requested_time_s,
        label=label,
        soc_context=derived_soc_context,
    )
    # The nearest voltage sample need not be one of the compact selection samples.
    actual = selected.time_s - origin
    diagnostics = tuple(
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code is not DiagnosticCode.DCIR_TIME_AFTER_LAST_SAMPLE
    )
    if current is None:
        current = first_three[2]
    if max(requested_time_s, current_time_s or 0.0) > last.time_s - origin:
        diagnostics = (
            Diagnostic(
                DiagnosticCode.DCIR_TIME_AFTER_LAST_SAMPLE,
                DiagnosticKind.WARNING,
                "Requested time exceeds the observed pulse; the last real sample was used.",
                operation=OperationId.MEASURE_DCIR.value,
                event_id=event.event_id,
            ),
            *diagnostics,
        )
    delta_i, delta_v, resistance, numeric = _stable_resistance(
        previous.current_a,
        previous.voltage_v,
        current.current_a,
        selected.voltage_v,
        event.event_id,
    )
    diagnostics = tuple(
        diagnostic
        for diagnostic in diagnostics
        if diagnostic.code
        not in (
            DiagnosticCode.DCIR_ZERO_CURRENT_DIFFERENCE,
            DiagnosticCode.NUMERIC_RESULT_UNREPRESENTABLE,
        )
    )
    if numeric is not None:
        diagnostics = (numeric, *diagnostics)
    return replace(
        result,
        actual_time_s=actual,
        time_deviation_s=abs(requested_time_s - actual),
        i_meas_a=current.current_a,
        v_meas_v=selected.voltage_v,
        delta_i_a=delta_i,
        delta_v_v=delta_v,
        resistance_ohm=resistance,
        diagnostics=diagnostics,
        reference_sample=previous,
        current_sample=current,
        voltage_sample=selected,
        current_requested_time_s=current_time_s,
    )


def _verify_publication_measurements(request: DcirPublicationRequest) -> None:
    """Re-verify every sample-backed value against stored data."""
    if not request.measurements:
        return
    needs_m2 = any(
        value.event.source is EventSource.AUTO
        or (
            value.manual_values is not None
            and any(item.sample is not None for item in value.manual_values)
        )
        or value.boundary_samples is not None
        for value in request.measurements
    )
    root = None
    if needs_m2:
        assert request.elaboration_path is not None
        root = _validated_m2_root(request.elaboration_path, request.elaboration_fingerprint)
    for measurement in request.measurements:
        references = (
            measurement.reference_sample,
            measurement.current_sample,
            measurement.voltage_sample,
        )
        if measurement.event.source is EventSource.AUTO and any(
            value is None for value in references
        ):
            raise ValueError("Automatic measurement lacks resolved elaboration sample provenance")
        samples = [value for value in references if value is not None]
        if measurement.manual_values is not None:
            samples.extend(
                value.sample for value in measurement.manual_values if value.sample is not None
            )
        if measurement.boundary_samples is not None:
            samples.extend(measurement.boundary_samples)
        for supplied in samples:
            if root is None:
                raise ValueError("Sample-backed DCIR measurement lacks its elaboration result")
            actual = _m2_sample(root, supplied.source_row, request.elaboration_fingerprint)
            if not _same_sample(actual, supplied):
                raise ValueError(
                    "DCIR sample provenance does not match the immutable elaboration result"
                )
        if measurement.event.source is EventSource.AUTO:
            assert root is not None
            if measurement.requested_time_s is None:
                raise ValueError("Automatic measurement lacks its requested time")
            resolved = measure_automatic_from_generation(
                root,
                request.elaboration_fingerprint,
                measurement.event,
                requested_time_s=measurement.requested_time_s,
                current_time_s=measurement.current_requested_time_s,
                label=measurement.display_label,
            )
            fields = (
                "actual_time_s",
                "time_deviation_s",
                "i_ref_a",
                "v_ref_v",
                "i_meas_a",
                "v_meas_v",
                "delta_i_a",
                "delta_v_v",
                "resistance_ohm",
                "soc_pct",
                "soc_context",
                "reference_sample",
                "current_sample",
                "voltage_sample",
                "current_requested_time_s",
            )
            if any(getattr(measurement, name) != getattr(resolved, name) for name in fields):
                raise ValueError(
                    "Automatic DCIR measurement was not resolved from its elaboration result"
                )
            expected_soc_diagnostics = tuple(
                item
                for item in resolved.diagnostics
                if item.code is DiagnosticCode.DCIR_SOC_UNAVAILABLE
            )
            actual_soc_diagnostics = tuple(
                item
                for item in measurement.diagnostics
                if item.code is DiagnosticCode.DCIR_SOC_UNAVAILABLE
            )
            if actual_soc_diagnostics != expected_soc_diagnostics:
                raise ValueError(
                    "Automatic DCIR SOC diagnostics do not match the elaboration context"
                )
        elif measurement.manual_values is not None:
            i0, v0, i1, v1 = measurement.manual_values
            for value in measurement.manual_values:
                value.validate()
            for value in (i0, i1):
                if value.sample is not None and value.value != value.sample.current_a:
                    raise ValueError(
                        "Sample-backed manual current does not match its elaboration sample"
                    )
            for value in (v0, v1):
                if value.sample is not None and value.value != value.sample.voltage_v:
                    raise ValueError(
                        "Sample-backed manual voltage does not match its elaboration sample"
                    )
            if v1.sample is not None and (
                measurement.event.event_time_s,
                measurement.event.cycle,
                measurement.event.half_cycle,
            ) != (v1.sample.time_s, v1.sample.cycle, v1.sample.half_cycle):
                raise ValueError("Manual event context does not match its authenticated V1 sample")
            if v1.sample is None and (
                measurement.event.cycle is not None or measurement.event.half_cycle is not None
            ):
                raise ValueError("Typed V1 cannot establish a cycle or half-cycle")
            boundary_diagnostics: tuple[Diagnostic, ...] = ()
            if measurement.boundary_samples is None:
                if (
                    measurement.event.correction_start_row is not None
                    or measurement.event.correction_end_row is not None
                ):
                    raise ValueError(
                        "Manual correction rows require authenticated boundary samples"
                    )
            else:
                first, second = measurement.boundary_samples
                invalid_boundary = (
                    first.source_row == second.source_row or first.time_s == second.time_s
                )
                if invalid_boundary:
                    if (
                        measurement.event.correction_start_row is not None
                        or measurement.event.correction_end_row is not None
                    ):
                        raise ValueError(
                            "Invalid manual correction boundaries cannot publish a span"
                        )
                    boundary_diagnostics = (
                        Diagnostic(
                            DiagnosticCode.DCIR_MANUAL_CORRECTION_BOUNDARY_INVALID,
                            DiagnosticKind.ERROR,
                            "Manual correction boundaries must be two distinct real samples.",
                            operation=OperationId.MEASURE_DCIR.value,
                        ),
                    )
                else:
                    earlier, later = sorted(
                        measurement.boundary_samples, key=lambda item: item.time_s
                    )
                    if (
                        measurement.event.correction_start_row,
                        measurement.event.correction_end_row,
                    ) != (earlier.source_row, later.source_row):
                        raise ValueError(
                            "Manual correction rows do not match the authenticated ordered samples"
                        )
            delta_i, delta_v, resistance, numeric_diagnostic = _stable_resistance(
                i0.value, v0.value, i1.value, v1.value, measurement.event.event_id
            )
            expected_diagnostics = (
                *((numeric_diagnostic,) if numeric_diagnostic is not None else ()),
                *boundary_diagnostics,
            )
            if measurement.diagnostics != expected_diagnostics:
                raise ValueError("Manual DCIR diagnostics do not match the first causal result")
            if (
                measurement.i_ref_a,
                measurement.v_ref_v,
                measurement.i_meas_a,
                measurement.v_meas_v,
                measurement.delta_i_a,
                measurement.delta_v_v,
                measurement.resistance_ohm,
                measurement.soc_pct,
                measurement.requested_time_s,
                measurement.actual_time_s,
                measurement.time_deviation_s,
                measurement.soc_context,
                measurement.current_requested_time_s,
            ) != (
                i0.value,
                v0.value,
                i1.value,
                v1.value,
                delta_i,
                delta_v,
                resistance,
                None,
                None,
                None,
                None,
                None,
                None,
            ):
                raise ValueError(
                    "Manual DCIR result does not match its persisted four-value inputs"
                )


def _sample_dict(sample: SampleRef | None) -> dict[str, Any] | None:
    """Return a sample as a JSON-compatible dictionary (or None)."""
    if sample is None:
        return None
    return {name: getattr(sample, name) for name in sample.__dataclass_fields__}


def _manual_value_dict(value: ManualValue) -> dict[str, Any]:
    """Return a manual value as a JSON-compatible dictionary."""
    return {
        "value": value.value,
        "provenance": value.provenance.value,
        "sample": _sample_dict(value.sample),
    }


def _soc_context_dict(context: SocContext | None) -> dict[str, Any] | None:
    """Return a SOC context as a JSON-compatible dictionary (or None)."""
    if context is None:
        return None
    row = {name: getattr(context, name) for name in context.__dataclass_fields__}
    row["direction"] = context.direction.value
    return row


def accepted_event_fingerprint(
    events: tuple[DcirEvent, ...], measurements: tuple[DcirMeasurement, ...] = ()
) -> str:
    """Return the fingerprint of the Accepted events and their spans.

    Only scientifically relevant content is hashed (identity, time, pulse rows, correction spans,
    manual values and boundaries). Pending/Rejected events and display labels do not change it.
    """
    measurement_by_event = {value.event.event_id: value for value in measurements}
    accepted = [
        {
            "event_id": event.event_id,
            "source": event.source.value,
            "time": event.event_time_s,
            "pulse_first_row": event.pulse_first_row,
            "pulse_last_row": event.pulse_last_row,
            "correction_start_row": event.correction_start_row,
            "correction_end_row": event.correction_end_row,
            "manual_values": (
                [_manual_value_dict(value) for value in measurement.manual_values]
                if (measurement := measurement_by_event.get(event.event_id)) is not None
                and measurement.manual_values is not None
                else None
            ),
            "boundary_samples": (
                [_sample_dict(value) for value in measurement.boundary_samples]
                if measurement is not None and measurement.boundary_samples is not None
                else None
            ),
        }
        for event in sorted(events, key=lambda value: value.event_id)
        if event.state is ReviewState.ACCEPTED
    ]
    return hashlib.sha256(
        b"gregorithm-dcir-accepted-v1\0" + canonical_parameter_bytes(cast(JsonValue, accepted))
    ).hexdigest()


def _event_dict(event: DcirEvent) -> dict[str, Any]:
    """Return an event as a JSON-compatible dictionary."""
    row = {name: getattr(event, name) for name in event.__dataclass_fields__}
    row["source"] = event.source.value
    row["state"] = event.state.value
    row["polarity"] = event.polarity.value if event.polarity is not None else None
    return row


def _event_json_line(event: DcirEvent) -> bytes:
    """Return an event as one compact JSON line."""
    return (
        json.dumps(_event_dict(event), sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def serialize_events(events: tuple[DcirEvent, ...]) -> bytes:
    """Return the canonical JSON bytes stored for a sequence of events."""
    rows = [_event_dict(event) for event in events]
    return (
        json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _serialize_measurements(measurements: tuple[DcirMeasurement, ...]) -> bytes:
    """Return the canonical JSON bytes stored for measurements."""
    rows = [
        {
            "event_id": value.event.event_id,
            "display_label": value.display_label,
            "requested_time_s": value.requested_time_s,
            "actual_time_s": value.actual_time_s,
            "time_deviation_s": value.time_deviation_s,
            "i_ref_a": value.i_ref_a,
            "v_ref_v": value.v_ref_v,
            "i_meas_a": value.i_meas_a,
            "v_meas_v": value.v_meas_v,
            "delta_i_a": value.delta_i_a,
            "delta_v_v": value.delta_v_v,
            "resistance_ohm": value.resistance_ohm,
            "soc_pct": value.soc_pct,
            "soc_context": _soc_context_dict(value.soc_context),
            "reference_sample": _sample_dict(value.reference_sample),
            "current_sample": _sample_dict(value.current_sample),
            "voltage_sample": _sample_dict(value.voltage_sample),
            "manual_values": (
                {
                    name: _manual_value_dict(item)
                    for name, item in zip(
                        ("I0", "V0", "I1", "V1"), value.manual_values, strict=True
                    )
                }
                if value.manual_values is not None
                else None
            ),
            "boundary_samples": (
                [_sample_dict(item) for item in value.boundary_samples]
                if value.boundary_samples is not None
                else None
            ),
            "diagnostics": [diagnostic.to_dict() for diagnostic in value.diagnostics],
            # Written only when chosen, so results measured with the third sample keep their bytes.
            **(
                {"current_requested_time_s": value.current_requested_time_s}
                if value.current_requested_time_s is not None
                else {}
            ),
        }
        for value in measurements
    ]
    return (
        json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def publish_dcir_events(
    request: DcirPublicationRequest,
    *,
    progress: ProgressSink | None = None,
    cancel: CancelToken | None = None,
    fault_hook: Any | None = None,
) -> DcirResult:
    """Publish reviewed DCIR events and measurements immutably.

    Writes ``tables/dcir_events.json``, optional ``tables/dcir_measurements.json`` and
    ``reports/dcir.json``. Sample-backed values are re-verified against the elaboration result
    before publication. Returns a `DcirResult`.
    """
    request.validate()
    check_cancelled(cancel, request.operation.value)
    _verify_publication_measurements(request)
    accepted_fingerprint = accepted_event_fingerprint(request.events, request.measurements)
    all_bytes = serialize_events(request.events)
    measurement_bytes = _serialize_measurements(request.measurements)
    fingerprint = hashlib.sha256(
        b"gregorithm-dcir-result-v1\0"
        + request.operation.value.encode()
        + b"\0"
        + request.elaboration_fingerprint.encode()
        + b"\0"
        + canonical_parameter_bytes(request.parameters)
        + b"\0"
        + all_bytes
        + b"\0"
        + measurement_bytes
    ).hexdigest()
    dependency = DependencyRef("elaboration", request.elaboration_fingerprint)
    store = ResultStore(request.caller_root)
    staging = store.begin_job(request.sample_id, request.job_id, request.generation_id)
    try:
        report_progress(
            progress,
            ProgressEvent(request.operation, "write-events", 0, len(request.events), "events"),
        )
        staging.write_bytes("tables/dcir_events.json", all_bytes)
        if request.measurements:
            staging.write_bytes("tables/dcir_measurements.json", measurement_bytes)
        staging.write_bytes(
            "reports/dcir.json",
            (
                json.dumps(
                    {
                        "schema_version": 1,
                        "event_records": len(request.events),
                        "accepted_records": sum(
                            event.state is ReviewState.ACCEPTED for event in request.events
                        ),
                        "measurement_records": len(request.measurements),
                        "original_source_scans": 0,
                        "accepted_event_fingerprint": accepted_fingerprint,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode(),
        )
        check_cancelled(cancel, request.operation.value)
        manifest = staging.build_manifest(
            operation=request.operation,
            parameter_hash=hash_parameters(request.parameters),
            dependency_hash=hash_dependencies((dependency,)),
            dependencies=(dependency,),
            content_hash=fingerprint,
            row_counts={
                "dcir_events": len(request.events),
                "dcir_measurements": len(request.measurements),
            },
            logical_schemas={
                "tables/dcir_events.json": "gregorithm.dcir_events.v1",
                "reports/dcir.json": "gregorithm.dcir_report.v1",
                **(
                    {"tables/dcir_measurements.json": "gregorithm.dcir_measurements.v1"}
                    if request.measurements
                    else {}
                ),
            },
        )
        report_progress(
            progress,
            ProgressEvent(
                request.operation,
                "events-complete",
                len(request.events),
                len(request.events),
                "events",
            ),
        )
        ref = store.publish(staging, manifest, cancel=cancel, fault_hook=fault_hook)
        return DcirResult(ref, len(request.events), fingerprint, accepted_fingerprint)
    except Exception:
        staging.close()
        raise


def dcir_descriptor(operation: OperationId) -> AnalysisDescriptor:
    """Describe DCIR detection (`OperationId.DETECT_DCIR`) or measurement
    (`OperationId.MEASURE_DCIR`) for user interfaces.
    """
    if operation not in (OperationId.DETECT_DCIR, OperationId.MEASURE_DCIR):
        raise ValueError("DCIR descriptor supports detect or measure operations")
    detecting = operation is OperationId.DETECT_DCIR
    outputs = (
        OutputDescriptor(
            "event_id",
            "Stable event identity",
            "string",
            None,
            True,
            "dcir",
            AvailabilityDescriptor("always", "", "Available for every event."),
        ),
        OutputDescriptor(
            "resistance_ohm",
            "DCIR",
            "float64",
            "ohm",
            not detecting,
            "dcir",
            AvailabilityDescriptor(
                "accepted_measurement",
                DiagnosticCode.DCIR_ZERO_CURRENT_DIFFERENCE.value,
                "Available when the accepted event has a nonzero current difference.",
            ),
        ),
    )
    parameters = (
        ParameterDescriptor(
            "pulse_s" if detecting else "requested_time_s",
            "Pulse duration" if detecting else "Requested voltage time",
            "Nominal pulse duration."
            if detecting
            else "Voltage sample time relative to the reference sample V0.",
            "number",
            "s",
            30.0 if detecting else 10.0,
            0.0,
            None,
            None,
            ParameterLevel.BASIC,
            "dcir",
        ),
    )
    descriptor = AnalysisDescriptor(
        1,
        operation,
        "Detect DCIR" if detecting else "Measure DCIR",
        "DCIR pulse candidate detection." if detecting else "Automatic or manual DCIR measurement.",
        ("elaboration",),
        (ParameterGroup("dcir", "DCIR", 0),),
        parameters,
        outputs,
        (
            WarningDescriptor(
                DiagnosticCode.DCIR_TIME_AFTER_LAST_SAMPLE.value,
                "Requested time is after the last pulse sample.",
                ("event_id",),
            ),
        ),
        True,
        True,
        "segments" if detecting else "events",
    )
    descriptor.validate()
    return descriptor
