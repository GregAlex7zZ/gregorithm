"""Explicit cycle correction ("Recalculate Cycles") using Accepted DCIR pulses.

A DCIR pulse of opposite polarity can be mistaken for a new half-cycle. `recalculate_cycles`
excludes the row ranges of valid Accepted DCIR events from half-cycle segmentation and publishes
a new corrected generation; it is never applied implicitly. Every original sample is kept and its
measured contribution to capacity and energy is preserved. Step classification is unchanged,
cumulative quantities and Capacity are recomputed for the corrected half-cycles, DCIR resistance
values are unchanged (only their cycle context and optional SOC are refreshed) and an existing
dQ/dV table keeps its values with re-mapped cycle indices. Pending, Rejected and unbounded events
are ignored. Running with ``enabled=False`` restores the uncorrected interpretation.

The corrected generation contains ``points/elaborated.parquet``, ``indexes/*.parquet``,
``tables/capacity.parquet``, ``tables/capacity_diagnostics.parquet``,
``tables/dcir_events.json``, optional ``tables/dcir_measurements.json``,
``tables/dqdv.parquet`` and ``tables/dqdv_reconstruction.parquet`` (re-indexed copies of an
existing dQ/dV generation), and ``reports/cycle_correction.json``. It can be used as input to
`analyze_dqdv`, `read_window` and a further `recalculate_cycles`.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from . import __version__
from .capacity import (
    CapacityParameters,
    _finalize_rows,
    _write_base_rows,
    _write_cc_rows,
)
from .contracts import (
    AnalysisDescriptor,
    AvailabilityDescriptor,
    DependencyRef,
    OperationId,
    OutputDescriptor,
    ParameterDescriptor,
    ParameterGroup,
    ParameterLevel,
    ResultRef,
)
from .dcir import (
    DcirEvent,
    DcirMeasurement,
    EventSource,
    PulsePolarity,
    ReviewState,
    SocContext,
    _serialize_measurements,
    _soc,
    accepted_event_fingerprint,
    serialize_events,
)
from .elaboration import (
    HALF_INDEX_SCHEMA,
    PARQUET_ROW_GROUP_SIZE,
    CycleOrder,
    _BoundedParquetSink,
    _HalfMetaCursor,
    _release_batch_memory,
    _S3Stream,
    _S5Stream,
    segment_half_cycles,
)
from .identity import canonical_parameter_bytes, hash_dependencies, hash_parameters
from .progress import CancelToken, ProgressEvent, ProgressSink, check_cancelled, report_progress
from .storage import GenerationManifest, ResultStore, cleanup_job, open_result

_DEFAULT_CAPACITY_PARAMETERS = CapacityParameters()


@dataclass(frozen=True, slots=True)
class CycleCorrectionParameters:
    """Settings of the cycle correction.

    Attributes:
        enabled: Apply the Accepted spans (True) or recompute without correction (False).
        rest_current_threshold_a: Rest threshold in A, normally the elaboration value.
        cycle_order: Cycle order, normally the elaboration value.
        capacity: `CapacityParameters` for the recomputed Capacity table; its cycle order must
            equal `cycle_order`.
    """

    enabled: bool = True
    rest_current_threshold_a: float = 0.0
    cycle_order: CycleOrder = CycleOrder.CHARGE_DISCHARGE
    capacity: CapacityParameters = _DEFAULT_CAPACITY_PARAMETERS

    def validate(self) -> None:
        """Validate the threshold and Capacity parameters."""
        if not math.isfinite(self.rest_current_threshold_a) or self.rest_current_threshold_a < 0:
            raise ValueError("rest_current_threshold_a must be finite and nonnegative")
        self.capacity.validate()
        if self.capacity.cycle_order is not self.cycle_order:
            raise ValueError("Capacity and correction cycle orders must match")

    def as_dict(self) -> dict[str, Any]:
        """Return the parameters as a JSON-compatible dictionary."""
        return {
            "enabled": self.enabled,
            "rest_current_threshold_a": self.rest_current_threshold_a,
            "cycle_order": self.cycle_order.value,
            "capacity": self.capacity.as_dict(),
        }


_DEFAULT_CORRECTION_PARAMETERS = CycleCorrectionParameters()


@dataclass(frozen=True, slots=True)
class CycleCorrectionRequest:
    """Parameters of `recalculate_cycles`.

    Attributes:
        elaboration_path: Elaboration (or previous correction) generation directory.
        elaboration_fingerprint: Fingerprint of that generation.
        caller_root, sample_id, job_id, generation_id: Destination in the result store.
        events: All reviewed `DcirEvent` records; only valid Accepted events are used.
        measurements: Matching `DcirMeasurement` records (required for manual events, whose
            correction span comes from their boundary samples).
        parameters: `CycleCorrectionParameters`.
        batch_size: Rows per streaming batch.
        existing_dqdv_path: Optional dQ/dV generation of the same elaboration to re-index.
    """

    elaboration_path: Path
    elaboration_fingerprint: str
    caller_root: Path
    sample_id: str
    job_id: str
    generation_id: str
    events: tuple[DcirEvent, ...]
    measurements: tuple[DcirMeasurement, ...] = ()
    parameters: CycleCorrectionParameters = _DEFAULT_CORRECTION_PARAMETERS
    batch_size: int = 65_536
    existing_dqdv_path: Path | None = None

    def validate(self) -> None:
        """Validate parameters, fingerprint and event/measurement IDs."""
        self.parameters.validate()
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len(self.elaboration_fingerprint) != 64 or any(
            value not in "0123456789abcdef" for value in self.elaboration_fingerprint
        ):
            raise ValueError("elaboration_fingerprint must be lowercase SHA-256 hex")
        events_by_id = {event.event_id: event for event in self.events}
        if len(events_by_id) != len(self.events):
            raise ValueError("Correction event IDs must be unique")
        if any(
            value.event.event_id not in events_by_id
            or value.event != events_by_id[value.event.event_id]
            for value in self.measurements
        ):
            raise ValueError("Correction measurements must match their event records")


@dataclass(frozen=True, slots=True)
class CycleCorrectionResult:
    """Published corrected generation.

    Attributes:
        ref: Reference to the corrected generation.
        rows: Number of points.
        half_cycles, cycles: Counts after correction.
        spans: Number of merged correction spans applied.
        fingerprint: Fingerprint for downstream requests.
    """

    ref: ResultRef
    rows: int
    half_cycles: int
    cycles: int
    spans: int
    fingerprint: str


@dataclass(frozen=True, slots=True)
class CorrectedArrays:
    """In-memory result of `recalculate_cycle_arrays`.

    Per-point `cycle`, `half_cycle`, `rest` and `mask` (True for rows inside a correction span),
    the applied `spans` and `first_direction`.
    """

    cycle: np.ndarray
    half_cycle: np.ndarray
    rest: np.ndarray
    mask: np.ndarray
    spans: tuple[tuple[int, int], ...]
    first_direction: int


@dataclass(slots=True)
class _SpanContext:
    """Neighbouring half-cycles and first signed rows around a correction span."""

    first: int
    last: int
    left_half: int | None = None
    right_half: int | None = None
    first_positive: int | None = None
    first_negative: int | None = None


class _LabelCursor:
    """Ordered lookup of cycle labels by source row, reading one batch at a time."""

    def __init__(self, path: Path, batch_size: int) -> None:
        """Open the labelled point file for batched reading."""
        self.file = pq.ParquetFile(path)
        self.batches = iter(
            self.file.iter_batches(
                batch_size=batch_size, columns=["source_row", "cycle", "half_cycle"]
            )
        )
        self.rows = np.empty(0, dtype=np.uint64)
        self.cycles = np.empty(0, dtype=np.int32)
        self.halves = np.empty(0, dtype=np.int32)
        self.position = 0

    def _load(self) -> None:
        """Load the next batch of labels."""
        batch = next(self.batches)
        table = pa.Table.from_batches((batch,))
        self.rows = table["source_row"].to_numpy(zero_copy_only=False)
        self.cycles = table["cycle"].to_numpy(zero_copy_only=False)
        self.halves = table["half_cycle"].to_numpy(zero_copy_only=False)
        self.position = 0

    def lookup(self, requested: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return cycle and half-cycle labels for sorted source rows."""
        cycles = np.empty(len(requested), dtype=np.int32)
        halves = np.empty(len(requested), dtype=np.int32)
        done = 0
        while done < len(requested):
            if self.position == len(self.rows):
                self._load()
            take = int(np.searchsorted(requested, self.rows[-1], side="right")) - done
            if take <= 0:
                self.position = len(self.rows)
                continue
            expected = requested[done : done + take]
            indices = np.searchsorted(self.rows, expected)
            if np.any(indices >= len(self.rows)) or not np.array_equal(
                self.rows[indices], expected
            ):
                raise RuntimeError("Correction label stream is not aligned with source rows")
            cycles[done : done + take] = self.cycles[indices]
            halves[done : done + take] = self.halves[indices]
            self.position = int(indices[-1]) + 1
            done += take
        return cycles, halves

    def close(self) -> None:
        """Close the point file."""
        self.file.close()


def _span_mask(rows: np.ndarray, spans: tuple[tuple[int, int], ...]) -> np.ndarray:
    """Return a mask of rows lying in any inclusive span."""
    mask = np.zeros(len(rows), dtype=bool)
    if not len(rows):
        return mask
    low, high = int(rows[0]), int(rows[-1])
    for first, last in spans:
        if last < low:
            continue
        if first > high:
            break
        mask |= (rows >= first) & (rows <= last)
    return mask


def _cycle_values(half: np.ndarray, first_direction: int, order: CycleOrder) -> np.ndarray:
    """Convert half-cycle numbers to cycle numbers for a first direction and order."""
    cycle = np.zeros(len(half), dtype=np.int32)
    active = half > 0
    expected = 1 if order is CycleOrder.CHARGE_DISCHARGE else -1
    if first_direction == expected:
        cycle[active] = (half[active] + 1) // 2
    else:
        cycle[active] = half[active] // 2 + 1
    return cycle


def cycle_correction_descriptor() -> AnalysisDescriptor:
    """Describe the cycle-correction operation for user interfaces."""
    descriptor = AnalysisDescriptor(
        1,
        OperationId.RECALCULATE_CYCLES,
        "Recalculate Cycles",
        "Explicitly apply or remove Accepted-DCIR cycle correction.",
        ("elaboration", "accepted_dcir"),
        (ParameterGroup("correction", "Cycle correction", 0),),
        (
            ParameterDescriptor(
                "enabled",
                "Apply correction",
                "Apply valid Accepted DCIR spans; false restores the uncorrected cycles.",
                "boolean",
                None,
                True,
                None,
                None,
                None,
                ParameterLevel.BASIC,
                "correction",
            ),
        ),
        (
            OutputDescriptor(
                "cycle",
                "Corrected cycle",
                "int32",
                None,
                True,
                "correction",
                AvailabilityDescriptor("always", "", "Available for every retained row."),
            ),
            OutputDescriptor(
                "half_cycle",
                "Corrected half-cycle",
                "int32",
                None,
                True,
                "correction",
                AvailabilityDescriptor("always", "", "Available for every retained row."),
            ),
        ),
        (),
        True,
        True,
        "rows",
    )
    descriptor.validate()
    return descriptor


def accepted_correction_spans(
    events: tuple[DcirEvent, ...], measurements: tuple[DcirMeasurement, ...] = ()
) -> tuple[tuple[int, int], ...]:
    """Return the merged inclusive source-row spans used for correction.

    Automatic events contribute their pulse rows; manual events contribute the rows of their two
    boundary samples (from `measurements`). Pending, Rejected and events without valid boundaries
    are skipped. Overlapping or adjacent spans are merged. Raises ValueError for inconsistent
    manual boundaries.
    """
    measurement_by_id = {value.event.event_id: value for value in measurements}
    ranges: list[tuple[int, int]] = []
    for event in events:
        if event.state is not ReviewState.ACCEPTED:
            continue
        if event.source is EventSource.AUTO:
            if event.pulse_first_row is None or event.pulse_last_row is None:
                continue
            first, last = event.pulse_first_row, event.pulse_last_row
        else:
            measurement = measurement_by_id.get(event.event_id)
            if measurement is None or measurement.boundary_samples is None:
                continue
            first_sample, last_sample = measurement.boundary_samples
            if (
                first_sample.source_row == last_sample.source_row
                or first_sample.time_s == last_sample.time_s
            ):
                continue
            first = min(first_sample.source_row, last_sample.source_row)
            last = max(first_sample.source_row, last_sample.source_row)
            if (event.correction_start_row, event.correction_end_row) != (first, last):
                raise ValueError("Manual correction rows do not match authenticated boundaries")
        if first < 0 or last < first:
            raise ValueError("Accepted correction span is invalid")
        ranges.append((first, last))
    if not ranges:
        return ()
    ranges.sort()
    merged = [ranges[0]]
    for first, last in ranges[1:]:
        old_first, old_last = merged[-1]
        if first <= old_last + 1:
            merged[-1] = (old_first, max(old_last, last))
        else:
            merged.append((first, last))
    return tuple(merged)


def _labels_for_masked_rows(
    current: np.ndarray,
    source_row: np.ndarray,
    mask: np.ndarray,
    unmasked_half: np.ndarray,
) -> np.ndarray:
    """Assign half-cycle labels to masked rows from their neighbours."""
    half = np.zeros(len(current), dtype=np.int32)
    half[~mask] = unmasked_half
    positions = np.flatnonzero(mask)
    if not len(positions):
        return half
    starts = positions[np.r_[True, np.diff(positions) > 1]]
    ends = positions[np.r_[np.diff(positions) > 1, True]]
    active_positions = np.flatnonzero((~mask) & (half > 0) & (current != 0))
    active_halves = half[active_positions]
    unique_halves, first_indices = np.unique(active_halves, return_index=True)
    active_sign_by_half = {
        int(half_id): 1 if current[active_positions[index]] > 0 else -1
        for half_id, index in zip(unique_halves.tolist(), first_indices.tolist(), strict=True)
    }
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        left_position = start - 1
        while left_position >= 0 and (mask[left_position] or half[left_position] == 0):
            left_position -= 1
        right_position = end + 1
        while right_position < len(mask) and (mask[right_position] or half[right_position] == 0):
            right_position += 1
        left = int(half[left_position]) if left_position >= 0 else None
        right = int(half[right_position]) if right_position < len(mask) else None
        if left is None and right is None:
            continue
        if left is None or right is None or left == right:
            selected = left if left is not None else right
            assert selected is not None
            half[start : end + 1] = selected
            continue
        left_sign = active_sign_by_half.get(left, 0)
        right_sign = active_sign_by_half.get(right, 0)
        if left_sign == right_sign:
            half[start : end + 1] = left
            continue
        values = current[start : end + 1]
        matches_right = np.flatnonzero(np.sign(values) == right_sign)
        split = int(matches_right[0]) if len(matches_right) else len(values)
        half[start : start + split] = left
        half[start + split : end + 1] = right
    return half


def recalculate_cycle_arrays(
    current_a: np.ndarray,
    source_row: np.ndarray,
    spans: tuple[tuple[int, int], ...],
    *,
    enabled: bool,
    rest_current_threshold_a: float,
    cycle_order: CycleOrder,
) -> CorrectedArrays:
    """Recompute half-cycle and cycle labels with spans excluded (pure).

    Segmentation runs on the rows outside `spans`; rows inside a span then receive the label of
    the neighbouring half-cycle (split at the first sample matching the following direction when
    the neighbours differ). With ``enabled=False`` or no span, the result equals
    `segment_half_cycles`.
    """
    current = np.asarray(current_a, dtype=np.float64)
    rows = np.asarray(source_row, dtype=np.uint64)
    if len(current) != len(rows):
        raise ValueError("current and source_row arrays must align")
    mask = np.zeros(len(current), dtype=bool)
    if enabled:
        for first, last in spans:
            mask |= (rows >= first) & (rows <= last)
    if not enabled or not np.any(mask):
        cycle, half, rest, first_direction = segment_half_cycles(
            current, rest_current_threshold_a, cycle_order
        )
        return CorrectedArrays(cycle, half, rest, mask, spans if enabled else (), first_direction)
    if np.all(mask):
        raise ValueError(
            "Correction has no unmasked sample from which to establish half-cycle context"
        )
    unmasked_cycle, unmasked_half, _, first_direction = segment_half_cycles(
        current[~mask], rest_current_threshold_a, cycle_order
    )
    del unmasked_cycle
    half = _labels_for_masked_rows(current, rows, mask, unmasked_half)
    rest = np.abs(current) <= rest_current_threshold_a
    cycle = np.zeros(len(current), dtype=np.int32)
    active = half > 0
    expected_first = 1 if cycle_order is CycleOrder.CHARGE_DISCHARGE else -1
    if first_direction == expected_first:
        cycle[active] = (half[active] + 1) // 2
    else:
        cycle[active] = half[active] // 2 + 1
    return CorrectedArrays(cycle, half, rest, mask, spans, first_direction)


def _validate_input(request: CycleCorrectionRequest) -> Path:
    """Check the input operation and fingerprint; return the resolved path."""
    root = request.elaboration_path.resolve(strict=True)
    info = open_result(root)
    if info.operation not in (OperationId.INGEST_AND_ELABORATE, OperationId.RECALCULATE_CYCLES):
        raise ValueError("Cycle correction requires an elaboration generation")
    manifest = GenerationManifest.from_dict(
        json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    )
    if manifest.content_hash != request.elaboration_fingerprint:
        raise ValueError("elaboration_fingerprint does not match the source generation")
    return root


_DQDV_TABLES = {
    "dqdv.parquet": "gregorithm.dqdv.v1",
    "dqdv_reconstruction.parquet": "gregorithm.dqdv_reconstruction.v1",
}
"""dQ/dV tables copied with re-indexed cycle labels, and their logical schemas."""


def _validate_existing_dqdv(
    request: CycleCorrectionRequest,
) -> tuple[Path, GenerationManifest] | None:
    """Check that an existing dQ/dV generation belongs to the input."""
    if request.existing_dqdv_path is None:
        return None
    root = request.existing_dqdv_path.resolve(strict=True)
    info = open_result(root)
    if info.operation is not OperationId.DQDV:
        raise ValueError("existing_dqdv_path must name a dQ/dV generation")
    manifest = GenerationManifest.from_dict(
        json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    )
    if manifest.content_hash is None:
        raise ValueError("Existing dQ/dV generation has no content fingerprint")
    lineage = {value.id: value.fingerprint for value in manifest.dependencies}
    if lineage.get("elaboration") != request.elaboration_fingerprint:
        raise ValueError("Existing dQ/dV generation has different elaboration lineage")
    source = root / "tables" / "dqdv.parquet"
    if not source.is_file():
        raise ValueError("Existing dQ/dV generation has no dQ/dV table")
    required = {
        "cycle",
        "half_cycle",
        "phase",
        "anchor_source_row",
        "segment_index",
        "voltage_mid_v",
        "dqdv_raw_ah_per_v",
        "dqdv_smoothed_ah_per_v",
    }
    if not required.issubset(pq.read_schema(source).names):
        raise ValueError("Existing dQ/dV table has an incompatible schema")
    return root, manifest


def _refresh_soc(
    measurements: tuple[DcirMeasurement, ...],
    totals: dict[int, tuple[float, ...]],
    first_direction: int,
    fingerprint: str,
) -> tuple[DcirMeasurement, ...]:
    """Recompute the optional SOC of automatic measurements after correction."""
    refreshed: list[DcirMeasurement] = []
    for measurement in measurements:
        context: SocContext | None = None
        event = measurement.event
        half_id = event.half_cycle
        if event.source is EventSource.AUTO and half_id is not None and half_id > 1:
            neighbor_ids = (half_id - 1, half_id, half_id + 1)
            if all(value in totals for value in neighbor_ids):
                directions = {
                    value: first_direction * (1 if value % 2 == 1 else -1) for value in neighbor_ids
                }
                if directions[half_id - 1] == directions[half_id + 1]:
                    flank_sign = directions[half_id - 1]
                    pulse_sign = directions[half_id]
                    flank_index = 0 if flank_sign > 0 else 1
                    pulse_index = 0 if pulse_sign > 0 else 1
                    q1 = totals[half_id - 1][flank_index]
                    q2 = totals[half_id + 1][flank_index]
                    q_pulse = totals[half_id][pulse_index]
                    if all(math.isfinite(value) for value in (q1, q2, q_pulse)):
                        context = SocContext(
                            q1,
                            q2,
                            q_pulse,
                            PulsePolarity.CHARGE if flank_sign > 0 else PulsePolarity.DISCHARGE,
                            half_id - 1,
                            half_id + 1,
                            half_id,
                            fingerprint,
                        )
        soc, diagnostic = _soc(context, event.event_id)
        diagnostics = tuple(
            value for value in measurement.diagnostics if value.code.value != "DCIR_SOC_UNAVAILABLE"
        )
        if diagnostic is not None:
            diagnostics = (*diagnostics, diagnostic)
        refreshed.append(
            replace(
                measurement,
                soc_pct=soc,
                soc_context=context,
                diagnostics=diagnostics,
            )
        )
    return tuple(refreshed)


def recalculate_cycles(
    request: CycleCorrectionRequest,
    *,
    progress: ProgressSink | None = None,
    cancel: CancelToken | None = None,
    fault_hook: Any | None = None,
) -> CycleCorrectionResult:
    """Apply (or remove) cycle correction and publish a corrected generation.

    Args:
        request: `CycleCorrectionRequest`.
        progress: Optional progress sink (unit ``rows``).
        cancel: Optional cancellation token.
        fault_hook: Testing hook; leave it None.

    Returns:
        `CycleCorrectionResult`.

    Raises:
        ValueError: invalid input, fingerprint mismatch, inconsistent events, or spans covering
            every sample.
        StorageError, OperationCancelled: storage failures or cancellation.

    Work is done in bounded Parquet passes with temporary scratch files in the staging directory,
    so memory use does not grow with the file size.
    """
    request.validate()
    root = _validate_input(request)
    existing_dqdv = _validate_existing_dqdv(request)
    points_path = root / "points" / "elaborated.parquet"
    rows = pq.ParquetFile(points_path).metadata.num_rows
    if rows < 1:
        raise ValueError("Cycle correction requires at least one point")
    # A stable internal chunk size makes floating-point accumulation independent
    # of the caller's I/O preference while remaining far below the process budget.
    internal_batch_rows = 65_536
    spans = (
        accepted_correction_spans(request.events, request.measurements)
        if request.parameters.enabled
        else ()
    )
    contexts = [_SpanContext(first, last) for first, last in spans]
    check_cancelled(cancel, OperationId.RECALCULATE_CYCLES.value)
    disk_probe = request.caller_root.resolve(strict=False)
    while not disk_probe.exists():
        if disk_probe.parent == disk_probe:
            raise OSError("No existing filesystem is available for correction scratch data")
        disk_probe = disk_probe.parent
    source_bytes = sum(
        path.stat().st_size
        for path in (
            points_path,
            root / "indexes" / "half_cycles.parquet",
            root / "indexes" / "segments.parquet",
        )
        if path.is_file()
    )
    required_free_bytes = max(64 * 1024 * 1024, source_bytes * 5)
    available_bytes = shutil.disk_usage(disk_probe).free
    if available_bytes < required_free_bytes:
        raise OSError(
            "Cycle correction requires at least "
            f"{required_free_bytes} free scratch bytes; {available_bytes} are available"
        )
    store = ResultStore(request.caller_root)
    staging = store.begin_job(request.sample_id, request.job_id, request.generation_id)
    try:
        point_dir = staging.path / "points"
        index_dir = staging.path / "indexes"
        table_dir = staging.path / "tables"
        report_dir = staging.path / "reports"
        scratch_dir = staging.path / ".cycle-correction"
        for directory in (point_dir, index_dir, table_dir, report_dir, scratch_dir):
            directory.mkdir()
        unmasked_path = scratch_dir / "unmasked.parquet"
        provisional_path = scratch_dir / "provisional.parquet"
        half_meta_path = scratch_dir / "half-meta.parquet"

        # Pass 1: apply half-cycle segmentation only to unmasked rows. _S3Stream carries
        # qualifying sign runs across physical batch boundaries.
        s3 = _S3Stream(request.parameters.rest_current_threshold_a, request.parameters.cycle_order)
        unmasked_writer: pq.ParquetWriter | None = None
        masked_rows = 0
        source_file = pq.ParquetFile(points_path)
        try:
            for batch in source_file.iter_batches(
                batch_size=internal_batch_rows,
                columns=["time_s", "current_a", "voltage_v", "step", "source_row"],
            ):
                check_cancelled(cancel, OperationId.RECALCULATE_CYCLES.value)
                source = pa.Table.from_batches((batch,))
                source_rows = source["source_row"].to_numpy(zero_copy_only=False)
                mask = _span_mask(source_rows, spans)
                masked_rows += int(np.count_nonzero(mask))
                if np.all(mask):
                    continue
                emitted = s3.feed(source.filter(pa.array(~mask)))
                if emitted is not None:
                    if unmasked_writer is None:
                        unmasked_writer = pq.ParquetWriter(
                            unmasked_path, emitted.schema, compression="zstd"
                        )
                    unmasked_writer.write_table(emitted, row_group_size=internal_batch_rows)
            emitted = s3.finish()
            if emitted is not None:
                if unmasked_writer is None:
                    unmasked_writer = pq.ParquetWriter(
                        unmasked_path, emitted.schema, compression="zstd"
                    )
                unmasked_writer.write_table(emitted, row_group_size=internal_batch_rows)
        finally:
            source_file.close()
            if unmasked_writer is not None:
                unmasked_writer.close()
        if unmasked_writer is None or not unmasked_path.exists():
            raise ValueError(
                "Correction has no unmasked sample from which to establish half-cycle context"
            )
        first_direction = s3.first_direction
        _release_batch_memory()

        # Derive compact context for each masked range without retaining point arrays.
        if contexts:
            previous_half: int | None = None
            unmasked_file = pq.ParquetFile(unmasked_path)
            for batch in unmasked_file.iter_batches(
                batch_size=internal_batch_rows, columns=["source_row", "half_cycle"]
            ):
                values = pa.Table.from_batches((batch,))
                batch_rows = values["source_row"].to_numpy(zero_copy_only=False)
                halves = values["half_cycle"].to_numpy(zero_copy_only=False)
                for context in contexts:
                    if int(batch_rows[-1]) < context.first:
                        active_left = np.flatnonzero(halves > 0)
                        if len(active_left):
                            context.left_half = int(halves[active_left[-1]])
                    elif context.left_half is None:
                        left_at = int(np.searchsorted(batch_rows, context.first, side="left")) - 1
                        active_left = np.flatnonzero(
                            (np.arange(len(halves)) <= left_at) & (halves > 0)
                        )
                        if len(active_left):
                            context.left_half = int(halves[active_left[-1]])
                        elif previous_half is not None:
                            context.left_half = previous_half
                    if context.right_half is None:
                        active_right = np.flatnonzero((batch_rows > context.last) & (halves > 0))
                        if len(active_right):
                            context.right_half = int(halves[active_right[0]])
                active = np.flatnonzero(halves > 0)
                if len(active):
                    previous_half = int(halves[active[-1]])
            unmasked_file.close()

            source_file = pq.ParquetFile(points_path)
            for batch in source_file.iter_batches(
                batch_size=internal_batch_rows, columns=["source_row", "current_a"]
            ):
                values = pa.Table.from_batches((batch,))
                batch_rows = values["source_row"].to_numpy(zero_copy_only=False)
                current = values["current_a"].to_numpy(zero_copy_only=False)
                for context in contexts:
                    inside = (batch_rows >= context.first) & (batch_rows <= context.last)
                    if not np.any(inside):
                        continue
                    positive = np.flatnonzero(inside & (current > 0))
                    negative = np.flatnonzero(inside & (current < 0))
                    if context.first_positive is None and len(positive):
                        context.first_positive = int(batch_rows[positive[0]])
                    if context.first_negative is None and len(negative):
                        context.first_negative = int(batch_rows[negative[0]])
            source_file.close()

        needed_halves = {
            value
            for context in contexts
            for value in (context.left_half, context.right_half)
            if value is not None and value > 0
        }
        half_signs: dict[int, int] = {}
        if needed_halves:
            unmasked_file = pq.ParquetFile(unmasked_path)
            for batch in unmasked_file.iter_batches(
                batch_size=internal_batch_rows, columns=["current_a", "half_cycle"]
            ):
                values = pa.Table.from_batches((batch,))
                current = values["current_a"].to_numpy(zero_copy_only=False)
                halves = values["half_cycle"].to_numpy(zero_copy_only=False)
                for half_id in needed_halves - half_signs.keys():
                    positions = np.flatnonzero((halves == half_id) & (current != 0))
                    if len(positions):
                        half_signs[half_id] = 1 if current[positions[0]] > 0 else -1
            unmasked_file.close()

        # Pass 2: merge the ordered unmasked labels back onto all real rows and
        # stream cumulative quantities into a disk-backed provisional table.
        label_cursor = _LabelCursor(unmasked_path, internal_batch_rows)
        s5 = _S5Stream()
        provisional_writer: _BoundedParquetSink | None = None
        meta_writer: _BoundedParquetSink | None = None
        source_file = pq.ParquetFile(points_path)
        try:
            for batch in source_file.iter_batches(
                batch_size=internal_batch_rows,
                columns=["time_s", "current_a", "voltage_v", "step", "source_row"],
            ):
                check_cancelled(cancel, OperationId.RECALCULATE_CYCLES.value)
                source = pa.Table.from_batches((batch,))
                source_rows = source["source_row"].to_numpy(zero_copy_only=False)
                current = source["current_a"].to_numpy(zero_copy_only=False)
                mask = _span_mask(source_rows, spans)
                half = np.zeros(batch.num_rows, dtype=np.int32)
                if np.any(~mask):
                    _, values = label_cursor.lookup(source_rows[~mask])
                    half[~mask] = values
                for context in contexts:
                    inside = (source_rows >= context.first) & (source_rows <= context.last)
                    if not np.any(inside):
                        continue
                    left, right = context.left_half, context.right_half
                    if left is None and right is None:
                        continue
                    if left is None or right is None or left == right:
                        selected = left if left is not None else right
                        assert selected is not None
                        half[inside] = selected
                        continue
                    left_sign = half_signs.get(left, 0)
                    right_sign = half_signs.get(right, 0)
                    if left_sign == right_sign:
                        half[inside] = left
                        continue
                    split = context.first_positive if right_sign > 0 else context.first_negative
                    half[inside & ((source_rows < split) if split is not None else True)] = left
                    if split is not None:
                        half[inside & (source_rows >= split)] = right
                cycle = _cycle_values(half, first_direction, request.parameters.cycle_order)
                rest = np.abs(current) <= request.parameters.rest_current_threshold_a
                labelled = (
                    source.append_column("cycle", pa.array(cycle))
                    .append_column("half_cycle", pa.array(half))
                    .append_column("rest", pa.array(rest))
                )
                integrated = s5.feed(labelled)
                if provisional_writer is None:
                    provisional_writer = _BoundedParquetSink(
                        provisional_path, integrated.schema, internal_batch_rows
                    )
                provisional_writer.write_table(integrated)
                if s5.closed is not None:
                    if meta_writer is None:
                        meta_writer = _BoundedParquetSink(
                            half_meta_path,
                            s5.closed.schema,
                            internal_batch_rows,
                            baseline_row_group_size=False,
                        )
                    meta_writer.write_table(s5.closed)
            final_meta = s5.finish()
            if meta_writer is None:
                meta_writer = _BoundedParquetSink(
                    half_meta_path,
                    final_meta.schema,
                    internal_batch_rows,
                    baseline_row_group_size=False,
                )
            meta_writer.write_table(final_meta)
        finally:
            source_file.close()
            label_cursor.close()
            if provisional_writer is not None:
                provisional_writer.close()
            if meta_writer is not None:
                meta_writer.close()
        _release_batch_memory()

        # Pass 3: apply invalid-energy metadata and build the compact half index.
        corrected_points = point_dir / "elaborated.parquet"
        half_path = index_dir / "half_cycles.parquet"
        point_writer: pq.ParquetWriter | None = None
        half_database = sqlite3.connect(scratch_dir / "half-index.sqlite3")
        half_database.execute(
            "CREATE TABLE halves (half_cycle INTEGER PRIMARY KEY, cycle INTEGER, "
            "first_row INTEGER, last_row INTEGER, first_direction INTEGER, invalid INTEGER, "
            "cq REAL, dq REAL, ce REAL, de REAL, ocq REAL, odq REAL, oce REAL, ode REAL)"
        )

        def store_half(record: dict[str, Any]) -> None:
            existing = half_database.execute(
                "SELECT first_row, invalid, ocq, odq, oce, ode FROM halves WHERE half_cycle=?",
                (record["half_cycle"],),
            ).fetchone()
            first_row = record["first_row"] if existing is None else existing[0]
            invalid_value = int(record["invalid"]) | (0 if existing is None else int(existing[1]))
            prior_outgoing = (0.0, 0.0, 0.0, 0.0) if existing is None else existing[2:]
            outgoing_values = tuple(
                float(prior_outgoing[index]) + float(record["outgoing"][index])
                for index in range(4)
            )
            half_database.execute(
                "INSERT OR REPLACE INTO halves VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record["half_cycle"],
                    record["cycle"],
                    first_row,
                    record["last_row"],
                    first_direction,
                    invalid_value,
                    *record["cumulative"],
                    *outgoing_values,
                ),
            )

        half_cursor = _HalfMetaCursor(half_meta_path, internal_batch_rows)
        open_half: dict[str, Any] | None = None
        half_count = 0
        cycle_count = 0
        provisional_file = pq.ParquetFile(provisional_path)
        try:
            for batch in provisional_file.iter_batches(batch_size=internal_batch_rows):
                check_cancelled(cancel, OperationId.RECALCULATE_CYCLES.value)
                table = pa.Table.from_batches((batch,))
                half = table["half_cycle"].to_numpy(zero_copy_only=False)
                starts = np.r_[0, np.flatnonzero(half[1:] != half[:-1]) + 1]
                ends = np.r_[starts[1:], len(half)]
                ids = half[starts].astype(np.int32)
                invalid_groups, outgoing = half_cursor.lookup(ids)
                invalid = pa.array(np.repeat(invalid_groups, ends - starts))
                for name in ("charge_energy_wh", "discharge_energy_wh"):
                    index = table.schema.get_field_index(name)
                    table = table.set_column(index, name, pc.if_else(invalid, None, table[name]))
                table = table.select(
                    [
                        "time_s",
                        "current_a",
                        "voltage_v",
                        "charge_capacity_ah",
                        "discharge_capacity_ah",
                        "charge_energy_wh",
                        "discharge_energy_wh",
                        "cycle",
                        "half_cycle",
                        "rest",
                        "step",
                        "source_row",
                    ]
                )
                if point_writer is None:
                    point_writer = pq.ParquetWriter(
                        corrected_points,
                        table.schema,
                        compression="zstd",
                        write_statistics=["time_s", "cycle", "half_cycle"],
                    )
                point_writer.write_table(table, row_group_size=PARQUET_ROW_GROUP_SIZE)
                source_rows = table["source_row"].to_numpy(zero_copy_only=False)
                cycle = table["cycle"].to_numpy(zero_copy_only=False)
                cq = table["charge_capacity_ah"].to_numpy(zero_copy_only=False)
                dq = table["discharge_capacity_ah"].to_numpy(zero_copy_only=False)
                ce = table["charge_energy_wh"].to_numpy(zero_copy_only=False)
                de = table["discharge_energy_wh"].to_numpy(zero_copy_only=False)
                last = ends - 1
                first_rows = source_rows[starts].astype(np.uint64)
                records: list[dict[str, Any]] = [
                    {
                        "half_cycle": int(ids[index]),
                        "cycle": int(cycle[starts[index]]),
                        "first_row": int(first_rows[index]),
                        "last_row": int(source_rows[last[index]]),
                        "invalid": bool(invalid_groups[index]),
                        "cumulative": (
                            float(cq[last[index]]),
                            float(dq[last[index]]),
                            float(ce[last[index]]),
                            float(de[last[index]]),
                        ),
                        "outgoing": tuple(map(float, outgoing[index])),
                    }
                    for index in range(len(ids))
                ]
                if open_half is not None and records[0]["half_cycle"] == open_half["half_cycle"]:
                    records[0]["first_row"] = open_half["first_row"]
                    records[0]["invalid"] |= open_half["invalid"]
                    records[0]["outgoing"] = tuple(
                        open_half["outgoing"][index] + records[0]["outgoing"][index]
                        for index in range(4)
                    )
                    open_half = None
                elif open_half is not None:
                    store_half(open_half)
                    open_half = None
                for record in records[:-1]:
                    store_half(record)
                open_half = records[-1]
        finally:
            provisional_file.close()
            half_cursor.close()
            if point_writer is not None:
                point_writer.close()
        if open_half is None:
            half_database.close()
            raise RuntimeError("Correction produced no half-cycle record")
        store_half(open_half)
        half_database.commit()
        half_writer = _BoundedParquetSink(
            half_path, HALF_INDEX_SCHEMA, internal_batch_rows, baseline_row_group_size=False
        )
        pending: list[dict[str, Any]] = []
        for value in half_database.execute("SELECT * FROM halves ORDER BY half_cycle"):
            invalid_value = bool(value[5])
            pending.append(
                {
                    "half_cycle": value[0],
                    "cycle": value[1],
                    "first_row": value[2],
                    "last_row": value[3],
                    "first_direction": value[4],
                    "charge_capacity_ah": value[6] + value[10],
                    "discharge_capacity_ah": value[7] + value[11],
                    "charge_energy_wh": None if invalid_value else value[8] + value[12],
                    "discharge_energy_wh": None if invalid_value else value[9] + value[13],
                }
            )
            if len(pending) == internal_batch_rows:
                half_writer.write_table(pa.Table.from_pylist(pending, schema=HALF_INDEX_SCHEMA))
                pending.clear()
        if pending:
            half_writer.write_table(pa.Table.from_pylist(pending, schema=HALF_INDEX_SCHEMA))
        half_writer.close()
        half_count = int(half_database.execute("SELECT COUNT(*) FROM halves").fetchone()[0])
        cycle_count = int(half_database.execute("SELECT MAX(cycle) FROM halves").fetchone()[0])
        half_database.close()
        _release_batch_memory()

        # Stream the existing segment index and replace only corrected labels.
        segment_source = root / "indexes" / "segments.parquet"
        segment_target = index_dir / "segments.parquet"
        labels = _LabelCursor(corrected_points, internal_batch_rows)
        segment_writer: pq.ParquetWriter | None = None
        segment_file = pq.ParquetFile(segment_source)
        try:
            for batch in segment_file.iter_batches(batch_size=internal_batch_rows):
                table = pa.Table.from_batches((batch,))
                first_rows = table["first_row"].to_numpy(zero_copy_only=False)
                cycles, halves = labels.lookup(first_rows)
                table = table.set_column(
                    table.schema.get_field_index("cycle"), "cycle", pa.array(cycles)
                )
                table = table.set_column(
                    table.schema.get_field_index("half_cycle"), "half_cycle", pa.array(halves)
                )
                if segment_writer is None:
                    segment_writer = pq.ParquetWriter(
                        segment_target, table.schema, compression="zstd"
                    )
                segment_writer.write_table(table)
        finally:
            segment_file.close()
            labels.close()
            if segment_writer is not None:
                segment_writer.close()

        # Resolve only event anchors and SOC-neighbor totals, never all points/indexes.
        anchor_ids: set[int] = set()
        measurement_by_id = {value.event.event_id: value for value in request.measurements}
        for event in request.events:
            if event.source is EventSource.AUTO and event.pulse_first_row is not None:
                anchor_ids.add(event.pulse_first_row)
            elif event.source is EventSource.MANUAL:
                measurement = measurement_by_id.get(event.event_id)
                if measurement is not None and measurement.manual_values is not None:
                    sample = measurement.manual_values[3].sample
                    if sample is not None:
                        anchor_ids.add(sample.source_row)
        anchor_labels: dict[int, tuple[int, int]] = {}
        points_file = pq.ParquetFile(corrected_points)
        for batch in points_file.iter_batches(
            batch_size=internal_batch_rows, columns=["source_row", "cycle", "half_cycle"]
        ):
            table = pa.Table.from_batches((batch,))
            batch_rows = table["source_row"].to_numpy(zero_copy_only=False)
            for anchor_value in anchor_ids - anchor_labels.keys():
                position = int(np.searchsorted(batch_rows, anchor_value))
                if position < len(batch_rows) and int(batch_rows[position]) == anchor_value:
                    anchor_labels[anchor_value] = (
                        int(table["cycle"][position].as_py()),
                        int(table["half_cycle"][position].as_py()),
                    )
        points_file.close()
        updated: dict[str, DcirEvent] = {}
        for event in request.events:
            anchor: int | None = event.pulse_first_row if event.source is EventSource.AUTO else None
            if event.source is EventSource.MANUAL:
                measurement = measurement_by_id.get(event.event_id)
                if measurement is not None and measurement.manual_values is not None:
                    sample = measurement.manual_values[3].sample
                    anchor = None if sample is None else sample.source_row
            label = None if anchor is None else anchor_labels.get(anchor)
            updated[event.event_id] = replace(
                event,
                cycle=None if label is None else label[0],
                half_cycle=None if label is None else label[1],
            )
        updated_events = tuple(updated[value.event_id] for value in request.events)
        updated_measurements = tuple(
            replace(value, event=updated[value.event.event_id]) for value in request.measurements
        )
        required_totals = {
            half_id + offset
            for event in updated_events
            if event.source is EventSource.AUTO and event.half_cycle is not None
            for half_id in (event.half_cycle,)
            for offset in (-1, 0, 1)
            if half_id + offset > 0
        }
        totals: dict[int, tuple[float, ...]] = {}
        half_file = pq.ParquetFile(half_path)
        for batch in half_file.iter_batches(batch_size=internal_batch_rows):
            table = pa.Table.from_batches((batch,))
            ids = table["half_cycle"].to_numpy(zero_copy_only=False)
            for half_id in required_totals - totals.keys():
                positions = np.flatnonzero(ids == half_id)
                if len(positions):
                    position = int(positions[0])
                    totals[half_id] = tuple(
                        float(table[name][position].as_py())
                        if table[name][position].as_py() is not None
                        else math.nan
                        for name in (
                            "charge_capacity_ah",
                            "discharge_capacity_ah",
                            "charge_energy_wh",
                            "discharge_energy_wh",
                        )
                    )
        half_file.close()
        updated_measurements = _refresh_soc(
            updated_measurements, totals, first_direction, fingerprint="pending"
        )
        accepted_fingerprint = accepted_event_fingerprint(updated_events, updated_measurements)
        dqdv_fingerprint = None if existing_dqdv is None else existing_dqdv[1].content_hash
        parameters = {
            **request.parameters.as_dict(),
            "accepted_event_fingerprint": accepted_fingerprint,
            "existing_dqdv_fingerprint": dqdv_fingerprint,
        }
        fingerprint = hashlib.sha256(
            b"gregorithm-cycle-correction-v1\0"
            + __version__.encode()
            + b"\0"
            + request.elaboration_fingerprint.encode()
            + b"\0"
            + canonical_parameter_bytes(parameters)
        ).hexdigest()
        updated_measurements = _refresh_soc(
            updated_measurements, totals, first_direction, fingerprint
        )
        (table_dir / "dcir_events.json").write_bytes(serialize_events(updated_events))
        if updated_measurements:
            (table_dir / "dcir_measurements.json").write_bytes(
                _serialize_measurements(updated_measurements)
            )

        capacity_scratch = scratch_dir / "capacity"
        capacity_scratch.mkdir()
        base_path = capacity_scratch / "base.parquet"
        cc_path = capacity_scratch / "cc.parquet"
        base_count, reference, _ = _write_base_rows(
            half_path,
            base_path,
            request.parameters.capacity.cycle_order,
            internal_batch_rows,
            cancel,
            request.parameters.capacity.reference_cycle,
        )
        cc_count, _ = _write_cc_rows(
            corrected_points,
            cc_path,
            internal_batch_rows,
            cancel,
            None,
            rows,
            first_direction,
        )
        capacity_count = _finalize_rows(
            base_path,
            cc_path,
            table_dir / "capacity.parquet",
            table_dir / "capacity_diagnostics.parquet",
            request.parameters.capacity,
            reference,
            internal_batch_rows,
            cancel,
            None,
            rows,
        )
        if base_count != cc_count or base_count != capacity_count:
            raise RuntimeError("Corrected Capacity streams produced inconsistent counts")

        dqdv_sources = (
            ()
            if existing_dqdv is None
            else tuple((existing_dqdv[0] / "tables" / name, name) for name in _DQDV_TABLES)
        )
        for source, name in dqdv_sources:
            if not source.is_file():
                continue
            target = table_dir / name
            labels = _LabelCursor(corrected_points, internal_batch_rows)
            writer: pq.ParquetWriter | None = None
            source_file = pq.ParquetFile(source)
            try:
                for batch in source_file.iter_batches(batch_size=internal_batch_rows):
                    table = pa.Table.from_batches((batch,))
                    anchors = table["anchor_source_row"].to_numpy(zero_copy_only=False)
                    cycles, halves = labels.lookup(anchors)
                    table = table.set_column(
                        table.schema.get_field_index("cycle"), "cycle", pa.array(cycles)
                    )
                    table = table.set_column(
                        table.schema.get_field_index("half_cycle"),
                        "half_cycle",
                        pa.array(halves),
                    )
                    if writer is None:
                        writer = pq.ParquetWriter(target, table.schema, compression="zstd")
                    writer.write_table(table)
                if writer is None:
                    writer = pq.ParquetWriter(target, source_file.schema_arrow, compression="zstd")
            finally:
                source_file.close()
                labels.close()
                if writer is not None:
                    writer.close()

        scratch_bytes = sum(
            path.stat().st_size for path in scratch_dir.rglob("*") if path.is_file()
        )
        shutil.rmtree(scratch_dir)
        report = {
            "schema_version": 1,
            "enabled": request.parameters.enabled,
            "accepted_spans": [list(value) for value in spans],
            "masked_rows": masked_rows,
            "point_records": rows,
            "half_cycle_records": half_count,
            "capacity_records": capacity_count,
            "elaboration_point_scans": 3,
            "scratch_peak_bytes_upper_bound": scratch_bytes,
            "bounded_batch_rows": internal_batch_rows,
            "s4_reclassified": False,
            "dcir_redetected": False,
            "dqdv_recomputed": False,
        }
        (report_dir / "cycle_correction.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report_progress(
            progress,
            ProgressEvent(OperationId.RECALCULATE_CYCLES, "corrected", rows, rows, "rows"),
        )
        dependencies = [
            DependencyRef("accepted_dcir", accepted_fingerprint),
            DependencyRef("elaboration", request.elaboration_fingerprint),
        ]
        if dqdv_fingerprint is not None:
            dependencies.append(DependencyRef("dqdv", dqdv_fingerprint))
        dependency_tuple = tuple(sorted(dependencies, key=lambda value: value.id))
        schemas = {
            "points/elaborated.parquet": "gregorithm.elaborated_points.v1",
            "indexes/half_cycles.parquet": "gregorithm.half_cycle_index.v1",
            "indexes/segments.parquet": "gregorithm.step_segment_index.v1",
            "tables/capacity.parquet": "gregorithm.capacity.v1",
            "tables/capacity_diagnostics.parquet": "gregorithm.capacity_diagnostics.v1",
            "tables/dcir_events.json": "gregorithm.dcir_events.v1",
            "reports/cycle_correction.json": "gregorithm.cycle_correction_report.v1",
        }
        if updated_measurements:
            schemas["tables/dcir_measurements.json"] = "gregorithm.dcir_measurements.v1"
        for name, schema_name in _DQDV_TABLES.items():
            if (table_dir / name).exists():
                schemas[f"tables/{name}"] = schema_name
        manifest = staging.build_manifest(
            operation=OperationId.RECALCULATE_CYCLES,
            parameter_hash=hash_parameters(parameters),
            dependency_hash=hash_dependencies(dependency_tuple),
            dependencies=dependency_tuple,
            content_hash=fingerprint,
            row_counts={
                "points": rows,
                "half_cycle_records": half_count,
                "capacity_records": capacity_count,
            },
            logical_schemas=schemas,
        )
        ref = store.publish(staging, manifest, cancel=cancel, fault_hook=fault_hook)
        return CycleCorrectionResult(ref, rows, s5.half, cycle_count, len(spans), fingerprint)
    except Exception:
        staging.close()
        if staging.path.exists():
            cleanup_job(request.caller_root, staging.path)
        raise
