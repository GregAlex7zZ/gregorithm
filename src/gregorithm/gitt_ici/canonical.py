"""Read-only adapter from a published elaboration generation to GITT/ICI canonical records.

This is the only supported way to obtain `CanonicalSegment` and `CanonicalSample` objects from
stored data. It reads ``indexes/segments.parquet`` and ``points/elaborated.parquet`` of an
elaboration (or corrected elaboration) generation after `open_result` has verified every checksum.
It never reads raw sources, never re-imports, never deduplicates and never re-segments: segment
boundaries, steps, cycles, half-cycles, polarities and ``source_row`` identities are copied as
gregorithm produced them.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from gregorithm.contracts import OperationId
from gregorithm.storage import GenerationManifest, open_result

from .contracts import CanonicalSample, CanonicalSegment, GittIciEvent, Protocol
from .events import associate_events

CONTEXT_SEGMENTS = 2
"""Segments read on each side of the scope: the rest before a pulse and the active segment that
precedes that rest (time origin of its relaxation) are context, not analysed events."""

_POINT_COLUMNS = (
    "source_row",
    "time_s",
    "current_a",
    "voltage_v",
    "cycle",
    "half_cycle",
    "charge_capacity_ah",
    "discharge_capacity_ah",
)


@dataclass(frozen=True, slots=True)
class CanonicalScope:
    """Canonical segments and samples of an analysis scope plus their context.

    ``segments`` and ``samples`` include up to `CONTEXT_SEGMENTS` neighbouring segments on each
    side so that rests and time origins outside the scope are available as context;
    ``scope_segment_ids`` are the segments actually selected. Use `events` to associate only the
    pulses inside the scope.
    """

    generation_path: Path
    elaboration_fingerprint: str
    cycles: tuple[int, ...]
    half_cycles: tuple[int, ...]
    segments: tuple[CanonicalSegment, ...]
    samples: tuple[CanonicalSample, ...]
    scope_segment_ids: frozenset[int]

    def events(self, protocol: Protocol, *, complete_only: bool = True) -> tuple[GittIciEvent, ...]:
        """Associate events whose active segment belongs to the scope (context excluded)."""
        return tuple(
            event
            for event in associate_events(
                self.segments,
                protocol=protocol,
                elaboration_fingerprint=self.elaboration_fingerprint,
                complete_only=complete_only,
            )
            if event.active_segment_id in self.scope_segment_ids
        )


def load_canonical_scope(
    generation: Path,
    *,
    cycles: Sequence[int] = (),
    half_cycles: Sequence[int] = (),
    elaboration_fingerprint: str | None = None,
) -> CanonicalScope:
    """Load the canonical segments and samples of selected cycles/half-cycles.

    Empty ``cycles`` or ``half_cycles`` mean no restriction on that field. When
    ``elaboration_fingerprint`` is given it must match the generation's content hash. Memory is
    bounded by the scope plus its context: points are read with a time-range filter on the
    elaborated table, which carries time statistics, and then restricted to the selected rows.
    Raises ValueError for a generation that is not an elaboration or a mismatching fingerprint,
    and `StorageError` (from `open_result`) for a corrupt generation.
    """
    root = generation.resolve(strict=True)
    info = open_result(root)
    if info.operation not in (OperationId.INGEST_AND_ELABORATE, OperationId.RECALCULATE_CYCLES):
        raise ValueError("GITT/ICI analysis requires an elaboration generation")
    manifest = GenerationManifest.from_dict(
        json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    )
    content_hash = manifest.content_hash
    if content_hash is None:
        raise ValueError("the elaboration generation has no content fingerprint")
    if elaboration_fingerprint is not None and content_hash != elaboration_fingerprint:
        raise ValueError("elaboration_fingerprint does not match the generation")
    table = pq.read_table(root / "indexes" / "segments.parquet")
    columns = {name: table[name].to_numpy(zero_copy_only=False) for name in table.column_names}
    order = np.lexsort((columns["segment_id"], columns["first_row"]))
    columns = {name: values[order] for name, values in columns.items()}
    selected = np.ones(len(order), dtype=bool)
    if cycles:
        selected &= np.isin(columns["cycle"], np.asarray(cycles))
    if half_cycles:
        selected &= np.isin(columns["half_cycle"], np.asarray(half_cycles))
    chosen = np.flatnonzero(selected)
    keep = np.zeros(len(order), dtype=bool)
    for offset in range(-CONTEXT_SEGMENTS, CONTEXT_SEGMENTS + 1):
        positions = chosen + offset
        keep[positions[(positions >= 0) & (positions < len(order))]] = True
    kept = np.flatnonzero(keep)
    segments = tuple(
        CanonicalSegment(
            int(columns["segment_id"][index]),
            int(columns["first_row"][index]),
            int(columns["last_row"][index]),
            float(columns["first_time_s"][index]),
            float(columns["last_time_s"][index]),
            int(columns["cycle"][index]),
            int(columns["half_cycle"][index]),
            int(columns["step"][index]),
            int(columns["polarity"][index]),
        )
        for index in kept
    )
    samples: tuple[CanonicalSample, ...] = ()
    if segments:
        first_time = min(segment.first_time_s for segment in segments)
        last_time = max(segment.last_time_s for segment in segments)
        points = pq.read_table(
            root / "points" / "elaborated.parquet",
            columns=list(_POINT_COLUMNS),
            filters=[("time_s", ">=", first_time), ("time_s", "<=", last_time)],
        )
        rows = points["source_row"].to_numpy(zero_copy_only=False).astype(np.int64)
        starts = np.array([segment.first_row for segment in segments], dtype=np.int64)
        ends = np.array([segment.last_row for segment in segments], dtype=np.int64)
        owner = np.searchsorted(starts, rows, side="right") - 1
        inside = (owner >= 0) & (rows <= ends[np.maximum(owner, 0)])
        if not inside.all():
            points = points.filter(pa.array(inside))
        values = [points[name].to_pylist() for name in _POINT_COLUMNS]
        samples = tuple(CanonicalSample(*row) for row in zip(*values, strict=True))
    return CanonicalScope(
        root,
        content_hash,
        tuple(int(value) for value in cycles),
        tuple(int(value) for value in half_cycles),
        segments,
        samples,
        frozenset(int(columns["segment_id"][index]) for index in chosen),
    )
