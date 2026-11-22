"""Immutable result snapshots and non-blocking review transitions.

A `ResultSnapshot` holds at most one current record per logical `ResultIdentity` (impulse, method,
variant, OCV source). Review state (provisional/accepted/rejected/error), freshness (current/stale)
and advisory quality are orthogonal. Every transition returns a new snapshot; the content
fingerprint lets a caller reject a batch computed against an older snapshot (late results).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields, replace
from enum import StrEnum

from gregorithm.contracts import JsonValue
from gregorithm.identity import canonical_parameter_bytes, hash_parameters

from .contracts import DelacourtResult, DiffusionMethod, DiffusionResult, OcvSource
from .quality import QualityAssessment


class ResultState(StrEnum):
    """Review state of the current record of one identity."""

    PROVISIONAL = "provisional"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ERROR = "error"


class StaleBatchError(RuntimeError):
    """Raised when a batch outcome is committed onto a snapshot other than its base."""


@dataclass(frozen=True, slots=True)
class AppliedSettings:
    """Separate editable draft values from the parameters governing current results.

    Editing changes only ``draft``; nothing scientific changes until `apply`. Mapping settings
    expose one dependency fingerprint per top-level key so that only dependent results are
    invalidated.
    """

    applied: JsonValue
    draft: JsonValue

    def edit(self, draft: JsonValue) -> AppliedSettings:
        """Return settings with a new draft and unchanged applied values."""
        return AppliedSettings(self.applied, draft)

    def apply(self) -> tuple[AppliedSettings, bool]:
        """Promote the draft and report whether the applied values changed."""
        changed = self.draft != self.applied
        return AppliedSettings(self.draft, self.draft), changed

    def dependency_fingerprints(self) -> dict[str, str]:
        """Return ``{"setting:<key>": fingerprint}`` for the applied values."""
        applied = self.applied
        if isinstance(applied, Mapping):
            return {
                f"setting:{key}": hash_parameters({key: value}) for key, value in applied.items()
            }
        return {"setting": hash_parameters({"value": applied})}


@dataclass(frozen=True, slots=True, order=True)
class ResultIdentity:
    """Logical identity replaced by recalculation while other identities coexist.

    ``event_id`` already binds the canonical generation fingerprint. ``variant`` distinguishes
    approved series of the same method, e.g. the Delacourt fixed/free offset.
    """

    event_id: str
    method: DiffusionMethod
    variant: str
    ocv_source: OcvSource


@dataclass(frozen=True, slots=True)
class EventCoordinates:
    """Branch and capacity coordinates of a result.

    D is placed at ``capacity_start_ah``, the canonical capacity at the start of the CC
    segment; the end capacity is kept as metadata. ``polarity`` separates charge and discharge.
    """

    cycle: int
    half_cycle: int
    polarity: int
    capacity_start_ah: float
    capacity_end_ah: float


class NonFiniteRecordError(ValueError):
    """Raised when a record carries NaN or infinity, which no snapshot can persist losslessly.

    Kernels report an unavailable quantity as None; a non-finite number therefore signals an
    event-local technical failure, which the batch stores as an error record.
    """


def _require_finite(value: object, path: str) -> None:
    """Reject NaN/infinity in result, quality and coordinate numbers, recursing into tuples."""
    if isinstance(value, float) and not math.isfinite(value):
        raise NonFiniteRecordError(f"{path} must be finite")
    if isinstance(value, tuple):
        for position, item in enumerate(value):
            _require_finite(item, f"{path}[{position}]")


def _json_details(value: object, path: str) -> JsonValue:
    """Return ``value`` as plain JSON (tuples become lists) so persistence round-trips exactly.

    Accepts the same types as the canonical parameter encoding; raises `NonFiniteRecordError`
    for NaN/infinity and `TypeError` for keys or values JSON cannot represent.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        _require_finite(value, path)
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{path} mapping keys must be strings")
        return {key: _json_details(item, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_details(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"{path} is not a JSON value: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class AnalysisRecord:
    """The current record of one identity.

    ``dependencies`` are the named fingerprints the record was computed from; `invalidate` marks it
    ``stale`` when one of them changes. ``details`` is the JSON technical panel: effective
    parameters, rows, windows, statistics, sources and reasons.
    """

    identity: ResultIdentity
    calculation_fingerprint: str
    state: ResultState
    result: DiffusionResult | DelacourtResult | None
    quality: QualityAssessment | None
    error: str | None = None
    dependencies: tuple[tuple[str, str], ...] = ()
    stale: bool = False
    details: JsonValue = None
    coordinates: EventCoordinates | None = None

    def __post_init__(self) -> None:
        """Enforce error/result exclusivity, unique dependencies and finite, JSON-only content.

        ``details`` is normalized to plain JSON here so that a persisted snapshot reopens equal to
        the published one; the snapshot fingerprint does not depend on it.
        """
        if not self.calculation_fingerprint:
            raise ValueError("calculation fingerprint must not be empty")
        if self.state is ResultState.ERROR:
            if self.result is not None or not self.error:
                raise ValueError("error records require an error and no result")
        elif self.result is None or self.error is not None:
            raise ValueError("non-error records require a result and no error")
        names = [name for name, _ in self.dependencies]
        if len(names) != len(set(names)):
            raise ValueError("record dependency names must be unique")
        for name, part in (
            ("result", self.result),
            ("quality", self.quality),
            ("coordinates", self.coordinates),
        ):
            if part is not None:
                for item in fields(part):
                    _require_finite(getattr(part, item.name), f"{name}.{item.name}")
        object.__setattr__(self, "details", _json_details(self.details, "details"))

    @property
    def diffusion_m2_s(self) -> float | None:
        """Return D in m²/s, or None when the record has no technically available D."""
        if self.result is None:
            return None
        return self.result.diffusion_m2_s

    @property
    def has_diffusion(self) -> bool:
        """Whether a finite D value is technically available."""
        value = self.diffusion_m2_s
        return value is not None and math.isfinite(value)

    @property
    def log_plottable(self) -> bool:
        """Whether D can be drawn on a logarithmic axis (finite and > 0)."""
        value = self.diffusion_m2_s
        return value is not None and math.isfinite(value) and value > 0

    @property
    def exportable(self) -> bool:
        """Accepted and current; provisional, rejected, stale and error records never qualify."""
        return self.state is ResultState.ACCEPTED and not self.stale


def _fingerprint(records: Iterable[AnalysisRecord]) -> str:
    """Hash identities, calculations, review states and freshness of records."""
    payload: JsonValue = [
        [
            record.identity.event_id,
            record.identity.method.value,
            record.identity.variant,
            record.identity.ocv_source.value,
            record.calculation_fingerprint,
            record.state.value,
            record.stale,
        ]
        for record in records
    ]
    return hashlib.sha256(
        b"gregorithm-gitt-ici-snapshot-v1\0" + canonical_parameter_bytes(payload)
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ResultSnapshot:
    """Current records sorted by identity; at most one record exists for each identity."""

    records: tuple[AnalysisRecord, ...] = ()
    _index: dict[ResultIdentity, int] = field(init=False, repr=False, compare=False)
    fingerprint: str = field(init=False, compare=False)

    def __post_init__(self) -> None:
        """Index identities, reject duplicates and compute the content fingerprint."""
        index = {record.identity: position for position, record in enumerate(self.records)}
        if len(index) != len(self.records):
            raise ValueError("snapshot contains duplicate result identities")
        object.__setattr__(self, "_index", index)
        object.__setattr__(self, "fingerprint", _fingerprint(self.records))

    @staticmethod
    def _build(records: Iterable[AnalysisRecord]) -> ResultSnapshot:
        """Return a snapshot with records sorted by identity."""
        return ResultSnapshot(tuple(sorted(records, key=lambda item: item.identity)))

    def current(self, identity: ResultIdentity) -> AnalysisRecord | None:
        """Return the current record of ``identity`` or None."""
        position = self._index.get(identity)
        return None if position is None else self.records[position]

    def _require(self, identity: ResultIdentity) -> AnalysisRecord:
        """Return the current record of ``identity`` or raise KeyError."""
        record = self.current(identity)
        if record is None:
            raise KeyError(identity)
        return record

    def replace_many(self, records: Iterable[AnalysisRecord]) -> ResultSnapshot:
        """Replace several identities at once; see `replace_calculation` for the rules."""
        updated = dict(zip((item.identity for item in self.records), self.records, strict=True))
        for record in records:
            if record.state is not ResultState.ERROR:
                record = replace(record, state=ResultState.PROVISIONAL)
            updated[record.identity] = replace(record, stale=False)
        return self._build(updated.values())

    def replace_calculation(self, record: AnalysisRecord) -> ResultSnapshot:
        """Replace one identity; a successful recalculation is always provisional.

        An error record replaces the previous value too, so an old D never survives as a
        fallback. Other identities, including deselected methods, are untouched.
        """
        return self.replace_many((record,))

    def accept(self, identity: ResultIdentity) -> ResultSnapshot:
        """Accept one current record; quality colors never block."""
        record = self._require(identity)
        if record.state is ResultState.ERROR:
            raise ValueError("an error result cannot be accepted")
        if record.stale:
            raise ValueError("a stale result must be recalculated before acceptance")
        return self._with_state({identity}, ResultState.ACCEPTED)

    def accept_all(
        self,
        *,
        method: DiffusionMethod,
        variant: str,
        ocv_source: OcvSource,
    ) -> ResultSnapshot:
        """Accept every available current result of one method/configuration.

        Rejected, error, stale and technically unavailable (no D) records are skipped, so manual
        rejections are preserved; colors never filter. Repeating the call changes nothing.
        """
        targets = {
            record.identity
            for record in self.records
            if record.identity.method is method
            and record.identity.variant == variant
            and record.identity.ocv_source is ocv_source
            and record.state is ResultState.PROVISIONAL
            and not record.stale
            and record.has_diffusion
        }
        return self._with_state(targets, ResultState.ACCEPTED)

    def revoke(self, identity: ResultIdentity) -> ResultSnapshot:
        """Revoke one acceptance; the record stays consultable and may be re-accepted."""
        record = self._require(identity)
        if record.state is not ResultState.ACCEPTED:
            raise ValueError("only an accepted result can be revoked")
        return self._with_state({identity}, ResultState.PROVISIONAL)

    def reject(self, identity: ResultIdentity) -> ResultSnapshot:
        """Reject one result as in the DCIR review; it stays consultable and is never exported."""
        record = self._require(identity)
        if record.state is ResultState.ERROR:
            raise ValueError("an error result cannot be rejected")
        return self._with_state({identity}, ResultState.REJECTED)

    def invalidate(self, dependencies: Mapping[str, str]) -> ResultSnapshot:
        """Mark stale every record whose named dependency now has a different fingerprint.

        Only records carrying a supplied dependency name are affected (targeted invalidation); the
        values and review states stay consultable, but stale records are not exportable or
        acceptable until recalculated.
        """
        changed = False
        records = []
        for record in self.records:
            stored = dict(record.dependencies)
            if not record.stale and any(
                name in stored and stored[name] != value for name, value in dependencies.items()
            ):
                record = replace(record, stale=True)
                changed = True
            records.append(record)
        return ResultSnapshot(tuple(records)) if changed else self

    def exportable_records(self) -> tuple[AnalysisRecord, ...]:
        """Return the accepted current records, the only ones a later export may contain."""
        return tuple(record for record in self.records if record.exportable)

    def _with_state(self, identities: set[ResultIdentity], state: ResultState) -> ResultSnapshot:
        """Return a snapshot where the given identities have the given review state."""
        if not identities:
            return self
        return ResultSnapshot(
            tuple(
                replace(record, state=state) if record.identity in identities else record
                for record in self.records
            )
        )
