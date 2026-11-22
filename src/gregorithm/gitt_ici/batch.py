"""Sequential bounded-memory batches with transactional snapshot semantics.

A batch stages one new record per task and never mutates the snapshot it started from. Recoverable
event-local errors, including results carrying NaN or infinity, become error records for that
identity only; any other exception is structural and aborts the whole batch.
Cancellation raises `OperationCancelled` before anything is returned or written to the cache.
The outcome is applied with `commit_batch`, which refuses an outcome computed against an older
snapshot so late results are discarded.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from gregorithm.contracts import JsonValue, OperationId
from gregorithm.identity import canonical_parameter_bytes
from gregorithm.progress import (
    CancelToken,
    ProgressEvent,
    ProgressSink,
    check_cancelled,
    report_progress,
)

from .cache import CacheWrite, IntermediateCache, cache_key
from .contracts import DelacourtResult, DiffusionResult
from .quality import QualityAssessment
from .review import (
    AnalysisRecord,
    EventCoordinates,
    NonFiniteRecordError,
    ResultIdentity,
    ResultSnapshot,
    ResultState,
    StaleBatchError,
)

ALGORITHM_VERSION = "gitt-ici-analysis-v1"
"""Version of the GITT/ICI calculation chain; part of every cache key."""


class RecoverableAnalysisError(RuntimeError):
    """An event-local technical error that replaces that identity and lets the batch continue."""


@dataclass(frozen=True, slots=True)
class ComputeOutput:
    """What one task computes: the result plus its quality, technical details and coordinates.

    ``cache_value`` is an optional JSON intermediate to persist after a successful batch; it must
    not contain NaN or infinity and never carries review state.
    """

    result: DiffusionResult | DelacourtResult
    quality: QualityAssessment | None = None
    details: JsonValue = None
    coordinates: EventCoordinates | None = None
    cache_value: JsonValue = None


type ComputeCallback = Callable[[JsonValue], ComputeOutput]
"""Task body; receives the cached intermediate (or None) and returns a `ComputeOutput`."""


@dataclass(frozen=True, slots=True)
class BatchTask:
    """One identity to (re)compute with its named dependency fingerprints and parameters.

    ``dependencies`` must name the canonical ``generation`` so cache entries and records can never
    be reused across generations; they also drive targeted invalidation of the record.
    """

    identity: ResultIdentity
    dependencies: tuple[tuple[str, str], ...]
    parameters: JsonValue
    compute: ComputeCallback

    def __post_init__(self) -> None:
        """Require unique dependency names including ``generation``."""
        names = [name for name, _ in self.dependencies]
        if len(names) != len(set(names)):
            raise ValueError("task dependency names must be unique")
        if "generation" not in names:
            raise ValueError("task dependencies must include the canonical generation")

    def dependency_map(self) -> dict[str, str]:
        """Return the dependencies as a dictionary."""
        return dict(self.dependencies)


@dataclass(frozen=True, slots=True)
class BatchOutcome:
    """Staged result of a completed batch.

    ``snapshot`` is the base snapshot with the new records applied; ``records`` are those new
    records; ``base_fingerprint`` identifies the snapshot the batch started from. ``completed``
    counts tasks and ``impulses`` distinct events.
    """

    snapshot: ResultSnapshot
    completed: int
    recoverable_errors: int
    cache_hits: int
    base_fingerprint: str = ""
    records: tuple[AnalysisRecord, ...] = ()
    impulses: int = 0


def run_batch(
    tasks: Sequence[BatchTask],
    *,
    previous: ResultSnapshot,
    cache: IntermediateCache | None = None,
    progress: ProgressSink | None = None,
    cancel: CancelToken | None = None,
) -> BatchOutcome:
    """Compute all tasks one at a time and return the staged outcome only after the last one.

    Memory is bounded by one task's working set plus the staged immutable records. Progress is
    reported in impulses (distinct event IDs in task order) with the phase
    ``calculating:<method>``. Tasks for the same identity twice are a structural error.
    """
    identities = [task.identity for task in tasks]
    if len(identities) != len(set(identities)):
        raise ValueError("a batch must not contain the same identity twice")
    last_task_of_event = {task.identity.event_id: index for index, task in enumerate(tasks)}
    total_impulses = len(last_task_of_event)
    staged: list[AnalysisRecord] = []
    cache_writes: list[CacheWrite] = []
    errors = 0
    hits = 0
    done_impulses = 0
    operation = OperationId.GITT_ICI.value
    report_progress(
        progress, ProgressEvent(OperationId.GITT_ICI, "starting", 0, total_impulses, "impulses")
    )
    for index, task in enumerate(tasks):
        check_cancelled(cancel, operation)
        phase = f"calculating:{task.identity.method.value}"
        report_progress(
            progress,
            ProgressEvent(OperationId.GITT_ICI, phase, done_impulses, total_impulses, "impulses"),
        )
        dependencies = task.dependency_map()
        key = cache_key(
            f"{task.identity.method.value}:{ALGORITHM_VERSION}", dependencies, task.parameters
        )
        cached = None if cache is None else cache.get(key, dependencies)
        if cached is not None:
            hits += 1
        try:
            output = task.compute(cached)
            record = AnalysisRecord(
                task.identity,
                key,
                ResultState.PROVISIONAL,
                output.result,
                output.quality,
                dependencies=task.dependencies,
                details=output.details,
                coordinates=output.coordinates,
            )
            if output.cache_value is not None:
                canonical_parameter_bytes(output.cache_value)
                cache_writes.append(
                    CacheWrite(key, tuple(sorted(dependencies.items())), output.cache_value)
                )
        except (RecoverableAnalysisError, NonFiniteRecordError) as error:
            errors += 1
            record = AnalysisRecord(
                task.identity,
                key,
                ResultState.ERROR,
                None,
                None,
                str(error) or type(error).__name__,
                dependencies=task.dependencies,
            )
        staged.append(record)
        if last_task_of_event[task.identity.event_id] == index:
            done_impulses += 1
        check_cancelled(cancel, operation)
        report_progress(
            progress,
            ProgressEvent(OperationId.GITT_ICI, phase, done_impulses, total_impulses, "impulses"),
        )
    check_cancelled(cancel, operation)
    if cache is not None:
        cache.commit(tuple(cache_writes))
    report_progress(
        progress,
        ProgressEvent(OperationId.GITT_ICI, "complete", total_impulses, total_impulses, "impulses"),
    )
    return BatchOutcome(
        previous.replace_many(staged),
        len(tasks),
        errors,
        hits,
        previous.fingerprint,
        tuple(staged),
        total_impulses,
    )


def commit_batch(current: ResultSnapshot, outcome: BatchOutcome) -> ResultSnapshot:
    """Apply a completed batch only if ``current`` is still the snapshot it started from.

    Any review action or invalidation since the batch started changes the fingerprint; the batch
    is then a late result and `StaleBatchError` is raised so the caller discards it.
    """
    if current.fingerprint != outcome.base_fingerprint:
        raise StaleBatchError("the batch was computed against an older result snapshot")
    return current.replace_many(outcome.records)
