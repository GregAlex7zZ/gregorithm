"""Cooperative cancellation and progress reporting for long-running operations.

Every streaming operation (`import_source`, `elaborate`, `analyze_capacity`, ...) accepts an
optional `progress` callable and an optional `cancel` token. The library never starts threads
itself: the caller decides where an operation runs and may cancel it from another thread.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Event
from typing import Protocol, runtime_checkable

from .contracts import OperationId
from .diagnostics import Diagnostic, DiagnosticCode, DiagnosticKind


@runtime_checkable
class CancelToken(Protocol):
    """Protocol for objects that tell an operation whether it should stop.

    Any object with an `is_cancelled()` method qualifies; `CancellationToken` is the ready-made
    thread-safe implementation. Cancellation is cooperative: operations poll the token between
    bounded batches and at publication boundaries, so stopping is prompt but not instantaneous.
    """

    def is_cancelled(self) -> bool:
        """Return True once the caller has requested cancellation."""


class CancellationToken:
    """Thread-safe, caller-owned cancellation token.

    Create one token per operation, pass it as `cancel=` and call `cancel()` from any thread. The
    operation then raises `OperationCancelled`; the previously published result stays current.
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        """Create a token in the not-cancelled state."""
        self._event = Event()

    def cancel(self) -> None:
        """Request cancellation; calling it more than once has no extra effect."""
        self._event.set()

    def is_cancelled(self) -> bool:
        """Return True after `cancel()` has been called."""
        return self._event.is_set()


class OperationCancelled(RuntimeError):
    """Raised when an operation stops because its cancel token was set.

    The `diagnostic` attribute carries an `OPERATION_CANCELLED` diagnostic naming the operation.
    Partially staged output is never published and may be removed with `cleanup_job`.
    """

    def __init__(self, operation: str) -> None:
        """Build the exception and its structured diagnostic."""
        self.diagnostic = Diagnostic(
            code=DiagnosticCode.OPERATION_CANCELLED,
            kind=DiagnosticKind.ERROR,
            message="The operation was cancelled by the caller.",
            operation=operation,
        )
        super().__init__(self.diagnostic.message)


def check_cancelled(token: CancelToken | None, operation: str) -> None:
    """Raise `OperationCancelled` if `token` is set; do nothing when it is None.

    Library code calls this between batches. It is public so that callers composing several
    operations can use the same cancellation convention.
    """
    if token is not None and token.is_cancelled():
        raise OperationCancelled(operation)


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """Immutable progress notification emitted by streaming operations.

    Attributes:
        operation: Operation that emitted the event.
        phase: Short, stable name of the current processing phase.
        completed: Units processed so far in this phase.
        total: Total units in this phase, or None when unknown in advance.
        unit: Unit of `completed`/`total`, for example ``"rows"`` or ``"bytes"``.

    Progress is informative only; it is not a performance guarantee and phases may repeat.
    """

    operation: OperationId
    phase: str
    completed: int
    total: int | None
    unit: str

    def __post_init__(self) -> None:
        """Reject empty names, negative counts and completed > total."""
        if not self.phase or not self.unit:
            raise ValueError("Progress phase and unit must not be empty")
        if self.completed < 0 or (self.total is not None and self.total < 0):
            raise ValueError("Progress values must not be negative")
        if self.total is not None and self.completed > self.total:
            raise ValueError("Completed progress exceeds total")


@runtime_checkable
class ProgressSink(Protocol):
    """Protocol for callables that receive `ProgressEvent` objects.

    A sink is called synchronously on the worker thread. Keep it fast and do not raise from it;
    GUI callers should forward events to their own event loop.
    """

    def __call__(self, event: ProgressEvent) -> None:
        """Receive one progress event."""


def report_progress(sink: Callable[[ProgressEvent], None] | None, event: ProgressEvent) -> None:
    """Deliver `event` to `sink` when a sink was supplied."""
    if sink is not None:
        sink(event)
