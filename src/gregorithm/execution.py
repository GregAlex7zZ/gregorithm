"""Bounded progress, cancellation, and managed-memory accounting."""

from dataclasses import asdict, dataclass
from threading import Event
from time import monotonic

from .diagnostics import BudgetExceeded, Cancelled


class CancellationToken:
    """Thread/process-compatible cooperative cancellation flag. Pass one to analyze()/iter_plot_updates(), then call cancel() from the controlling context."""

    def __init__(self, event=None):
        """Initialize the state required by this object. Use the public factory or dataclass configuration unless constructing this internal component deliberately."""
        self._event = event if event is not None else Event()

    def cancel(self):
        """Request cooperative cancellation; completed analytical results remain available and interrupted work is not marked complete."""
        self._event.set()

    @property
    def cancelled(self):
        """Return whether the shared thread/process cancellation event is set."""
        return self._event.is_set()

    def check(self):
        """Raise Cancelled at a cooperative boundary when cancellation was requested."""
        if self.cancelled:
            raise Cancelled("Operation cancelled")


@dataclass(frozen=True)
class ProgressEvent:
    """Serializable operation/trace progress and terminal state. Running callbacks are throttled; completion, partial, failure, and cancellation remain distinct."""

    operation_id: str
    phase: str
    status: str = "running"
    trace_id: str | None = None
    elapsed_s: float = 0
    rows_read: int = 0
    rows_selected: int = 0
    rows_processed: int = 0
    batches_completed: int = 0
    total: int | None = None
    percent: float | None = None
    eta_s: float | None = None
    files_completed: int = 0
    files_failed: int = 0
    files_total: int | None = None

    def to_dict(self):
        """Return serializable configuration/provenance fields without materializing signal data."""
        return asdict(self)


class Reporter:
    """Internal throttled ProgressEvent publisher; callbacks execute in the coordinating context, not a GUI worker."""

    def __init__(self, callback, operation_id, trace_id=None):
        """Initialize the state required by this object. Use the public factory or dataclass configuration unless constructing this internal component deliberately."""
        self.callback = callback
        self.operation_id = operation_id
        self.trace_id = trace_id
        self.start = monotonic()
        self.last = -1e100

    def emit(self, phase, status="running", **kw):
        """Publish a ProgressEvent at most five times per second for running work; terminal statuses bypass throttling."""
        now = monotonic()
        if self.callback and (status != "running" or now - self.last >= 0.2):
            self.last = now
            self.callback(
                ProgressEvent(
                    self.operation_id,
                    phase,
                    status,
                    kw.pop("trace_id", self.trace_id),
                    now - self.start,
                    **kw,
                )
            )


class MemoryBudget:
    """Shared accounting for managed buffers; runtime RSS is measured separately."""

    def __init__(self, limit):
        """Initialize the state required by this object. Use the public factory or dataclass configuration unless constructing this internal component deliberately."""
        self.limit = limit
        self.used = 0

    def reserve(self, n):
        """Reserve n managed bytes or raise BudgetExceeded before crossing the accounting limit."""
        if n < 0 or self.used + n > self.limit:
            raise BudgetExceeded("Global managed-buffer budget exceeded")
        self.used += n

    def release(self, n):
        """Release previously reserved managed bytes; this accounting does not measure process RSS."""
        if n < 0 or n > self.used:
            raise ValueError("Cannot release unreserved managed bytes")
        self.used -= n
