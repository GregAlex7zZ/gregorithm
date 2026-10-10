"""Readable bounded multi-file coordination for redesigned prepared sources."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from threading import Lock

from .diagnostics import Cancelled
from .execution import CancellationToken, MemoryBudget

_OPERATIONS = {"totals", "capacity", "steps", "events", "dcir", "dqdv"}


@dataclass(frozen=True)
class CollectionProgress:
    """One readable coordinator update; callbacks run on the coordinating thread."""

    source_file: str | None
    operation: str
    status: str
    files_completed: int
    files_failed: int
    files_total: int


@dataclass
class PreparedCollectionRun(Mapping):
    """Per-file results and isolated failures from one collection operation."""

    operation: str
    results: dict
    failures: dict[str, Exception]
    settings: dict = field(default_factory=dict)

    def __getitem__(self, key):
        """Return one successful result by complete source basename."""
        return self.results[key]

    def __iter__(self):
        """Iterate successful complete source basenames in input order."""
        return iter(self.results)

    def __len__(self):
        """Return the number of successful files."""
        return len(self.results)

    @property
    def status(self):
        """Return complete, partial or failed without hiding individual exceptions."""
        if self.results and self.failures:
            return "partial"
        return "complete" if self.results else "failed"

    def close(self):
        """Close every successful result that owns temporary storage."""
        for result in self.results.values():
            close = getattr(result, "close", None)
            if close is not None:
                close()


@dataclass(frozen=True)
class PreparedCollection(Mapping):
    """Filename-keyed lazy prepared sources with bounded multi-file execution."""

    sources: dict

    def __getitem__(self, key):
        """Return one lazy prepared source by its complete basename."""
        return self.sources[key]

    def __iter__(self):
        """Iterate complete basenames in the user's original order."""
        return iter(self.sources)

    def __len__(self):
        """Return the number of prepared sources."""
        return len(self.sources)

    def run(
        self,
        operation,
        *,
        workers=1,
        managed_memory_bytes=2 * 1024**3,
        cancellation=None,
        on_progress=None,
        **settings,
    ):
        """Run one supported operation independently for every prepared file.

        Example: ``batch = cells.run("capacity", workers=2, include_energy=True)``.
        Worker threads are Windows-safe and receive separate managed budgets whose sum
        does not exceed ``managed_memory_bytes``. Results are never converted to pandas
        or transferred between processes. One failure does not discard other files.
        """
        if operation not in _OPERATIONS:
            raise ValueError(f"Unsupported collection operation: {operation}")
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer")
        if (
            isinstance(managed_memory_bytes, bool)
            or not isinstance(managed_memory_bytes, int)
            or managed_memory_bytes < 8 * 1024**2
        ):
            raise ValueError("managed_memory_bytes must be an integer of at least 8 MiB")
        token = cancellation or CancellationToken()
        active = min(workers, max(1, len(self.sources)))
        per_worker = managed_memory_bytes // active
        if per_worker < 8 * 1024**2:
            raise ValueError("Global memory budget provides less than 8 MiB per worker")
        callback_lock = Lock()
        results, failures = {}, {}
        completed = failed = 0

        def notify(name, status):
            """Internal helper for PreparedCollection.run.notify; receives only the bounded state owned by its containing operation."""
            if on_progress is not None:
                with callback_lock:
                    on_progress(
                        CollectionProgress(
                            name, operation, status, completed, failed, len(self.sources)
                        )
                    )

        def calculate(name, source):
            """Internal helper for PreparedCollection.run.calculate; receives only the bounded state owned by its containing operation."""
            token.check()
            method = getattr(source, operation)
            call_settings = dict(settings)
            if operation in {"totals", "capacity", "steps", "events", "dcir", "dqdv"}:
                call_settings.setdefault("memory_budget", MemoryBudget(per_worker))
            call_settings.setdefault("cancellation", token)
            return method(**call_settings)

        names = list(self.sources)
        notify(None, "running")
        if active == 1:
            for name in names:
                try:
                    result = calculate(name, self.sources[name])
                except Cancelled:
                    raise
                except Exception as error:
                    failures[name] = error
                    failed += 1
                    notify(name, "failed")
                else:
                    results[name] = result
                    completed += 1
                    notify(name, "complete")
        else:
            with ThreadPoolExecutor(max_workers=active, thread_name_prefix="gregorithm") as pool:
                pending = {pool.submit(calculate, name, self.sources[name]): name for name in names}
                while pending:
                    token.check()
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        name = pending.pop(future)
                        try:
                            results[name] = future.result()
                        except Cancelled:
                            token.cancel()
                            raise
                        except Exception as error:
                            failures[name] = error
                            failed += 1
                            notify(name, "failed")
                        else:
                            completed += 1
                            notify(name, "complete")
        ordered_results = {name: results[name] for name in names if name in results}
        ordered_failures = {name: failures[name] for name in names if name in failures}
        outcome = PreparedCollectionRun(
            operation, ordered_results, ordered_failures, dict(settings)
        )
        notify(None, outcome.status)
        return outcome
