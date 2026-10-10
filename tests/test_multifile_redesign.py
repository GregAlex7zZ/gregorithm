"""Regressions for readable bounded redesigned multi-file coordination."""

from threading import current_thread

import pytest

from gregorithm import CancellationToken, PreparedCollection
from gregorithm.diagnostics import Cancelled


class _Result:
    def __init__(self, name):
        self.name = name
        self.closed = False

    def close(self):
        self.closed = True


class _Source:
    def __init__(self, name, *, fail=False):
        self.name = name
        self.fail = fail
        self.budget_limits = []

    def capacity(self, *, memory_budget, cancellation, **settings):
        cancellation.check()
        self.budget_limits.append(memory_budget.limit)
        if self.fail:
            raise ValueError("isolated failure")
        return _Result(self.name)


def test_parallel_results_failures_budget_and_progress_are_readable():
    """Keep successful files when a peer fails and divide one global worker budget."""
    good_a = _Source("a")
    bad = _Source("bad", fail=True)
    good_b = _Source("b")
    collection = PreparedCollection({"a.parquet": good_a, "bad.csv": bad, "b.csv": good_b})
    progress = []
    coordinator = current_thread().ident
    run = collection.run(
        "capacity",
        workers=2,
        managed_memory_bytes=64 * 1024**2,
        on_progress=lambda event: progress.append((current_thread().ident, event)),
    )
    assert run.status == "partial"
    assert list(run) == ["a.parquet", "b.csv"]
    assert list(run.failures) == ["bad.csv"]
    assert all(source.budget_limits == [32 * 1024**2] for source in (good_a, bad, good_b))
    assert all(thread == coordinator for thread, _ in progress)
    assert progress[-1][1].status == "partial"
    run.close()
    assert all(result.closed for result in run.results.values())


def test_cancelled_collection_does_not_become_file_failure():
    """Propagate cooperative cancellation instead of reporting a misleading failed file."""
    token = CancellationToken()
    token.cancel()
    collection = PreparedCollection({"cell.parquet": _Source("cell")})
    with pytest.raises(Cancelled):
        collection.run("capacity", cancellation=token)


@pytest.mark.parametrize("operation", ["unknown", "plot"])
def test_unsupported_collection_operation_is_explicit(operation):
    """Reject accidental arbitrary method dispatch before starting workers."""
    with pytest.raises(ValueError, match="Unsupported"):
        PreparedCollection({}).run(operation)
