"""Synthetic regression tests; run with tools/run_tests.py. Passing verifies software behavior, not experimental electrochemistry."""

import time

import numpy as np
import pyarrow as pa

from gregorithm.execution import CancellationToken
from gregorithm.io import TableView
from gregorithm.plot_spec import PlotSpec, iter_plot_updates


def view(n=1000, batch=7):
    """Construct a lazy chunked synthetic plotting view so reduction can be tested across different batch boundaries."""

    def factory():
        """Yield bounded synthetic or captured helper batches; the containing test/tool defines the source and expected outcome."""
        for start in range(0, n, batch):
            ids = np.arange(start, min(n, start + batch))
            yield pa.RecordBatch.from_pydict(
                {
                    "time_s": ids.astype(float),
                    "voltage_V": np.sin(ids * 0.1),
                    "sample_id": ids,
                    "trace_id": ["s"] * len(ids),
                }
            )

    return TableView(factory, {"trace_id": "s"})


def final(updates):
    """Reconstruct final display-series payloads from replacement/append updates for assertions without browser rendering."""
    series = {}
    for u in updates:
        if u.operation == "replace":
            series[u.plot_trace_id] = u.data
        elif u.operation == "append":
            for k in ("x", "y", "ids", "aggregate"):
                series[u.plot_trace_id][k] += u.data[k]
    return series


def test_full_and_partial():
    """Verify full and partial using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    updates = list(
        iter_plot_updates(view(30), spec=PlotSpec(plot_mode="full", max_points_per_figure=100))
    )
    assert updates[-1].status == "complete"
    assert next(iter(final(updates).values()))["ids"] == list(range(30))
    updates = list(
        iter_plot_updates(view(300), spec=PlotSpec(plot_mode="full", max_points_per_figure=100))
    )
    assert updates[-1].status == "partial" and "PlotBudgetExceeded" in updates[-1].reason


def test_downsample_deterministic_global_budget():
    """Verify downsample deterministic global budget using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    results = []
    for batch in (1, 7, 1024):
        updates = list(
            iter_plot_updates(
                [view(1000, batch), view(1000, batch)],
                spec=PlotSpec(max_points_per_figure=100, max_points_per_trace=100),
            )
        )
        assert updates[-1].status == "complete" and updates[-1].points_represented <= 100
        data = final(updates)
        results.append([v["ids"] for v in data.values()])
        for d in data.values():
            assert d["ids"] == sorted(d["ids"])
    assert results[0] == results[1] == results[2]


def test_aggregate_cancel_and_figurewidget():
    """Verify aggregate cancel and figurewidget using explicit synthetic fixtures and assertions. Run through pytest; this is not experimental validation."""
    updates = list(
        iter_plot_updates(view(), spec=PlotSpec(plot_mode="aggregate", max_points_per_figure=100))
    )
    d = next(iter(final(updates).values()))
    assert all(x is None for x in d["ids"])
    assert all(x["count"] > 0 for x in d["aggregate"])
    token = CancellationToken()
    token.cancel()
    assert list(iter_plot_updates(view(), cancellation=token))[-1].status == "cancelled"
    from gregorithm.plotting import plot, plot_progressive

    result = plot(view(30))
    assert len(result.figure.data[0].x) == 30
    session = plot_progressive(view(30))
    assert len(session.figure.data) == 0
    session.start()
    deadline = time.monotonic() + 10
    while (session.thread.is_alive() or not session.queue.empty()) and time.monotonic() < deadline:
        session.poll()
        time.sleep(0.01)
    assert session.status == "complete" and len(session.figure.data[0].x) == 30
    session.close()
    assert session.poll() == []
