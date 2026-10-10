"""Reopen redesigned Parquet exports as bounded identity-free result handles."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

from ._result_tables import bounded_frame
from .execution import MemoryBudget


@dataclass(frozen=True)
class ExportedResult:
    """Lazy persistent Parquet result with its required calculation manifest."""

    path: Path
    report: dict
    _budget: object = field(default_factory=lambda: MemoryBudget(2 * 1024**3), repr=False)

    def iter_batches(self, batch_rows=65536, columns=None):
        """Yield bounded persisted result batches with an optional projection."""
        if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows < 1:
            raise ValueError("Positive batch_rows required")
        with pq.ParquetFile(self.path) as parquet:
            yield from parquet.iter_batches(batch_size=batch_rows, columns=columns)

    def table(self, max_rows=100000):
        """Materialize only a guarded small pandas result table."""
        return bounded_frame(self, max_rows)

    def plot(
        self,
        *,
        x,
        y,
        series_column=None,
        series_values=(),
        max_points=250000,
        mode="auto",
    ):
        """Plot explicitly selected exported columns without recalculation."""
        from .prepared_plotting import plot_compact_result

        return plot_compact_result(
            self,
            x=x,
            y=y,
            series_column=series_column,
            series_values=series_values,
            max_points=max_points,
            mode=mode,
        )

    def plot_progressive(
        self,
        *,
        x,
        y,
        series_column=None,
        series_values=(),
        max_points=250000,
        mode="auto",
    ):
        """Create an idle progressive plot session over persisted result columns."""
        from .prepared_plotting import plot_compact_result

        return plot_compact_result(
            self,
            x=x,
            y=y,
            series_column=series_column,
            series_values=series_values,
            progressive=True,
            max_points=max_points,
            mode=mode,
        )


def open_result(path, *, memory_budget=None):
    """Open a streamed Parquet export and require its JSON calculation manifest.

    Usage: ``saved = open_result("capacity.parquet"); saved.table()``. The operation
    never reopens or recalculates the measurement source. A missing or malformed
    sidecar is rejected because scientific settings and quality context would be lost.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise ValueError("Result manifest sidecar is required")
    try:
        report = json.loads(manifest_path.read_text(encoding="utf8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Result manifest sidecar is invalid") from error
    if not isinstance(report, dict) or not report.get("operation"):
        raise ValueError("Result manifest lacks an operation")
    with pq.ParquetFile(path):
        pass
    budget = memory_budget or MemoryBudget(2 * 1024**3)
    return ExportedResult(path, report, budget)
