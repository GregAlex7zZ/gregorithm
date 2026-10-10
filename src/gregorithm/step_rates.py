"""Fixed-reference C-rate of already computed within-step time-weighted means."""

from dataclasses import dataclass, field, replace

import pyarrow as pa
import pyarrow.compute as pc

from ._result_tables import bounded_frame
from .diagnostics import BudgetExceeded
from .io import atomic_parquet


@dataclass(frozen=True)
class StepRates:
    """Lazy step summaries plus signed mean C-rate; no classification or source reread.

    Use steps.types('CC').c_rate(capacity=capacity).table(). The scalar reference
    was resolved from the full source before constructing this view; filtering
    displayed steps never changes it. Storage lifetime is shared with steps.
    """

    source: object = field(repr=False)
    reference: dict

    @property
    def _budget(self):
        """Share the originating step result's managed buffer budget."""
        return self.source._budget

    @property
    def report(self):
        """Expose the fixed reference once and keep classification/source provenance."""
        return {
            **self.source.report,
            "operation": "within_step_mean_c_rate",
            "reference": self.reference,
            "selected_step_types": self.source._types,
            "c_rate_unit": "h^-1",
        }

    def types(self, labels):
        """Select readable physical step types without changing reference or renumbering."""
        return replace(self, source=self.source.types(labels))

    def iter_batches(self, batch_rows=512):
        """Append mean C-rate and shared quality using bounded compact input batches."""
        if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows < 1:
            raise ValueError("Positive batch_rows required")
        rows = min(batch_rows, (self._budget.limit - self._budget.used - 2 * 1024**2) // 8192)
        if rows < 1:
            raise BudgetExceeded("Insufficient shared step-rate budget")
        reservation = rows * 4096
        self._budget.reserve(reservation)
        reader = self.source.iter_batches(batch_rows=rows)
        try:
            for block in reader:
                denominator = self.reference["capacity_mAh"]
                if denominator is None:
                    value = pa.nulls(block.num_rows, type=pa.float64())
                    quality = pa.array(["unavailable"] * block.num_rows, type=pa.string())
                else:
                    value = pc.divide(block.column("mean_current_mA"), denominator)
                    quality = block.column("current_quality")
                    if self.reference["quality"] == "estimated":
                        quality = pc.if_else(pc.equal(quality, "complete"), "estimated", quality)
                yield block.append_column("mean_c_rate_h_inv", value).append_column(
                    "c_rate_quality", quality
                )
        finally:
            reader.close()
            self._budget.release(reservation)

    def table(self, max_rows=100000):
        """Return a guarded table; displayed frames are caller-owned memory."""
        return bounded_frame(self, max_rows)

    def write_parquet(self, path):
        """Stream scaled means to a new Parquet without replacing source or existing results."""
        from pathlib import Path

        if Path(path).exists():
            raise FileExistsError("Choose a new mean-rate export path")
        return atomic_parquet(self.iter_batches(), path, manifest=self.report)

    def close(self):
        """Close shared step storage; all originating and selected views become unavailable."""
        self.source.close()
