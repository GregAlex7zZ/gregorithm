"""Fixed-reference signed C-rate without measurement IDs or full-file materialization.

Use data.c_rate(capacity=capacity).preview() to reuse existing half-cycle results,
or data.c_rate(reference_mAh=1.0) for an explicit absolute capacity. Current is mA,
reference capacity mAh, and C-rate inverse hours. No CC/CV classification occurs.
"""

import math
import sqlite3
import warnings
from contextlib import closing
from dataclasses import dataclass, field, replace
from numbers import Integral
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .diagnostics import BudgetExceeded, MaterializationLimitError, SourceChangedError
from .execution import CancellationToken, MemoryBudget
from .half_cycles import HalfCycleConfig, HalfCycleResult
from .io import atomic_parquet, fingerprint
from .point_calculations import (
    EstimatedMeasurementWarning,
    SourceTotals,
    _numeric,
    estimate_isolated,
)


class UnavailableReferenceWarning(UserWarning):
    """The requested fixed reference cannot support C-rate; no substitute is selected."""


def _positive_number(value, name):
    """Validate a manual absolute capacity before performing any scientific scan."""
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _positive_label(value, name):
    """Accept readable Python/NumPy segment numbers, never booleans or fractional labels."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _reference(source, capacity, config, cycle, half, current_column, budget):
    """Read one indexed unnormalized reference row from the entire source result."""
    if not isinstance(capacity, HalfCycleResult):
        raise TypeError("capacity must be an unnormalized HalfCycleResult")
    if capacity._source_path != source.path.resolve() or capacity.report[
        "prepared_source_metadata"
    ] != fingerprint(source.path):
        raise SourceChangedError(
            "Capacity reference does not belong to this unchanged prepared source"
        )
    if capacity.report["settings"] != vars(config):
        raise ValueError("C-rate configuration must match the supplied capacity result")
    if capacity.report["measurement_columns"]["current"] != current_column:
        raise ValueError("Current column must match the supplied capacity result")
    if not capacity.path.is_file():
        raise ValueError("Capacity reference result is closed")
    reservation = 2 * 1024**2
    budget.reserve(reservation)
    try:
        with closing(sqlite3.connect(capacity.path)) as database:
            database.execute("PRAGMA cache_size=-2048")
            database.execute("PRAGMA temp_store=FILE")
            direction = "discharge" if config.cycle_order == "charge_then_discharge" else "charge"
            if half is not None:
                row = database.execute(
                    "SELECT half_cycle,cycle,direction,capacity_mAh,capacity_quality FROM summaries WHERE half_cycle=?",
                    (half,),
                ).fetchone()
            else:
                row = database.execute(
                    "SELECT half_cycle,cycle,direction,capacity_mAh,capacity_quality FROM summaries WHERE cycle=? AND direction=?",
                    (cycle, direction),
                ).fetchone()
    finally:
        budget.release(reservation)
    reference = {
        "method": "selected_half_cycle" if half is not None else "cycle_closing_half",
        "requested_cycle": cycle if half is None else None,
        "requested_half_cycle": half,
        "capacity_mAh": None,
        "quality": "unavailable",
        "reason": "missing_reference",
    }
    if row is not None:
        h, c, direction, value, quality = row
        reference.update(half_cycle=h, cycle=c, direction=direction)
        if quality not in {"complete", "estimated"}:
            reference["reason"] = "incomplete_reference_support"
        elif value is None or not math.isfinite(value) or value <= 0:
            reference["reason"] = "nonpositive_or_unavailable_reference"
        else:
            reference.update(capacity_mAh=value, quality=quality, reason=None)
    return reference


@dataclass(frozen=True)
class CRateView:
    """Lazy original measurements plus one signed C-rate column and metadata once.

    preview() defaults to five rows. table() refuses excessive materialization.
    iter_batches()/write_parquet() process the entire file under a shared budget.
    Isolated missing current is estimated only for C-rate; the original stays NaN.
    """

    source: object = field(repr=False)
    reference: dict
    config: HalfCycleConfig = field(repr=False)
    current_column: str
    _budget: MemoryBudget = field(repr=False)
    _cancellation: CancellationToken = field(repr=False)
    _fingerprint: dict = field(repr=False)

    @property
    def report(self):
        """Expose fixed reference, units and missing-point policy without per-row metadata."""
        return {
            "source_file": self.source.report["source_file"],
            "operation": "point_c_rate",
            "reference": self.reference,
            "units": {"current": "mA", "reference_capacity": "mAh", "c_rate": "h^-1"},
            "current_column": self.current_column,
            "missing_span_s": self.config.missing_span_s,
            "generated_measurement_identifiers": False,
            "prepared_source_metadata": self._fingerprint,
        }

    def iter_batches(self, columns=None):
        """Append C-rate to all original columns, or to an explicitly requested projection."""
        token, budget = self._cancellation, self._budget
        token.check()
        if fingerprint(self.source.path) != self._fingerprint:
            raise SourceChangedError("Prepared source changed before C-rate reading")
        time_column = self.source.report["time_column"]
        with pq.ParquetFile(self.source.path) as parquet:
            schema = parquet.schema_arrow
        if "c_rate_h_inv" in schema.names:
            raise ValueError("C-rate would overwrite existing column c_rate_h_inv")
        requested = schema.names if columns is None else list(columns)
        if len(set(requested)) != len(requested) or any(
            name not in schema.names for name in requested
        ):
            raise ValueError("Projection must contain distinct existing source columns")
        read_columns = list(dict.fromkeys([*requested, time_column, self.current_column]))
        reader_bytes = min(
            self.source.read_options.max_batch_bytes, (budget.limit - budget.used) // 16
        )
        if reader_bytes < 64:
            raise BudgetExceeded("Insufficient shared C-rate reader budget")
        reservation = reader_bytes * 16
        budget.reserve(reservation)
        source = replace(
            self.source,
            read_options=replace(self.source.read_options, max_batch_bytes=reader_bytes),
        )
        reader, previous, alerted = source.iter_batches(columns=read_columns), None, False
        try:
            block = next(reader, None)
            emitted = False
            while block is not None:
                token.check()
                following = next(reader, None)
                if not block.num_rows:
                    block = following
                    continue
                times, raw = _numeric(block, time_column), _numeric(block, self.current_column)
                after = (
                    None
                    if following is None
                    else (
                        _numeric(following.slice(0, 1), time_column)[0],
                        _numeric(following.slice(0, 1), self.current_column)[0],
                    )
                )
                offset = int(previous is not None)
                padded_t = np.r_[
                    [] if previous is None else [previous[0]],
                    times,
                    [] if after is None else [after[0]],
                ]
                padded_i = np.r_[
                    [] if previous is None else [previous[1]],
                    raw,
                    [] if after is None else [after[1]],
                ]
                effective, estimated = estimate_isolated(
                    padded_t, padded_i, self.config.missing_span_s
                )
                effective = effective[offset : offset + len(times)]
                estimated = estimated[offset : offset + len(times)]
                if self.reference["capacity_mAh"] is not None and np.any(estimated) and not alerted:
                    warnings.warn(
                        "C-rate uses isolated-current estimates; original measurements unchanged",
                        EstimatedMeasurementWarning,
                        stacklevel=2,
                    )
                    alerted = True
                denominator = self.reference["capacity_mAh"]
                values = (
                    pa.nulls(len(times), type=pa.float64())
                    if denominator is None
                    else pa.array(effective / denominator, mask=~np.isfinite(effective))
                )
                yield block.select(requested).append_column("c_rate_h_inv", values)
                emitted = True
                previous = times[-1], raw[-1]
                block = following
            if not emitted:
                yield pa.RecordBatch.from_arrays(
                    [pa.array([], type=schema.field(name).type) for name in requested]
                    + [pa.array([], type=pa.float64())],
                    names=requested + ["c_rate_h_inv"],
                )
            token.check()
            if fingerprint(self.source.path) != self._fingerprint:
                raise SourceChangedError("Prepared source changed during C-rate reading")
        finally:
            reader.close()
            budget.release(reservation)

    def _table(self, rows, truncate):
        """Collect only a guarded preview/table; never silently truncate a full table."""
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
            raise ValueError("Nonnegative row limit required")
        allowance = min(64 * 1024**2, self._budget.limit // 4)
        self._budget.reserve(allowance)
        reader, blocks, count, size = self.iter_batches(), [], 0, 0
        try:
            for block in reader:
                if truncate:
                    block = block.slice(0, max(0, rows - count))
                count += block.num_rows
                size += block.nbytes
                if count > rows or size * 3 > allowance:
                    raise MaterializationLimitError(
                        "C-rate table exceeds limits; use iter_batches() or write_parquet()"
                    )
                blocks.append(block)
                if truncate and count >= rows:
                    break
            return pa.Table.from_batches(blocks).to_pandas()
        finally:
            reader.close()
            self._budget.release(allowance)

    def preview(self, rows=5):
        """Return at most rows measurements with C-rate; preview is explicitly truncated."""
        if isinstance(rows, bool) or not isinstance(rows, int) or not 0 <= rows <= 100000:
            raise ValueError("Preview rows must be between 0 and 100000")
        return self._table(rows, True)

    def table(self, max_rows=100000):
        """Return a guarded whole-source pandas table; use streaming for large files."""
        return self._table(max_rows, False)

    def write_parquet(self, path, columns=None):
        """Stream original columns plus C-rate and save reference metadata once."""
        if Path(path).resolve() == self.source.path.resolve():
            raise ValueError("C-rate output must not overwrite its prepared source")
        if Path(path).exists():
            raise FileExistsError("Choose a new C-rate output path; existing files are preserved")
        return atomic_parquet(self.iter_batches(columns=columns), path, manifest=self.report)

    def mean(self):
        """Return time-weighted signed current/C-rate over valid whole-source support.

        Uses exact point-current integration divided by covered seconds; unknown
        intervals remain excluded and quality states that the mean has partial support.
        This generic mean does not classify CC steps or omit switching fronts.
        """
        if fingerprint(self.source.path) != self._fingerprint:
            raise SourceChangedError("Prepared source changed before C-rate averaging")
        totals = self.source.totals(
            include_energy=False,
            current_column=self.current_column,
            missing_span_s=self.config.missing_span_s,
            memory_budget=self._budget,
            cancellation=self._cancellation,
        )
        values = totals.values
        covered = values["capacity_covered_s"]
        current = (
            None
            if covered <= 0 or values["net_mAh"] is None
            else values["net_mAh"] * 3600 / covered
        )
        denominator = self.reference["capacity_mAh"]
        rate = None if current is None or denominator is None else current / denominator
        quality = (
            "unavailable"
            if rate is None
            else "estimated"
            if values["capacity_quality"] == "complete" and self.reference["quality"] == "estimated"
            else values["capacity_quality"]
        )
        return SourceTotals(
            {
                "mean_current_mA": current,
                "mean_c_rate_h_inv": rate,
                "covered_s": covered,
                "duration_s": values["duration_s"],
                "current_quality": values["capacity_quality"],
                "c_rate_quality": quality,
            },
            {
                **self.report,
                "operation": "time_weighted_c_rate_mean",
                "scope": "entire prepared source",
                "integration_counts": totals.report["counts"],
            },
        )


def point_c_rate(
    source,
    *,
    reference_mAh=None,
    reference_cycle=2,
    reference_half_cycle=None,
    capacity=None,
    config=None,
    current_column="current_mA",
    memory_budget=None,
    cancellation=None,
):
    """Create a fixed-reference streaming C-rate view; automatic reference has no fallback.

    Reuse capacity=data.capacity() to avoid another segmentation/integration scan.
    Selected capacity views still resolve the reference from the underlying source.
    reference_cycle selects its closing half; reference_half_cycle selects an exact
    half-cycle. Manual reference_mAh must be unnormalized positive absolute mAh.
    """
    cycle = _positive_label(reference_cycle, "reference_cycle")
    half = (
        None
        if reference_half_cycle is None
        else _positive_label(reference_half_cycle, "reference_half_cycle")
    )
    if reference_mAh is not None and (half is not None or cycle != 2):
        raise ValueError("Manual and selected-segment references are mutually exclusive")
    if half is not None and cycle != 2:
        raise ValueError("Choose reference_cycle or reference_half_cycle, not both")
    cfg = config or (
        HalfCycleConfig(**capacity.report["settings"])
        if isinstance(capacity, HalfCycleResult)
        else HalfCycleConfig()
    )
    if not isinstance(cfg, HalfCycleConfig):
        raise TypeError("config must be HalfCycleConfig")
    budget = memory_budget or (
        capacity._budget if isinstance(capacity, HalfCycleResult) else MemoryBudget(2 * 1024**3)
    )
    token = cancellation or CancellationToken()
    token.check()
    if reference_mAh is not None:
        reference = {
            "method": "manual",
            "capacity_mAh": _positive_number(reference_mAh, "reference_mAh"),
            "quality": "complete",
            "reason": None,
        }
    else:
        owned = capacity is None
        if owned:
            capacity = source.capacity(
                config=cfg, current_column=current_column, memory_budget=budget, cancellation=token
            )
        try:
            reference = _reference(source, capacity, cfg, cycle, half, current_column, budget)
        finally:
            if owned:
                capacity.close()
        if reference["capacity_mAh"] is None:
            warnings.warn(
                f"C-rate reference unavailable: {reference['reason']}; no fallback selected",
                UnavailableReferenceWarning,
                stacklevel=2,
            )
    return CRateView(
        source, reference, cfg, current_column, budget, token, fingerprint(source.path)
    )
