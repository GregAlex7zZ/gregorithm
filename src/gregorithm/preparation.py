"""Prepare sources without generated identities or whole-file RAM materialization.

Use ``prepare_source(input_path, output_path)`` to obtain a validated, lazy
Parquet source. Timestamp duplicates retain their first complete original row.
The temporary on-disk timestamp index contains only original measurement times,
not generated row identifiers. Original files and retained columns are preserved.
"""

from __future__ import annotations

import tempfile
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import monotonic

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .diagnostics import BudgetExceeded, SchemaError, SourceChangedError
from .execution import CancellationToken, MemoryBudget
from .io import ReadOptions, atomic_parquet, fingerprint, source_batches


class TimestampOrderError(SchemaError):
    """A distinct retained measurement time decreases; source analysis must stop."""


class DuplicateTimestampWarning(UserWarning):
    """Later rows at an already retained timestamp were removed as configured."""


@dataclass(frozen=True)
class PreparationProgress:
    """Readable progress without technical IDs; callbacks receive bounded counts."""

    source_file: str
    rows_read: int
    rows_retained: int
    duplicates_removed: int
    status: str = "running"


@dataclass(frozen=True)
class PreparedSource:
    """Lazy prepared Parquet handle; use preview() or iter_batches(), not a full dataframe.

    Metadata is stored once in ``report`` and the JSON sidecar, never repeated
    per measurement. totals(), capacity() and c_rate() implement the redesigned
    point calculations; optional step/event/plot operations remain separate.
    """

    path: Path
    report: dict
    read_options: ReadOptions

    def iter_batches(self, columns=None):
        """Yield bounded original-column Arrow batches; columns is an explicit projection."""
        yield from source_batches(self.path, self.read_options, columns=columns)

    def preview(self, rows=5):
        """Return up to rows original measurements as pandas; limited to 100000 rows.

        The limit prevents accidentally materializing a GB-scale source. Use
        iter_batches() for larger processing. Previewing never changes the source.
        """
        if isinstance(rows, bool) or not isinstance(rows, int) or not 0 <= rows <= 100000:
            raise ValueError("preview rows must be an integer between 0 and 100000")
        with pq.ParquetFile(self.path) as parquet:
            schema = parquet.schema_arrow
        blocks = []
        retained = 0
        size = 0
        reader = self.iter_batches()
        try:
            if rows:
                for block in reader:
                    part = block.slice(0, rows - retained)
                    size += part.nbytes
                    if size > self.read_options.max_batch_bytes:
                        raise BudgetExceeded("Preview exceeds the bounded reader byte budget")
                    blocks.append(part)
                    retained += part.num_rows
                    if retained >= rows:
                        break
        finally:
            reader.close()
        return pa.Table.from_batches(blocks, schema=schema).to_pandas()

    def totals(self, **settings):
        """Calculate global charge/discharge/net capacity and energy; return a small result.

        Usage: data.totals().table(). These are whole-source totals, not per-cycle
        capacities. Use include_energy=False for capacity-only work. See
        point_calculations.source_totals for units, missing-point rules and budgets.
        """
        from .point_calculations import source_totals

        return source_totals(self, **settings)

    def capacity(self, **settings):
        """Return disk-backed net capacity per confirmed half-cycle; use .table() or .details().

        include_energy=True adds net energy without changing source columns.
        See half_cycles.half_cycle_capacity for confirmation, ownership and budgets.
        """
        from .half_cycles import half_cycle_capacity

        diagnostic_events = settings.pop("diagnostic_events", None)
        if diagnostic_events is not None:
            if "diagnostic_windows" in settings:
                raise ValueError("Choose diagnostic_events or diagnostic_windows")
            if Path(diagnostic_events.source.path).resolve() != self.path.resolve():
                raise ValueError("Diagnostic events belong to a different prepared source")
            settings["diagnostic_windows"] = diagnostic_events.diagnostic_windows()
        return half_cycle_capacity(self, **settings)

    def cycles(self, numbers):
        """Request original source cycles; example: data.cycles([2,3]).capacity().table()."""
        from .half_cycles import PreparedCycleSelection

        return PreparedCycleSelection(self, numbers)

    def c_rate(self, **settings):
        """Return a streaming signed C-rate view; use preview(), mean() or write_parquet().

        Automatic reference is the net closing half-cycle of source cycle 2.
        Supply reference_mAh for a manual reference or capacity to reuse summaries.
        Original measurements are preserved; no step classifier runs.
        """
        from .point_c_rate import point_c_rate

        return point_c_rate(self, **settings)

    def steps(self, **settings):
        """Explicitly classify CC/CV/rest steps; use .table() or request .measurements().

        Ordinary Q/E/C-rate calls do not run this operation. Source columns are
        unchanged; step_type appears only in the requested measurement view.
        """
        from .step_classification import classify_steps

        return classify_steps(self, **settings)

    def events(self, **settings):
        """Discover compact pulse opportunities; optionally assess a known profile.

        This explicit operation does not accept events, change cycle counting or
        calculate DCIR. Use ``PulseProfile`` in ``settings`` to check known
        rest-pulse-rest durations without adding fields to measurement rows.
        """
        from .pulse_events import detect_pulse_events

        return detect_pulse_events(self, **settings)

    def dcir(self, delay_s, **settings):
        """Measure DCIR from detected events or exact manual pulse boundaries.

        If neither ``events`` nor ``manual`` is supplied, event discovery is run
        explicitly for this call. One scalar delay produces one row per opportunity,
        including unavailable results and their reasons.
        """
        from .pulse_dcir import measure_pulse_dcir

        return measure_pulse_dcir(self, delay_s, **settings)

    def cumulative(self, **settings):
        """Request streamed original rows plus cumulative signed capacity/energy."""
        from .point_cumulative import cumulative_coordinates

        return cumulative_coordinates(self, **settings)

    def time_window(self, start_s, end_s, **settings):
        """Integrate an exact requested time window using surrounding point support."""
        from .time_windows import integrate_time_window

        return integrate_time_window(self, start_s, end_s, **settings)

    def dqdv_domains(self, **settings):
        """Return readable continuous CC domains eligible for later dQ/dV construction."""
        from .prepared_dqdv import plan_dqdv_domains

        return plan_dqdv_domains(self, **settings)

    def dqdv(self, **settings):
        """Calculate bounded CC-only dQ/dV curves; use .table() and .outcomes()."""
        from .prepared_dqdv import prepared_source_dqdv

        return prepared_source_dqdv(self, **settings)

    def plot(self, *, y="voltage_V", max_points=250000, mode="auto"):
        """Plot one original signal against time with display-only reduction."""
        from .prepared_plotting import plot_prepared_signal

        return plot_prepared_signal(self, y=y, max_points=max_points, mode=mode)

    def plot_progressive(self, *, y="voltage_V", max_points=250000, mode="auto"):
        """Create an idle progressive original-signal plot session."""
        from .prepared_plotting import plot_prepared_signal

        return plot_prepared_signal(self, y=y, progressive=True, max_points=max_points, mode=mode)


class _TimestampIndex:
    """Append increasing original float64 times on disk; search only exceptional repeats.

    Ordinary increasing batches use vector operations and sequential writes.
    A nonconsecutive repeated time uses a bounded binary disk search; there is
    no Python set growing with the source and no per-row generated identifier.
    """

    def __init__(self, stream):
        """Use an owned temporary binary stream that the caller closes after preparation."""
        self.stream = stream
        self.count = 0

    def contains(self, value):
        """Search previously appended sorted times using one eight-byte read per level."""
        lo, hi = 0, self.count
        while lo < hi:
            mid = (lo + hi) // 2
            self.stream.seek(mid * 8)
            measured = np.frombuffer(self.stream.read(8), dtype="<f8")[0]
            if measured == value:
                return True
            if measured < value:
                lo = mid + 1
            else:
                hi = mid
        return False

    def append(self, values):
        """Append one already validated increasing batch; scratch use is eight bytes per retained time."""
        self.stream.seek(self.count * 8)
        self.stream.write(np.asarray(values, dtype="<f8").tobytes())
        self.count += len(values)


def _measurement_times(block, time_column):
    """Read finite numeric seconds without changing the retained source time column."""
    if time_column not in block.schema.names:
        raise SchemaError(f"Missing timestamp column: {time_column}")
    original = block.column(block.schema.get_field_index(time_column))
    if pa.types.is_integer(original.type):
        values = original.to_numpy(zero_copy_only=False)
        if np.any(values > 2**53) or np.any(values < -(2**53)):
            raise SchemaError("Integer timestamps exceed exact float64 seconds precision")
    try:
        times = pc.cast(original, pa.float64()).to_numpy(zero_copy_only=False)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as error:
        raise SchemaError("Timestamp column must contain finite numeric seconds") from error
    if not np.all(np.isfinite(times)):
        raise SchemaError("Timestamp column contains missing or nonfinite time")
    return times


def prepare_source(
    source,
    destination,
    *,
    time_column="time_s",
    read_options=None,
    columns=None,
    memory_budget=None,
    scratch_dir=None,
    on_progress=None,
    cancellation=None,
):
    """Keep first timestamp rows and atomically publish a strictly ordered Parquet source.

    Parameters
    ----------
    source, destination
        CSV/TXT/Parquet input and separate Parquet output; input is never overwritten.
    time_column
        Original column containing elapsed seconds; units are explicit, never guessed.
    read_options
        ReadOptions for delimiter, encoding and decoder limits. Column mappings
        are not applied here: supply the original time_column and retain original names.
    columns
        Optional explicit original-column projection. Time is read for validation
        even if not projected; otherwise every original column is retained.
    memory_budget
        Shared MemoryBudget; defaults to 2 GiB. Reserve a conservative eight-reader-
        buffer allowance before reading and adapt batch bytes to available budget.
        This is managed-buffer accounting, not a hard process RSS guarantee.
    scratch_dir
        Optional directory for an eight-byte-per-retained-time temporary disk index.
    on_progress, cancellation
        Optional callback receiving PreparationProgress, and CancellationToken.
        Callbacks run in the caller context, at most five running updates per second.

    Returns
    -------
    PreparedSource
        Lazy handle and aggregate report. Duplicate rows warn once per source;
        backwards retained time raises TimestampOrderError without publishing output.
        Source metadata is recorded once; no sample/trace/result IDs are generated.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError("Preparation must not overwrite the original source")
    if destination.suffix.lower() != ".parquet":
        raise ValueError("Preparation destination must be a Parquet file")
    options = read_options or ReadOptions()
    if options.batch_rows <= 0 or options.max_batch_bytes <= 0:
        raise ValueError("Positive reader batch limits required")
    initial = fingerprint(source)
    budget = memory_budget or MemoryBudget(2 * 1024**3)
    reader_bytes = min(options.max_batch_bytes, (budget.limit - budget.used) // 8)
    if reader_bytes < 64:
        raise BudgetExceeded("Insufficient global budget for source preparation")
    reservation = reader_bytes * 8
    options = replace(options, max_batch_bytes=reader_bytes, column_mapping={})
    token = cancellation or CancellationToken()
    requested = None if columns is None else tuple(dict.fromkeys(columns))
    if requested == ():
        raise ValueError("Projection must contain at least one column")
    read_columns = None if requested is None else list(dict.fromkeys((*requested, time_column)))
    report = {
        "source_file": source.name,
        "source_metadata": initial,
        "preparation": "keep_first_timestamp_then_require_increasing_time",
        "time_column": time_column,
        "time_unit": "s",
        "rows_read": 0,
        "rows_retained": 0,
        "duplicates_removed": 0,
        "columns": None if requested is None else list(requested),
        "reader_settings": asdict(options),
        "managed_reservation_bytes": reservation,
        "generated_measurement_identifiers": False,
    }
    last_update = -1e100

    def notify(status="running"):
        """Send bounded readable progress and force a terminal publication after success."""
        nonlocal last_update
        now = monotonic()
        if on_progress and (status != "running" or now - last_update >= 0.2):
            last_update = now
            on_progress(
                PreparationProgress(
                    source.name,
                    report["rows_read"],
                    report["rows_retained"],
                    report["duplicates_removed"],
                    status,
                )
            )

    def prepared_batches():
        """Validate batch order and repeat membership before yielding retained whole rows."""
        previous = -np.inf
        warned = False
        seen_batch = False
        reader = source_batches(source, options, columns=read_columns)
        with tempfile.TemporaryFile(dir=scratch_dir) as stream:
            index = _TimestampIndex(stream)
            try:
                for block in reader:
                    token.check()
                    seen_batch = True
                    if requested is not None and set(requested) - set(block.schema.names):
                        raise SchemaError("Projection includes columns absent from the source")
                    times = _measurement_times(block, time_column)
                    if not len(times):
                        continue
                    maximum = np.maximum(np.maximum.accumulate(times), previous)
                    preceding = np.empty_like(maximum)
                    preceding[0] = previous
                    preceding[1:] = maximum[:-1]
                    keep = times > preceding
                    newly_retained = times[keep]
                    for position in np.flatnonzero(times < preceding):
                        value = times[position]
                        candidate = np.searchsorted(newly_retained, value)
                        in_batch = (
                            candidate < len(newly_retained) and newly_retained[candidate] == value
                        )
                        if not in_batch and not index.contains(value):
                            raise TimestampOrderError(
                                f"{source.name}: retained time decreases from "
                                f"{preceding[position]:g} s to {value:g} s"
                            )
                    removed = int(len(times) - np.count_nonzero(keep))
                    report["rows_read"] += len(times)
                    report["rows_retained"] += len(newly_retained)
                    report["duplicates_removed"] += removed
                    if removed and not warned:
                        warnings.warn(
                            f"{source.name}: duplicate timestamps; retaining the first complete "
                            "row and removing every later occurrence before calculations",
                            DuplicateTimestampWarning,
                            stacklevel=2,
                        )
                        warned = True
                    index.append(newly_retained)
                    previous = float(maximum[-1])
                    result = block if not removed else block.filter(pa.array(keep))
                    if requested is not None:
                        result = result.select(requested)
                    notify()
                    yield result
                if not seen_batch:
                    raise SchemaError("Source contains no readable measurement batches")
                token.check()
                if fingerprint(source) != initial:
                    raise SourceChangedError("Source changed during preparation")
            finally:
                reader.close()

    budget.reserve(reservation)
    try:
        atomic_parquet(prepared_batches(), destination, manifest=report)
        notify("complete")
        return PreparedSource(destination, report, options)
    finally:
        budget.release(reservation)


def prepare_sources(sources, output_dir, **kwargs):
    """Prepare files serially under one global budget; return filename-to-lazy-handle dict.

    Duplicate basenames are rejected case-insensitively before reading or writing.
    The dictionary contains metadata/paths, not full source dataframes. Output
    names retain the complete input filename to distinguish CSV from Parquet.
    Supply prepare_source keyword settings in kwargs; worker parallelism is not
    implemented by this entry point.
    """
    paths = [Path(source) for source in sources]
    names = [path.name.casefold() for path in paths]
    if len(names) != len(set(names)):
        raise ValueError("Files with the same basename must not be analyzed together")
    shared = kwargs.pop("memory_budget", None) or MemoryBudget(2 * 1024**3)
    output_dir = Path(output_dir)
    prepared = {
        source.name: prepare_source(
            source,
            output_dir / (source.name + ".prepared.parquet"),
            memory_budget=shared,
            **kwargs,
        )
        for source in paths
    }
    from .multifile import PreparedCollection

    return PreparedCollection(prepared)
