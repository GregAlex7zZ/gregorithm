"""Explicit, bounded CC/CV/rest interpretation without changing measurements or Q/E.

Call data.steps().table() for compact summaries. Only measurements() expands
step_type into requested measurement batches. Manual scopes take priority over
explicit protocol mappings, then inferred signal windows; measured rest remains rest.
"""

import math
import sqlite3
import tempfile
import warnings
from contextlib import closing
from dataclasses import asdict, dataclass, field, replace
from numbers import Integral
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ._result_tables import bounded_frame
from .diagnostics import BudgetExceeded, SourceChangedError
from .execution import CancellationToken, MemoryBudget
from .io import atomic_parquet, fingerprint
from .point_calculations import EstimatedMeasurementWarning, _effective_batches, _numeric

TYPES = ("CC", "CV", "rest", "variable", "unknown")
_ORIGINS = ("inferred", "protocol", "manual")
_DIRECTIONS = {1: "charge", -1: "discharge", 0: "rest", 3: "unknown"}


class AnnotationConflictWarning(UserWarning):
    """Requested annotations conflict with measured physical rest/active state."""


class UnmappedProtocolWarning(UserWarning):
    """Unmapped explicit protocol values fall back to signal inference, never guessed codes."""


@dataclass(frozen=True)
class StepConfig:
    """User-approved operational inference defaults; not instrument/protocol calibration.

    Current/voltage ranges use real observations. Windows target elapsed seconds,
    extend for sparse real support, and never establish CC from estimated samples.
    """

    rest_threshold_mA: float = 0.0001
    window_s: float = 3.0
    min_real_samples: int = 2
    min_real_span_s: float = 0.1
    current_relative: float = 0.05
    current_absolute_mA: float = 0.0001
    voltage_range_V: float = 0.002
    level_change_relative: float = 0.05
    missing_span_s: float = 10.0

    def __post_init__(self):
        """Validate settings before reading a file or creating temporary storage."""
        for name, value in asdict(self).items():
            if name == "min_real_samples":
                if (
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, Integral)
                    or value < 2
                ):
                    raise ValueError("min_real_samples must be an integer >= 2")
                object.__setattr__(self, name, int(value))
            elif isinstance(value, (bool, np.bool_)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            else:
                object.__setattr__(self, name, float(value))
        if self.window_s <= 0:
            raise ValueError("window_s must be positive")


_SCHEMA = pa.schema(
    [
        ("step", pa.int64()),
        ("step_type", pa.string()),
        ("direction", pa.string()),
        ("start_s", pa.float64()),
        ("end_s", pa.float64()),
        ("mean_current_mA", pa.float64()),
        ("current_quality", pa.string()),
        ("classification_origin", pa.string()),
        ("inferred_type", pa.string()),
        ("n_samples", pa.int64()),
        ("n_real_current", pa.int64()),
        ("n_estimated_current", pa.int64()),
        ("n_estimated_voltage", pa.int64()),
        ("assessment_windows", pa.int64()),
        ("window_span_min_s", pa.float64()),
        ("window_span_max_s", pa.float64()),
        ("reference_current_mA", pa.float64()),
        ("reference_start_s", pa.float64()),
        ("reference_end_s", pa.float64()),
        ("reference_real_samples", pa.int64()),
        ("current_min_mA", pa.float64()),
        ("current_max_mA", pa.float64()),
        ("preceding_sample_s", pa.float64()),
        ("following_sample_s", pa.float64()),
        ("covered_s", pa.float64()),
        ("classification_support", pa.string()),
    ]
)
_BASIC = (
    "step",
    "step_type",
    "direction",
    "start_s",
    "end_s",
    "mean_current_mA",
    "current_quality",
    "covered_s",
)


def _manual_ranges(windows):
    """Validate exact-time manual scopes and reject contradictory inclusive overlaps."""
    merged, endpoints = [], []
    for lo, hi, label in sorted(windows):
        if label not in TYPES or not math.isfinite(lo) or not math.isfinite(hi) or hi < lo:
            raise ValueError("Manual scopes require finite ordered times and a known step_type")
        lo, hi = float(lo), float(hi)
        endpoints.extend((lo, hi))
        if merged and lo <= merged[-1][1]:
            if label != merged[-1][2]:
                raise ValueError("Contradictory overlapping manual step annotations")
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]), label)
        else:
            merged.append((lo, hi, label))
    return merged, np.unique(endpoints)


def _annotations(block, times, column, mapping, manual):
    """Vectorize explicit protocol mappings and time scopes; never guess mode-code meanings."""
    codes, origins = np.full(len(times), -1, dtype=np.int8), np.zeros(len(times), dtype=np.int8)
    unmapped = 0
    if column is not None:
        encoded = pc.dictionary_encode(block.column(column))
        labels = np.array(
            [
                TYPES.index(mapping[value]) if value in mapping else -1
                for value in encoded.dictionary.to_pylist()
            ],
            dtype=np.int8,
        )
        indices = encoded.indices.to_numpy(zero_copy_only=False)
        valid = np.isfinite(indices)
        codes[valid] = labels[indices[valid].astype(int)]
        origins[codes >= 0] = 1
        unmapped = int(np.count_nonzero(codes < 0))
    if manual:
        starts = np.array([item[0] for item in manual])
        ends = np.array([item[1] for item in manual])
        labels = np.array([TYPES.index(item[2]) for item in manual], dtype=np.int8)
        locations = np.searchsorted(starts, times, side="right") - 1
        valid = (locations >= 0) & (times <= ends[np.maximum(locations, 0)])
        codes[valid], origins[valid] = labels[locations[valid]], 2
    return codes, origins, unmapped


class _Window:
    """Constant-size measured extrema, real support and compensated integral for one window."""

    def __init__(self):
        """Allocate scalar state only; window duration never implies a RAM row-count limit."""
        self.first = self.last = self.first_real = self.last_real = None
        self.previous = None
        self.n = self.real = self.ei = self.ev = self.vreal = 0
        self.imin, self.imax, self.vmin, self.vmax = math.inf, -math.inf, math.inf, -math.inf
        self.vfirst = self.vlast = None
        self.vmissing = False
        self.integral = self.correction = 0.0

    def add(self, t, raw_i, effective_i, raw_v, effective_v, ei, ev):
        """Reduce bounded slices; estimates neither supply extrema nor real confirmation."""
        if not len(t):
            return
        if self.first is None:
            self.first = float(t[0])
        self.last = float(t[-1])
        self.n += len(t)
        self.ei += int(np.count_nonzero(ei))
        self.ev += int(np.count_nonzero(ev))
        finite = np.flatnonzero(np.isfinite(raw_i))
        if len(finite):
            if self.first_real is None:
                self.first_real = float(t[finite[0]])
            self.last_real = float(t[finite[-1]])
            self.real += len(finite)
            self.imin = min(self.imin, float(np.min(raw_i[finite])))
            self.imax = max(self.imax, float(np.max(raw_i[finite])))
        if raw_v is not None:
            finite_v = np.flatnonzero(np.isfinite(raw_v))
            if len(finite_v):
                if self.vfirst is None:
                    self.vfirst = float(t[finite_v[0]])
                self.vlast = float(t[finite_v[-1]])
                self.vreal += len(finite_v)
                self.vmin = min(self.vmin, float(np.min(raw_v[finite_v])))
                self.vmax = max(self.vmax, float(np.max(raw_v[finite_v])))
        self.vmissing |= effective_v is None or bool(np.any(~np.isfinite(effective_v)))
        # Freeze the plateau mean using only real endpoint support inside this window.
        # Eligible interior estimates lie on the same line; estimated endpoints must
        # not pull the reference toward a neighbouring unconfirmed current plateau.
        points_t, points_i = t[finite], raw_i[finite]
        if not len(points_t):
            return
        if self.previous is not None:
            points_t = np.r_[self.previous[0], points_t]
            points_i = np.r_[self.previous[1], points_i]
        terms = (points_i[:-1] + points_i[1:]) * np.diff(points_t) / 2
        increment = float(np.sum(terms[np.isfinite(terms)])) - self.correction
        combined = self.integral + increment
        self.correction = (combined - self.integral) - increment
        self.integral = combined
        self.previous = float(points_t[-1]), float(points_i[-1])

    def infer(self, direction, config):
        """Apply rest priority, real-support eligibility, current range, then voltage range."""
        if direction == 0:
            return "rest", "measured_rest", None
        if direction == 3:
            return "unknown", "unavailable_current", None
        if self.real < config.min_real_samples or self.last_real < np.nextafter(
            self.first_real + config.min_real_span_s, -np.inf
        ):
            return "unknown", "insufficient_measured_support", None
        allowed = max(
            config.current_absolute_mA,
            config.current_relative * max(abs(self.imin), abs(self.imax)),
        )
        if self.imax - self.imin <= np.nextafter(allowed, np.inf):
            return (
                "CC",
                "estimated" if self.ei else "measured",
                self.integral / (self.last_real - self.first_real),
            )
        if (
            self.vmissing
            or self.vreal < config.min_real_samples
            or self.vlast < np.nextafter(self.vfirst + config.min_real_span_s, -np.inf)
        ):
            return "unknown", "unavailable_voltage_support", None
        return (
            (
                "CV"
                if self.vmax - self.vmin <= np.nextafter(config.voltage_range_V, np.inf)
                else "variable"
            ),
            "estimated" if self.ei or self.ev else "measured",
            None,
        )


class _Scanner:
    """Merge compatible windows into compact physical steps; freeze each CC level reference."""

    def __init__(self, database, config):
        """Keep one open window and step, never per-source measurement arrays."""
        self.db, self.config = database, config
        self.key = self.window = self.step = None
        self.previous_time = None
        self.window_preceding = None
        self.number = self.windows = self.contradictions = 0

    def emit(self, assessed):
        """Classify one completed domain and merge only compatible adjacent step support."""
        if self.window is None:
            return
        window, (direction, code, origin) = self.window, self.key
        inferred, support, level = window.infer(direction, self.config)
        label = inferred if code < 0 else TYPES[code]
        if direction == 0 and label != "rest" or direction != 0 and label == "rest":
            self.contradictions += window.n
            label, origin = inferred, 0
        origin = _ORIGINS[origin]
        incompatible = (
            self.step is None
            or self.step["step_type"] != label
            or self.step["direction"] != _DIRECTIONS[direction]
        )
        if (
            not incompatible
            and label == "CC"
            and level is not None
            and self.step["reference_current_mA"] is not None
        ):
            reference = self.step["reference_current_mA"]
            threshold = self.config.level_change_relative * max(abs(level), abs(reference))
            incompatible = abs(level - reference) > np.nextafter(threshold, np.inf)
        if incompatible:
            if self.step is not None:
                self.step["following_sample_s"] = window.first
                self.save()
            self.number += 1
            self.step = dict(
                step=self.number,
                step_type=label,
                direction=_DIRECTIONS[direction],
                start_s=window.first,
                end_s=window.last,
                mean_current_mA=None,
                current_quality="unavailable",
                classification_origin=origin,
                inferred_type=inferred,
                n_samples=0,
                n_real_current=0,
                n_estimated_current=0,
                n_estimated_voltage=0,
                assessment_windows=0,
                window_span_min_s=None,
                window_span_max_s=None,
                reference_current_mA=level if label == "CC" else None,
                reference_start_s=window.first_real
                if label == "CC" and level is not None
                else None,
                reference_end_s=window.last_real if label == "CC" and level is not None else None,
                reference_real_samples=window.real if label == "CC" and level is not None else None,
                current_min_mA=None,
                current_max_mA=None,
                preceding_sample_s=self.window_preceding,
                following_sample_s=None,
                covered_s=0.0,
                classification_support=support,
            )
        step = self.step
        if label == "CC" and step["reference_current_mA"] is None and level is not None:
            step.update(
                reference_current_mA=level,
                reference_start_s=window.first_real,
                reference_end_s=window.last_real,
                reference_real_samples=window.real,
            )
        step["end_s"] = window.last
        for name, value in (
            ("n_samples", window.n),
            ("n_real_current", window.real),
            ("n_estimated_current", window.ei),
            ("n_estimated_voltage", window.ev),
            ("assessment_windows", int(assessed)),
        ):
            step[name] += value
        for name, value in (
            ("classification_origin", origin),
            ("inferred_type", inferred),
            ("classification_support", support),
        ):
            if step[name] != value:
                step[name] = "mixed"
        if window.real:
            step["current_min_mA"] = (
                window.imin
                if step["current_min_mA"] is None
                else min(step["current_min_mA"], window.imin)
            )
            step["current_max_mA"] = (
                window.imax
                if step["current_max_mA"] is None
                else max(step["current_max_mA"], window.imax)
            )
        if assessed:
            span = window.last - window.first
            step["window_span_min_s"] = (
                span if step["window_span_min_s"] is None else min(step["window_span_min_s"], span)
            )
            step["window_span_max_s"] = (
                span if step["window_span_max_s"] is None else max(step["window_span_max_s"], span)
            )
            self.windows += 1
        self.window = None

    def save(self):
        """Persist one compact step in original order, using its readable local step number."""
        self.db.execute(
            "INSERT INTO steps VALUES (" + ",".join("?" for _ in _SCHEMA.names) + ")",
            [self.step[name] for name in _SCHEMA.names],
        )

    def consume(
        self, t, raw_i, effective_i, raw_v, effective_v, ei, ev, continuity, codes, origins
    ):
        """Process uniform-direction/annotation runs and consecutive nonoverlapping time windows."""
        directions = np.where(
            np.isfinite(raw_i),
            np.where(
                raw_i > self.config.rest_threshold_mA,
                1,
                np.where(raw_i < -self.config.rest_threshold_mA, -1, 0),
            ),
            np.where(continuity, np.sign(effective_i), 3),
        ).astype(np.int8)
        edges = np.r_[
            0,
            np.flatnonzero(
                (directions[1:] != directions[:-1])
                | (codes[1:] != codes[:-1])
                | (origins[1:] != origins[:-1])
            )
            + 1,
            len(t),
        ]
        for lo, hi in zip(edges[:-1], edges[1:]):
            key = int(directions[lo]), int(codes[lo]), int(origins[lo])
            if key != self.key:
                self.emit(self.key is not None and self.key[0] in {-1, 1})
                self.key = key
            offset = int(lo)
            while offset < hi:
                if self.window is None:
                    if key[0] in {-1, 1} and not np.isfinite(raw_i[offset]):
                        # An estimate cannot anchor a measured confirmation window.
                        original_key = self.key
                        self.key = (3, key[1], key[2])
                        self.window = _Window()
                        self.window_preceding = self.previous_time
                        end = offset + 1
                        self.window.add(
                            t[offset:end],
                            raw_i[offset:end],
                            effective_i[offset:end],
                            None if raw_v is None else raw_v[offset:end],
                            None if effective_v is None else effective_v[offset:end],
                            ei[offset:end],
                            ev[offset:end],
                        )
                        self.previous_time = float(t[offset])
                        self.emit(False)
                        self.key = original_key
                        offset = end
                        continue
                    self.window = _Window()
                    self.window_preceding = self.previous_time
                if key[0] not in {-1, 1}:
                    end = int(hi)
                else:
                    anchor = (
                        self.window.first if self.window.first is not None else float(t[offset])
                    )
                    end = min(
                        int(hi),
                        int(
                            np.searchsorted(
                                t, np.nextafter(anchor + self.config.window_s, np.inf), side="right"
                            )
                        ),
                    )
                    if end <= offset:
                        if self.window.real < self.config.min_real_samples:
                            end = offset + 1
                        else:
                            self.emit(True)
                            continue
                self.window.add(
                    t[offset:end],
                    raw_i[offset:end],
                    effective_i[offset:end],
                    None if raw_v is None else raw_v[offset:end],
                    None if effective_v is None else effective_v[offset:end],
                    ei[offset:end],
                    ev[offset:end],
                )
                self.previous_time = float(t[end - 1])
                offset = end
                if (
                    key[0] in {-1, 1}
                    and self.window.real >= self.config.min_real_samples
                    and self.window.last >= self.window.first + self.config.window_s
                ):
                    self.emit(True)

    def finish(self):
        """Assess the shorter final window, then persist the last observed step without extrapolation."""
        self.emit(self.key is not None and self.key[0] in {-1, 1})
        if self.step is not None:
            self.save()


@dataclass(frozen=True)
class StepResult:
    """Disk-backed optional step summaries; measurements() alone expands categorical labels."""

    source: object = field(repr=False)
    path: Path = field(repr=False)
    report: dict
    _workspace: object = field(repr=False)
    _budget: object = field(repr=False)
    _details: bool = False
    _types: tuple | None = None

    def details(self):
        """Expose measured support, effective windows, annotation origin and fixed CC reference."""
        return replace(self, _details=True)

    def types(self, labels):
        """Select readable step types without rerunning inference or joining separated regions."""
        values = (labels,) if isinstance(labels, str) else tuple(labels)
        if any(value not in TYPES for value in values):
            raise ValueError("Unknown step type")
        if self._types is not None:
            values = tuple(value for value in values if value in self._types)
        return replace(self, _types=values)

    def iter_batches(self, batch_rows=512):
        """Read bounded compact summaries in original step order, including typed empty results."""
        if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows < 1:
            raise ValueError("Positive batch_rows required")
        rows = min(batch_rows, (self._budget.limit - self._budget.used - 2 * 1024**2) // 4096)
        if rows < 1:
            raise BudgetExceeded("Insufficient step-summary reader budget")
        reservation = 2 * 1024**2 + rows * 4096
        self._budget.reserve(reservation)
        self._workspace.active_readers = getattr(self._workspace, "active_readers", 0) + 1
        try:
            if not self.path.is_file():
                raise ValueError("Step result is closed")
            names = _SCHEMA.names if self._details else _BASIC
            schema = pa.schema([_SCHEMA.field(name) for name in names])
            with closing(sqlite3.connect(self.path)) as db:
                db.execute("PRAGMA cache_size=-2048")
                where = (
                    ""
                    if self._types is None
                    else " WHERE step_type IN (" + ",".join("?" for _ in self._types) + ")"
                )
                cursor = db.execute(
                    "SELECT " + ",".join(names) + " FROM steps" + where + " ORDER BY step",
                    () if self._types is None else self._types,
                )
                emitted = False
                while values := cursor.fetchmany(rows):
                    emitted = True
                    yield pa.RecordBatch.from_pydict(
                        {name: [row[k] for row in values] for k, name in enumerate(names)},
                        schema=schema,
                    )
                if not emitted:
                    yield pa.RecordBatch.from_pylist([], schema=schema)
        finally:
            self._workspace.active_readers -= 1
            self._budget.release(reservation)

    def table(self, max_rows=100000):
        """Return a guarded pandas summary; use iter_batches() for very many steps."""
        return bounded_frame(self, max_rows)

    def measurements(self, *, include_step_number=False):
        """Explicitly request original measurements plus step_type (and optional readable step)."""
        if not isinstance(include_step_number, (bool, np.bool_)):
            raise ValueError("include_step_number must be boolean")
        return StepMeasurements(self, bool(include_step_number))

    def c_rate(self, **settings):
        """Scale within-step means by the fixed source reference; e.g. types('CC').c_rate().

        Pass capacity=existing_capacity to avoid a repeated capacity calculation,
        or reference_mAh for a manual absolute reference. Switching fronts across
        steps are excluded from these means. No new classifier scan is run.
        """
        from .step_rates import StepRates

        if fingerprint(self.source.path) != self.report["prepared_source_metadata"]:
            raise SourceChangedError("Prepared source changed since step interpretation")
        if settings.get("memory_budget") is None:
            settings["memory_budget"] = self._budget
        return StepRates(self, self.source.c_rate(**settings).reference)

    def quantities(self, **settings):
        """Calculate capacity and optional energy for these existing step boundaries."""
        from .step_quantities import integrate_steps

        return integrate_steps(self, **settings)

    def write_parquet(self, path):
        """Stream selected compact summaries with settings/annotations in one metadata sidecar."""
        if Path(path).exists():
            raise FileExistsError("Choose a new step-summary export path")
        return atomic_parquet(
            self.iter_batches(), path, manifest={**self.report, "selected_step_types": self._types}
        )

    def close(self):
        """Close shared views and verified owned storage; active iterators must close first."""
        if getattr(self._workspace, "active_readers", 0):
            raise ValueError("Close active step iterators first")
        if Path(self._workspace.name).resolve() != self.path.parent:
            raise ValueError("Step cleanup target changed")
        self._workspace.cleanup()


@dataclass(frozen=True)
class StepMeasurements:
    """Requested categorical interpretation alongside unchanged originals, streamed on demand."""

    result: StepResult = field(repr=False)
    include_step_number: bool = False

    @property
    def _budget(self):
        """Share the originating result's global managed budget."""
        return self.result._budget

    def iter_batches(self, columns=None, cancellation=None):
        """Expand compact boundaries into bounded batches, never a source-wide label array."""
        token = cancellation or CancellationToken()
        token.check()
        if not self.result.path.is_file():
            raise ValueError("Step result is closed")
        original_source = self.result.source
        if fingerprint(original_source.path) != self.result.report["prepared_source_metadata"]:
            raise SourceChangedError("Prepared source changed since step interpretation")
        with pq.ParquetFile(original_source.path) as parquet:
            original_schema = parquet.schema_arrow
        requested = original_schema.names if columns is None else list(columns)
        if len(set(requested)) != len(requested) or any(
            name not in original_schema.names for name in requested
        ):
            raise ValueError("Projection must contain distinct existing columns")
        budget = self._budget
        cache = 2 * 1024**2
        reader_bytes = min(
            self.result.source.read_options.max_batch_bytes,
            (budget.limit - budget.used - cache) // 16,
        )
        if reader_bytes < 64:
            raise BudgetExceeded("Insufficient step measurement budget")
        reservation = reader_bytes * 16 + cache
        budget.reserve(reservation)
        self.result._workspace.active_readers = (
            getattr(self.result._workspace, "active_readers", 0) + 1
        )
        source = replace(
            self.result.source,
            read_options=replace(self.result.source.read_options, max_batch_bytes=reader_bytes),
        )
        read_columns = list(dict.fromkeys([*requested, source.report["time_column"]]))
        reader = source.iter_batches(columns=read_columns)
        try:
            if fingerprint(source.path) != self.result.report["prepared_source_metadata"]:
                raise SourceChangedError("Prepared source changed since step interpretation")
            with closing(sqlite3.connect(self.result.path)) as db:
                db.execute("PRAGMA cache_size=-2048")
                cursor = db.execute("SELECT step,step_type,start_s,end_s FROM steps ORDER BY step")
                step = cursor.fetchone()
                for block in reader:
                    token.check()
                    if (
                        "step_type" in block.schema.names
                        or self.include_step_number
                        and "step" in block.schema.names
                    ):
                        raise ValueError("Step columns would overwrite existing measurements")
                    times = _numeric(block, source.report["time_column"])
                    labels = np.empty(len(times), dtype=np.int8)
                    numbers = (
                        np.empty(len(times), dtype=np.int64) if self.include_step_number else None
                    )
                    offset = 0
                    while offset < len(times):
                        while step is not None and step[3] < times[offset]:
                            step = cursor.fetchone()
                        if step is None or times[offset] < step[2]:
                            raise ValueError("Step boundaries do not cover original measurements")
                        end = int(np.searchsorted(times, step[3], side="right"))
                        labels[offset:end] = TYPES.index(step[1])
                        if numbers is not None:
                            numbers[offset:end] = step[0]
                        offset = end
                    selected = (
                        np.ones(len(times), dtype=bool)
                        if self.result._types is None
                        else np.isin(labels, [TYPES.index(value) for value in self.result._types])
                    )
                    output = block.select(requested).append_column(
                        "step_type",
                        pa.DictionaryArray.from_arrays(pa.array(labels), pa.array(TYPES)),
                    )
                    if self.include_step_number:
                        output = output.append_column("step", pa.array(numbers))
                    yield output.filter(pa.array(selected))
            token.check()
            if fingerprint(source.path) != self.result.report["prepared_source_metadata"]:
                raise SourceChangedError("Prepared source changed during step measurement reading")
        finally:
            reader.close()
            self.result._workspace.active_readers -= 1
            budget.release(reservation)

    def preview(self, rows=5):
        """Show at most rows explicitly requested labels; no source changes occur."""
        if rows > 100000:
            raise ValueError("Preview limited to 100000 rows")
        return bounded_frame(self, rows, preview=True)

    def table(self, max_rows=100000):
        """Materialize an explicitly guarded measurement view; stream large requests instead."""
        return bounded_frame(self, max_rows)

    def write_parquet(self, path, columns=None, cancellation=None):
        """Export originals plus requested categorical labels, preserving existing files."""
        if Path(path).exists():
            raise FileExistsError("Choose a new measurement export path")
        return atomic_parquet(
            self.iter_batches(columns=columns, cancellation=cancellation),
            path,
            manifest={
                **self.result.report,
                "selected_step_types": self.result._types,
                "source_projection": columns,
                "appended_measurement_columns": ["step_type"]
                + (["step"] if self.include_step_number else []),
            },
        )


def _update_means(db, source, config, columns, token):
    """Integrate only consecutive endpoints within the same step, excluding switching fronts."""
    cursor = db.execute("SELECT step,start_s,end_s FROM steps ORDER BY step")
    current = cursor.fetchone()
    integral = duration = correction = 0.0
    missing = estimated = False
    previous = None
    reader = _effective_batches(source, columns[:2], config.missing_span_s, token)

    def save():
        """Persist one mean with covered support and shared quality, never unknown-as-zero."""
        quality = (
            "unavailable"
            if duration <= 0
            else "partial"
            if missing
            else "estimated"
            if estimated
            else "complete"
        )
        db.execute(
            "UPDATE steps SET mean_current_mA=?,current_quality=?,covered_s=? WHERE step=?",
            (None if duration <= 0 else integral / duration, quality, duration, current[0]),
        )

    try:
        for t, _, i, _, ei, _ in reader:
            if previous is not None:
                t, i, ei = np.r_[previous[0], t], np.r_[previous[1], i], np.r_[previous[2], ei]
            while current is not None and len(t):
                lo = int(np.searchsorted(t, current[1], side="left"))
                hi = int(np.searchsorted(t, current[2], side="right"))
                if hi - lo >= 2:
                    dt = np.diff(t[lo:hi])
                    terms = (i[lo : hi - 1] + i[lo + 1 : hi]) * dt / 2
                    valid = np.isfinite(terms)
                    missing |= bool(np.any(~valid))
                    estimated |= bool(np.any((ei[lo : hi - 1] | ei[lo + 1 : hi])[valid]))
                    duration += float(np.sum(dt[valid]))
                    value = float(np.sum(terms[valid])) - correction
                    combined = integral + value
                    correction = (combined - integral) - value
                    integral = combined
                if t[-1] <= current[2]:
                    break
                save()
                current = cursor.fetchone()
                integral = duration = correction = 0.0
                missing = estimated = False
            previous = t[-1], i[-1], bool(ei[-1])
        if current is not None:
            save()
    finally:
        reader.close()


def classify_steps(
    source,
    *,
    config=None,
    current_column="current_mA",
    voltage_column="voltage_V",
    protocol_column=None,
    protocol_mapping=None,
    manual_windows=(),
    memory_budget=None,
    cancellation=None,
    scratch_dir=None,
):
    """Explicitly infer compact CC/CV/rest steps and apply user-specified annotation priority.

    data.steps().table() does not add measurement columns. Request
    result.measurements() to expand step_type. Manual scopes are inclusive exact
    retained timestamps: [(start_s, end_s, 'CV')]. Protocol codes require an explicit
    mapping; no arbitrary source column is trusted. Q/E and original values are unchanged.
    """
    cfg = config or StepConfig()
    if not isinstance(cfg, StepConfig):
        raise TypeError("config must be StepConfig")
    mapping = {
        key.item() if isinstance(key, np.generic) else key: value
        for key, value in dict(protocol_mapping or {}).items()
    }
    if any(
        not isinstance(key, (str, bool, int, float))
        or isinstance(key, float)
        and not math.isfinite(key)
        for key in mapping
    ):
        raise ValueError("Protocol codes must be finite numeric, boolean or text values")
    if (protocol_column is None) != (not mapping) or any(
        value not in TYPES for value in mapping.values()
    ):
        raise ValueError("Provide protocol_column and a nonempty valid protocol_mapping together")
    manual, endpoints = _manual_ranges(manual_windows)
    budget, token = memory_budget or MemoryBudget(2 * 1024**3), cancellation or CancellationToken()
    token.check()
    initial = fingerprint(source.path)
    with pq.ParquetFile(source.path) as parquet:
        names = parquet.schema_arrow.names
    columns = [source.report["time_column"], current_column]
    if voltage_column in names:
        columns.append(voltage_column)
    if protocol_column is not None and protocol_column not in names:
        raise ValueError("Missing mapped protocol column")
    cache = 2 * 1024**2
    reader_bytes = min(
        source.read_options.max_batch_bytes, (budget.limit - budget.used - cache) // 16
    )
    if reader_bytes < 64:
        raise BudgetExceeded("Insufficient shared step-classification budget")
    reservation = reader_bytes * 16 + cache
    budget.reserve(reservation)
    work = None
    try:
        parent = Path(scratch_dir or source.path.parent / ".gregorithm-work").resolve()
        parent.mkdir(parents=True, exist_ok=True)
        work = tempfile.TemporaryDirectory(prefix="steps-", dir=parent)
        folder = Path(work.name).resolve()
        if not folder.is_relative_to(parent):
            raise ValueError("Step workspace escaped its intended directory")
        path = folder / "steps.sqlite"
        view = replace(
            source, read_options=replace(source.read_options, max_batch_bytes=reader_bytes)
        )
        with closing(sqlite3.connect(path)) as db:
            db.execute("PRAGMA cache_size=-2048")
            db.execute("PRAGMA temp_store=FILE")
            db.execute(
                "CREATE TABLE steps ("
                + ",".join(
                    name
                    + (
                        " TEXT"
                        if pa.types.is_string(dtype)
                        else " INTEGER"
                        if pa.types.is_integer(dtype)
                        else " REAL"
                    )
                    + (" PRIMARY KEY" if name == "step" else "")
                    for name, dtype in [(field.name, field.type) for field in _SCHEMA]
                )
                + ")"
            )
            scanner = _Scanner(db, cfg)
            found, unmapped = np.zeros(len(endpoints), dtype=bool), 0
            reader = _effective_batches(
                view,
                columns,
                cfg.missing_span_s,
                token,
                extra_columns=() if protocol_column is None else (protocol_column,),
                include_original=True,
                rest_threshold_mA=cfg.rest_threshold_mA,
            )
            try:
                for t, raw_i, effective_i, effective_v, ei, ev, original, continuity in reader:
                    if len(endpoints):
                        positions = np.searchsorted(t, endpoints)
                        found |= (positions < len(t)) & (
                            t[np.minimum(positions, len(t) - 1)] == endpoints
                        )
                    codes, origins, count = _annotations(
                        original, t, protocol_column, mapping, manual
                    )
                    unmapped += count
                    raw_v = _numeric(original, voltage_column) if len(columns) == 3 else None
                    scanner.consume(
                        t,
                        raw_i,
                        effective_i,
                        raw_v,
                        effective_v,
                        ei,
                        ev,
                        continuity,
                        codes,
                        origins,
                    )
            finally:
                reader.close()
            if not np.all(found):
                raise ValueError("Manual annotation endpoints must be retained measured timestamps")
            scanner.finish()
            db.commit()
            _update_means(db, view, cfg, columns, token)
            db.commit()
            estimated_count = db.execute(
                "SELECT COALESCE(SUM(n_estimated_current),0) FROM steps"
            ).fetchone()[0]
            if fingerprint(source.path) != initial:
                raise SourceChangedError("Prepared source changed during step interpretation")
            token.check()
            report = dict(
                source_file=source.report["source_file"],
                prepared_source_metadata=initial,
                operation="optional_step_classification",
                settings=asdict(cfg),
                measurement_columns=dict(
                    time=columns[0],
                    current=current_column,
                    voltage=voltage_column if len(columns) == 3 else None,
                ),
                protocol_column=protocol_column,
                protocol_mapping=[
                    dict(value=key, step_type=value) for key, value in mapping.items()
                ],
                manual_windows=manual,
                steps=scanner.number,
                assessment_windows=scanner.windows,
                annotation_contradictions=scanner.contradictions,
                unmapped_protocol_rows=unmapped,
                estimated_current_points=estimated_count,
                generated_measurement_identifiers=False,
                appended_measurement_columns=False,
            )
        if scanner.contradictions:
            warnings.warn(
                "Step annotations conflict with measured rest/active support; physical rules retained",
                AnnotationConflictWarning,
                stacklevel=2,
            )
        if unmapped:
            warnings.warn(
                "Unmapped protocol values use signal inference; inspect the explicit mapping",
                UnmappedProtocolWarning,
                stacklevel=2,
            )
        if estimated_count:
            warnings.warn(
                "Step interpretation uses isolated-current estimates; real observations alone confirm stability",
                EstimatedMeasurementWarning,
                stacklevel=2,
            )
        return StepResult(source, path, report, work, budget)
    except BaseException:
        if work is not None:
            work.cleanup()
        raise
    finally:
        budget.release(reservation)
