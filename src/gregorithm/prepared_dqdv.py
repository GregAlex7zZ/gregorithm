"""CC-only support planning for the identity-free prepared-source dQ/dV path."""

from __future__ import annotations

import math
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa

from ._result_tables import bounded_frame
from .config import DQDVConfig
from .diagnostics import BudgetExceeded
from .dqdv import GridNeighbors
from .execution import CancellationToken, MemoryBudget
from .io import atomic_parquet
from .point_calculations import _effective_batches, directional_integrals


@dataclass(frozen=True)
class DQDVDomain:
    """One readable continuous CC domain within one confirmed half-cycle.

    The domain is a support-selection description. It does not own or copy source
    measurements and does not imply that a derivative curve can be constructed;
    voltage and capacity support are checked in the subsequent numerical stage.
    """

    cycle: int
    half_cycle: int
    step: int
    direction: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class DQDVSupport:
    """Disk-backed eligible samples and compact contiguous-portion summaries."""

    source: object
    path: Path
    domains: tuple[DQDVDomain, ...]
    report: dict
    _workspace: object

    def portions(self):
        """Return readable metadata for every supported contiguous portion."""
        with closing(sqlite3.connect(self.path)) as database:
            rows = database.execute(
                "SELECT cycle,half_cycle,step,portion,direction,start_s,end_s,n_samples,"
                "estimated_current,estimated_voltage,partial_context "
                "FROM portions ORDER BY domain,portion"
            ).fetchall()
        names = (
            "cycle",
            "half_cycle",
            "step",
            "portion",
            "direction",
            "start_s",
            "end_s",
            "n_samples",
            "estimated_current",
            "estimated_voltage",
            "partial_context",
        )
        return tuple(dict(zip(names, row, strict=True)) for row in rows)

    def close(self):
        """Remove owned support storage after downstream curve construction."""
        if Path(self._workspace.name).resolve() != self.path.parent:
            raise ValueError("dQ/dV support cleanup target changed")
        self._workspace.cleanup()


_CURVE_SCHEMA = pa.schema(
    [
        ("cycle", pa.int64()),
        ("half_cycle", pa.int64()),
        ("step", pa.int64()),
        ("portion", pa.int64()),
        ("direction", pa.string()),
        ("voltage_V", pa.float64()),
        ("capacity_mAh", pa.float64()),
        ("capacity_unsmoothed_mAh", pa.float64()),
        ("dqdv_mAh_V", pa.float64()),
        ("dqdv_unsmoothed_mAh_V", pa.float64()),
    ]
)


@dataclass(frozen=True)
class PreparedDQDVResult:
    """Disk-backed readable dQ/dV curves and one outcome per eligible portion."""

    source: object
    path: Path
    report: dict
    _workspace: object
    _budget: object = field(default_factory=lambda: MemoryBudget(2 * 1024**3), repr=False)

    def iter_batches(self, batch_rows=4096):
        """Yield curve points in readable cycle/half-cycle/step/portion order."""
        if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows < 1:
            raise ValueError("Positive batch_rows required")
        with closing(sqlite3.connect(self.path)) as database:
            cursor = database.execute(
                "SELECT " + ",".join(_CURVE_SCHEMA.names) + " FROM curve_points "
                "ORDER BY cycle,half_cycle,step,portion,voltage_V"
            )
            emitted = False
            while rows := cursor.fetchmany(batch_rows):
                emitted = True
                yield pa.RecordBatch.from_pylist(
                    [dict(zip(_CURVE_SCHEMA.names, row, strict=True)) for row in rows],
                    schema=_CURVE_SCHEMA,
                )
            if not emitted:
                yield pa.RecordBatch.from_pylist([], schema=_CURVE_SCHEMA)

    def table(self, max_rows=100000):
        """Return a guarded pandas table of numerical curve points."""
        return bounded_frame(self, max_rows)

    def outcomes(self):
        """Return one compact validity/settings row for every eligible portion."""
        with closing(sqlite3.connect(self.path)) as database:
            cursor = database.execute(
                "SELECT * FROM outcomes ORDER BY cycle,half_cycle,step,portion"
            )
            names = [item[0] for item in cursor.description]
            rows = cursor.fetchall()
        return pa.Table.from_pylist(
            [dict(zip(names, row, strict=True)) for row in rows]
        ).to_pandas()

    def magnitude_table(self, max_rows=100000):
        """Return an explicit display table with labelled absolute dQ/dV."""
        frame = self.table(max_rows=max_rows)
        frame["abs_dqdv_mAh_V"] = frame["dqdv_mAh_V"].abs()
        return frame

    def plot_views(self, *, magnitude=False):
        """Return lazy readable series consumed by final or progressive plotting."""
        from .prepared_plotting import dqdv_plot_views

        return dqdv_plot_views(self, magnitude=magnitude)

    def plot(self, *, magnitude=False, max_points=250000, mode="auto"):
        """Create a final interactive Plotly dQ/dV figure with a whole-figure limit."""
        from .plot_spec import PlotSpec
        from .plotting import plot

        source = _DQDVPlotRequest(self, magnitude)
        y = "abs_dqdv_mAh_V" if magnitude else "dqdv_mAh_V"
        spec = PlotSpec(
            x="voltage_V",
            y=y,
            plot_mode=mode,
            max_points_per_figure=max_points,
            max_points_per_trace=max_points,
        )
        return plot(source, spec=spec)

    def plot_progressive(self, *, magnitude=False, max_points=250000, mode="auto"):
        """Create an idle progressive dQ/dV widget session; display it, then call start()."""
        from .plot_spec import PlotSpec
        from .plotting import plot_progressive

        source = _DQDVPlotRequest(self, magnitude)
        y = "abs_dqdv_mAh_V" if magnitude else "dqdv_mAh_V"
        spec = PlotSpec(
            x="voltage_V",
            y=y,
            plot_mode=mode,
            max_points_per_figure=max_points,
            max_points_per_trace=max_points,
        )
        return plot_progressive(source, spec=spec)

    def peak(
        self,
        *,
        cycle,
        half_cycle,
        step,
        portion,
        window_V,
        extremum=None,
        max_candidates=1000,
    ):
        """Find a signed grid-node window extremum on exactly one readable curve."""
        return prepared_dqdv_peak(
            self,
            cycle=cycle,
            half_cycle=half_cycle,
            step=step,
            portion=portion,
            window_V=window_V,
            extremum=extremum,
            max_candidates=max_candidates,
        )

    def write_parquet(self, path):
        """Stream curve points to Parquet with settings stored once in its manifest."""
        if Path(path).exists():
            raise FileExistsError("Choose a new dQ/dV export path")
        return atomic_parquet(self.iter_batches(), path, manifest=self.report)

    def close(self):
        """Remove owned curve storage."""
        if Path(self._workspace.name).resolve() != self.path.parent:
            raise ValueError("dQ/dV result cleanup target changed")
        self._workspace.cleanup()


@dataclass(frozen=True)
class _DQDVPlotRequest:
    """Internal display request keeping magnitude separate from scientific results."""

    result: PreparedDQDVResult
    magnitude: bool

    def plot_views(self):
        """Resolve lazy curve views without copying curve tables."""
        return self.result.plot_views(magnitude=self.magnitude)


def _selected_cycles(cycles):
    """Return a validated optional cycle set without changing global numbering."""
    if cycles is None:
        return None
    try:
        values = tuple(cycles)
    except TypeError as error:
        raise ValueError("cycles must be an iterable of positive integers") from error
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values):
        raise ValueError("cycles must contain positive integers")
    return frozenset(values)


def eligible_cc_domains(capacity_rows, step_rows, *, cycles=None):
    """Intersect confirmed half-cycles with classified CC steps.

    Parameters are iterables of mappings so the planner remains independent of
    pandas and whole-file measurement materialization. Half-cycle bounds use the
    first and last retained samples; interpolated numerical sign-crossing bounds do
    not fabricate derivative observations. Only ``step_type == 'CC'`` and matching
    current direction are admitted. CV, rest, variable and unknown rows are ignored.
    """
    selected = _selected_cycles(cycles)
    halves = []
    for row in capacity_rows:
        cycle = int(row["cycle"])
        if selected is not None and cycle not in selected:
            continue
        direction = row["direction"]
        start, end = float(row["first_sample_s"]), float(row["last_sample_s"])
        if direction not in {"charge", "discharge"}:
            raise ValueError("Half-cycle direction must be charge or discharge")
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError("Half-cycle sample bounds must be finite and ordered")
        halves.append((start, end, cycle, int(row["half_cycle"]), direction))
    steps = []
    for row in step_rows:
        if row["step_type"] != "CC":
            continue
        direction = row["direction"]
        if direction not in {"charge", "discharge"}:
            continue
        start, end = float(row["start_s"]), float(row["end_s"])
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError("Step bounds must be finite and ordered")
        steps.append((start, end, int(row["step"]), direction))

    domains = []
    step_position = 0
    steps.sort()
    for h_start, h_end, cycle, half, direction in sorted(halves):
        while step_position < len(steps) and steps[step_position][1] < h_start:
            step_position += 1
        position = step_position
        while position < len(steps) and steps[position][0] <= h_end:
            s_start, s_end, step, s_direction = steps[position]
            if s_direction == direction:
                start, end = max(h_start, s_start), min(h_end, s_end)
                if end >= start:
                    domains.append(DQDVDomain(cycle, half, step, direction, start, end))
            position += 1
    return tuple(domains)


def plan_dqdv_domains(
    source,
    *,
    cycles=None,
    capacity=None,
    steps=None,
    memory_budget=None,
    cancellation=None,
    **step_settings,
):
    """Classify only when needed and return compact CC-only derivative domains.

    Existing ``capacity`` and ``steps`` results may be supplied to reuse prior work.
    Otherwise this call performs the required half-cycle and scoped step analyses.
    Owned temporary results are closed before returning the small immutable plan.
    """
    own_capacity = capacity is None
    own_steps = steps is None
    capacity = capacity or source.capacity(
        include_energy=False, memory_budget=memory_budget, cancellation=cancellation
    )
    steps = steps or source.steps(
        memory_budget=memory_budget, cancellation=cancellation, **step_settings
    )
    try:
        capacity_rows = capacity.details().table().to_dict("records")
        step_rows = steps.table().to_dict("records")
        return eligible_cc_domains(capacity_rows, step_rows, cycles=cycles)
    finally:
        if own_steps:
            steps.close()
        if own_capacity:
            capacity.close()


def spool_dqdv_support(
    source,
    domains,
    *,
    missing_span_s=10.0,
    cancellation=None,
    scratch_dir=None,
):
    """Store only eligible supported points and split every remaining support gap.

    Isolated current and voltage estimates come from the shared approved estimator.
    Estimates do not establish CC eligibility because ``domains`` were independently
    classified first. An invalid current interval or voltage coordinate closes the
    current portion. Later support starts a new portion with a local Q offset while
    retaining ``partial_context=True``.
    """
    domains = tuple(sorted(domains, key=lambda item: (item.start_s, item.end_s)))
    token = cancellation or CancellationToken()
    workspace = tempfile.TemporaryDirectory(prefix="gregorithm-dqdv-support-", dir=scratch_dir)
    path = Path(workspace.name) / "support.sqlite"
    database = sqlite3.connect(path)
    try:
        database.execute("PRAGMA journal_mode=OFF")
        database.execute("PRAGMA synchronous=OFF")
        database.execute(
            "CREATE TABLE samples (domain INTEGER,portion INTEGER,sequence INTEGER,"
            "time_s REAL,voltage_V REAL,capacity_mAh REAL,estimated_current INTEGER,"
            "estimated_voltage INTEGER,PRIMARY KEY(domain,portion,sequence))"
        )
        database.execute(
            "CREATE TABLE portions (domain INTEGER,cycle INTEGER,half_cycle INTEGER,"
            "step INTEGER,portion INTEGER,direction TEXT,start_s REAL,end_s REAL,"
            "n_samples INTEGER,estimated_current INTEGER,estimated_voltage INTEGER,"
            "partial_context INTEGER,PRIMARY KEY(domain,portion))"
        )
        states = {}
        domain_position = 0
        previous = None
        net_mAh = 0.0
        partial_context = False
        columns = [source.report["time_column"], "current_mA", "voltage_V"]
        reader = _effective_batches(source, columns, missing_span_s, token)
        try:
            for times, _, current, voltage, estimated_i, estimated_v in reader:
                token.check()
                for position, time_s in enumerate(times):
                    if previous is None:
                        q_valid = True
                    else:
                        increment = directional_integrals(
                            np.array([previous[0], time_s]),
                            np.array([previous[1], current[position]]),
                        )[0]
                        q_valid = bool(np.all(np.isfinite(increment[:2])))
                        if q_valid:
                            net_mAh += float(increment[0] - increment[1])
                        else:
                            partial_context = True
                    while (
                        domain_position < len(domains) and domains[domain_position].end_s < time_s
                    ):
                        domain_position += 1
                    scan = domain_position
                    while scan < len(domains) and domains[scan].start_s <= time_s:
                        domain = domains[scan]
                        if time_s <= domain.end_s:
                            valid = q_valid and np.isfinite(voltage[position])
                            state = states.setdefault(
                                scan,
                                {"portion": 0, "open": False, "sequence": 0, "offset": 0.0},
                            )
                            if not valid:
                                state["open"] = False
                            else:
                                oriented = net_mAh if domain.direction == "charge" else -net_mAh
                                if not state["open"]:
                                    state["portion"] += 1
                                    state["sequence"] = 0
                                    state["offset"] = oriented
                                    state["open"] = True
                                    database.execute(
                                        "INSERT INTO portions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                                        (
                                            scan,
                                            domain.cycle,
                                            domain.half_cycle,
                                            domain.step,
                                            state["portion"],
                                            domain.direction,
                                            float(time_s),
                                            float(time_s),
                                            0,
                                            0,
                                            0,
                                            int(partial_context),
                                        ),
                                    )
                                state["sequence"] += 1
                                database.execute(
                                    "INSERT INTO samples VALUES (?,?,?,?,?,?,?,?)",
                                    (
                                        scan,
                                        state["portion"],
                                        state["sequence"],
                                        float(time_s),
                                        float(voltage[position]),
                                        float(oriented - state["offset"]),
                                        int(estimated_i[position]),
                                        int(estimated_v[position]),
                                    ),
                                )
                                database.execute(
                                    "UPDATE portions SET end_s=?,n_samples=n_samples+1,"
                                    "estimated_current=estimated_current OR ?,"
                                    "estimated_voltage=estimated_voltage OR ?,"
                                    "partial_context=partial_context OR ? "
                                    "WHERE domain=? AND portion=?",
                                    (
                                        float(time_s),
                                        int(estimated_i[position]),
                                        int(estimated_v[position]),
                                        int(partial_context),
                                        scan,
                                        state["portion"],
                                    ),
                                )
                        scan += 1
                    previous = (time_s, current[position])
            database.commit()
        finally:
            reader.close()
    except BaseException:
        database.close()
        workspace.cleanup()
        raise
    database.close()
    report = {
        "operation": "dqdv_cc_support",
        "missing_span_s": float(missing_span_s),
        "generated_measurement_identifiers": False,
        "capacity_origin": "local_per_contiguous_portion",
    }
    return DQDVSupport(source, path, domains, report, workspace)


def _grid_settings(database, domain, portion, config):
    """Resolve one portion grid from bounded SQL aggregates, or return a reason."""
    row = database.execute(
        "SELECT MIN(voltage_V),MAX(voltage_V),COUNT(*) FROM samples WHERE domain=? AND portion=?",
        (domain, portion),
    ).fetchone()
    lo, hi, total = row
    if total < 3 or not hi > lo:
        return None, "fewer_than_3_distinct_voltages"
    trim_lo, trim_hi = lo + config.trim_extremes_V, hi - config.trim_extremes_V
    operator_lo, operator_hi = (">", "<") if config.trim_extremes_V > 0 else (">=", "<=")
    support_condition = f"voltage_V {operator_lo} ? AND voltage_V {operator_hi} ?"
    available_lo, available_hi = database.execute(
        f"SELECT MIN(voltage_V),MAX(voltage_V) FROM samples "
        f"WHERE domain=? AND portion=? AND {support_condition}",
        (domain, portion, trim_lo, trim_hi),
    ).fetchone()
    if available_lo is None or not available_hi > available_lo:
        return None, "empty_voltage_window_after_trim"
    e_lo = max(available_lo, config.e_lo) if config.e_lo is not None else available_lo
    e_hi = min(available_hi, config.e_hi) if config.e_hi is not None else available_hi
    if e_hi <= e_lo:
        return None, "empty_voltage_window_after_trim"
    usable, distinct = database.execute(
        "SELECT COUNT(*),COUNT(DISTINCT voltage_V) FROM samples "
        "WHERE domain=? AND portion=? AND voltage_V>=? AND voltage_V<=?",
        (domain, portion, e_lo, e_hi),
    ).fetchone()
    if distinct < 3:
        return None, "fewer_than_3_distinct_voltages"
    requested = round((config.fraction or 0.1) * usable)
    if config.profile == "voltage_grid":
        step = config.voltage_step_V or 0.002
        count = int(np.floor((e_hi - e_lo) / step)) + 1
        grid = e_lo + np.arange(count) * step
    else:
        count = config.n_points if config.n_points is not None else max(3, requested)
        if config.n_points is None and config.max_auto_points is not None:
            count = min(config.max_auto_points, count)
        grid = np.linspace(e_lo, e_hi, count)
    if len(grid) < 3:
        return None, "fewer_than_3_grid_nodes"
    return (
        grid,
        support_condition,
        trim_lo,
        trim_hi,
        usable,
        distinct,
        requested,
    ), None


def build_dqdv_curves(support, *, config=None, scratch_dir=None):
    """Construct bounded uniform-grid curves from a prepared CC-support spool."""
    config = config or DQDVConfig()
    workspace = tempfile.TemporaryDirectory(prefix="gregorithm-dqdv-result-", dir=scratch_dir)
    path = Path(workspace.name) / "curves.sqlite"
    output = sqlite3.connect(path)
    source = sqlite3.connect(support.path)
    try:
        output.execute("PRAGMA journal_mode=OFF")
        output.execute("PRAGMA synchronous=OFF")
        output.execute(
            "CREATE TABLE curve_points (cycle INTEGER,half_cycle INTEGER,step INTEGER,"
            "portion INTEGER,direction TEXT,voltage_V REAL,capacity_mAh REAL,"
            "capacity_unsmoothed_mAh REAL,dqdv_mAh_V REAL,dqdv_unsmoothed_mAh_V REAL)"
        )
        output.execute(
            "CREATE TABLE outcomes (cycle INTEGER,half_cycle INTEGER,step INTEGER,"
            "portion INTEGER,direction TEXT,valid INTEGER,reason TEXT,n_samples INTEGER,"
            "n_distinct_voltage INTEGER,n_grid INTEGER,effective_step_V REAL,"
            "estimated_current INTEGER,estimated_voltage INTEGER,partial_context INTEGER,"
            "nonmonotonic_voltage INTEGER,smoothing_width_V REAL)"
        )
        portion_rows = source.execute(
            "SELECT domain,cycle,half_cycle,step,portion,direction,n_samples,"
            "estimated_current,estimated_voltage,partial_context FROM portions "
            "ORDER BY domain,portion"
        ).fetchall()
        for (
            domain,
            cycle,
            half,
            step,
            portion,
            direction,
            n_samples,
            est_i,
            est_v,
            partial,
        ) in portion_rows:
            settings, reason = _grid_settings(source, domain, portion, config)
            if reason:
                output.execute(
                    "INSERT INTO outcomes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        cycle,
                        half,
                        step,
                        portion,
                        direction,
                        0,
                        reason,
                        n_samples,
                        0,
                        0,
                        None,
                        est_i,
                        est_v,
                        partial,
                        0,
                        0.0,
                    ),
                )
                continue
            grid, condition, support_lo, support_hi, usable, distinct, requested = settings
            if len(grid) * 192 > 512 * 1024**2:
                raise BudgetExceeded("Requested dQ/dV grid exceeds the 512 MiB curve buffer")
            neighbors = GridNeighbors(grid, "mean")
            cursor = source.execute(
                f"SELECT sequence,voltage_V,capacity_mAh FROM samples "
                f"WHERE domain=? AND portion=? AND {condition} ORDER BY sequence",
                (domain, portion, support_lo, support_hi),
            )
            previous_voltage = None
            nonmonotonic = False
            while rows := cursor.fetchmany(4096):
                ids = np.asarray([row[0] for row in rows], dtype=np.int64)
                voltage = np.asarray([row[1] for row in rows], dtype=np.float64)
                capacity = np.asarray([row[2] for row in rows], dtype=np.float64)
                sequence_voltage = voltage
                for value in sequence_voltage:
                    if previous_voltage is not None and (
                        direction == "charge"
                        and value < previous_voltage
                        or direction == "discharge"
                        and value > previous_voltage
                    ):
                        nonmonotonic = True
                    previous_voltage = value
                neighbors.update(voltage, capacity, ids)
            interpolated, _ = neighbors.interpolate()
            unsmoothed = interpolated.copy()
            if config.smoothing:
                if config.window_points > len(grid):
                    raise ValueError("Smoothing window longer than grid; no silent adaptation")
                from scipy.signal import savgol_filter

                interpolated = savgol_filter(
                    interpolated, config.window_points, config.polyorder, mode="interp"
                )
            derivative = np.gradient(interpolated, grid, edge_order=1)
            derivative_raw = np.gradient(unsmoothed, grid, edge_order=1)
            output.executemany(
                "INSERT INTO curve_points VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    (
                        cycle,
                        half,
                        step,
                        portion,
                        direction,
                        float(voltage),
                        float(capacity),
                        float(raw),
                        float(value),
                        float(raw_value) if config.smoothing else None,
                    )
                    for voltage, capacity, raw, value, raw_value in zip(
                        grid, interpolated, unsmoothed, derivative, derivative_raw, strict=True
                    )
                ),
            )
            spacing = float(grid[1] - grid[0])
            output.execute(
                "INSERT INTO outcomes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    cycle,
                    half,
                    step,
                    portion,
                    direction,
                    1,
                    None,
                    usable,
                    distinct,
                    len(grid),
                    spacing,
                    est_i,
                    est_v,
                    partial,
                    int(nonmonotonic),
                    (config.window_points - 1) * spacing if config.smoothing else 0.0,
                ),
            )
        output.commit()
    except BaseException:
        source.close()
        output.close()
        workspace.cleanup()
        raise
    source.close()
    output.close()
    report = {
        **support.report,
        "operation": "dqdv",
        "profile": config.profile,
        "dqdv_config": config.__dict__,
        "repeated_voltage": "mean_capacity",
        "generated_curve_identifiers": False,
    }
    return PreparedDQDVResult(support.source, path, report, workspace)


def prepared_source_dqdv(
    source,
    *,
    cycles=None,
    config=None,
    capacity=None,
    steps=None,
    missing_span_s=10.0,
    memory_budget=None,
    cancellation=None,
    scratch_dir=None,
    **step_settings,
):
    """Calculate readable CC-only dQ/dV curves from one prepared source.

    Usage: ``result = data.dqdv(cycles=[2]); result.table()``. Classification is
    scoped to this explicit call unless an existing step result is supplied.
    """
    domains = plan_dqdv_domains(
        source,
        cycles=cycles,
        capacity=capacity,
        steps=steps,
        memory_budget=memory_budget,
        cancellation=cancellation,
        **step_settings,
    )
    support = spool_dqdv_support(
        source,
        domains,
        missing_span_s=missing_span_s,
        cancellation=cancellation,
        scratch_dir=scratch_dir,
    )
    try:
        return build_dqdv_curves(support, config=config, scratch_dir=scratch_dir)
    finally:
        support.close()


def _curve_key_values(cycle, half_cycle, step, portion):
    """Validate one readable curve selection and return its SQLite parameters."""
    values = (cycle, half_cycle, step, portion)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values):
        raise ValueError("Curve selection requires positive integer labels")
    return values


def prepared_dqdv_peak(
    result,
    *,
    cycle,
    half_cycle,
    step,
    portion,
    window_V,
    extremum=None,
    max_candidates=1000,
):
    """Return all tied signed extrema on calculated nodes inside an inclusive window."""
    key = _curve_key_values(cycle, half_cycle, step, portion)
    try:
        lo, hi = (float(value) for value in window_V)
    except (TypeError, ValueError) as error:
        raise ValueError("window_V requires two finite ordered voltages") from error
    if not math.isfinite(lo) or not math.isfinite(hi) or hi < lo:
        raise ValueError("window_V requires two finite ordered voltages")
    if (
        isinstance(max_candidates, bool)
        or not isinstance(max_candidates, int)
        or max_candidates < 1
    ):
        raise ValueError("max_candidates must be a positive integer")
    with closing(sqlite3.connect(result.path)) as database:
        outcome = database.execute(
            "SELECT direction,valid,reason,effective_step_V FROM outcomes "
            "WHERE cycle=? AND half_cycle=? AND step=? AND portion=?",
            key,
        ).fetchone()
        if outcome is None:
            raise ValueError("Selected dQ/dV curve does not exist")
        direction, valid, reason, spacing = outcome
        method = extremum or ("min" if direction == "discharge" else "max")
        if method not in {"min", "max"}:
            raise ValueError("extremum must be 'min' or 'max'")
        if not valid:
            return {
                "valid": False,
                "reason": reason,
                "curve": dict(zip(("cycle", "half_cycle", "step", "portion"), key, strict=True)),
                "window_V": (lo, hi),
                "candidates": [],
                "candidate_count": 0,
                "ambiguous": False,
            }
        domain = database.execute(
            "SELECT MIN(voltage_V),MAX(voltage_V) FROM curve_points "
            "WHERE cycle=? AND half_cycle=? AND step=? AND portion=?",
            key,
        ).fetchone()
        order = "ASC" if method == "min" else "DESC"
        best = database.execute(
            "SELECT dqdv_mAh_V FROM curve_points WHERE cycle=? AND half_cycle=? "
            "AND step=? AND portion=? AND voltage_V>=? AND voltage_V<=? "
            f"AND dqdv_mAh_V IS NOT NULL ORDER BY dqdv_mAh_V {order} LIMIT 1",
            (*key, lo, hi),
        ).fetchone()
        if best is None:
            return {
                "valid": False,
                "reason": "no_finite_grid_node_in_window",
                "curve": dict(zip(("cycle", "half_cycle", "step", "portion"), key, strict=True)),
                "window_V": (lo, hi),
                "candidates": [],
                "candidate_count": 0,
                "ambiguous": False,
            }
        count = database.execute(
            "SELECT COUNT(*) FROM curve_points WHERE cycle=? AND half_cycle=? AND step=? "
            "AND portion=? AND voltage_V>=? AND voltage_V<=? AND dqdv_mAh_V=?",
            (*key, lo, hi, best[0]),
        ).fetchone()[0]
        candidates = database.execute(
            "SELECT voltage_V,dqdv_mAh_V FROM curve_points WHERE cycle=? AND half_cycle=? "
            "AND step=? AND portion=? AND voltage_V>=? AND voltage_V<=? AND dqdv_mAh_V=? "
            "ORDER BY voltage_V LIMIT ?",
            (*key, lo, hi, best[0], max_candidates),
        ).fetchall()
    return {
        "valid": True,
        "reason": None,
        "curve": dict(zip(("cycle", "half_cycle", "step", "portion"), key, strict=True)),
        "direction": direction,
        "extremum": method,
        "window_V": (lo, hi),
        "candidates": [
            {"voltage_V": voltage, "height_mAh_V": height} for voltage, height in candidates
        ],
        "candidate_count": count,
        "candidates_truncated": count > len(candidates),
        "ambiguous": count > 1,
        "grid_step_V": spacing,
        "at_window_edge": any(voltage in {lo, hi} for voltage, _ in candidates),
        "at_support_edge": any(voltage in set(domain) for voltage, _ in candidates),
        "precision": "calculated_grid_nodes_only",
        "smoothing": bool(result.report["dqdv_config"]["smoothing"]),
    }


def compare_prepared_dqdv_peaks(
    reference,
    reference_curve,
    comparison,
    comparison_curve,
    window_V,
    *,
    extremum=None,
):
    """Return comparison-minus-reference voltage for two unambiguous grid extrema."""
    first = reference.peak(**reference_curve, window_V=window_V, extremum=extremum)
    second = comparison.peak(**comparison_curve, window_V=window_V, extremum=extremum)
    comparable = (
        first["valid"]
        and second["valid"]
        and not first["ambiguous"]
        and not second["ambiguous"]
        and first.get("direction") == second.get("direction")
        and first.get("extremum") == second.get("extremum")
    )
    return {
        "reference": first,
        "comparison": second,
        "deltaV_V": (
            second["candidates"][0]["voltage_V"] - first["candidates"][0]["voltage_V"]
            if comparable
            else None
        ),
        "comparable": comparable,
        "different_resolution": first.get("grid_step_V") != second.get("grid_step_V"),
        "precision": "calculated_grid_nodes_only",
    }
