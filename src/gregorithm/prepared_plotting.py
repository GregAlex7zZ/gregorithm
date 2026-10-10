"""Simple bounded plotting views for redesigned identity-free results."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass, replace

import pyarrow as pa
import pyarrow.compute as pc

from .plot_spec import PlotSpec


@dataclass(frozen=True)
class PreparedDQDVCurveView:
    """One readable lazy dQ/dV curve view used only by the plotting adapter."""

    path: object
    key: tuple[int, int, int, int]
    direction: str
    magnitude: bool = False
    voltage_range: tuple[float, float] | None = None

    @property
    def manifest(self):
        """Return compact human-readable plot metadata without generated identities."""
        cycle, half, step, portion = self.key
        return {
            "cycle": cycle,
            "half_cycle": half,
            "step": step,
            "portion": portion,
            "direction": self.direction,
            "identity_free": True,
        }

    def select(self, **filters):
        """Return a voltage-clipped view for progressive viewport rereads."""
        if set(filters) != {"voltage_V"}:
            raise ValueError("dQ/dV plot viewport supports voltage_V only")
        request = filters["voltage_V"]
        if not isinstance(request, dict) or set(request) != {"min", "max"}:
            raise ValueError("Voltage viewport requires min and max")
        return replace(self, voltage_range=(float(request["min"]), float(request["max"])))

    def iter_batches(self, batch_rows=4096):
        """Yield bounded plotted columns directly from the compact curve store."""
        y_name = "abs_dqdv_mAh_V" if self.magnitude else "dqdv_mAh_V"
        schema = pa.schema(
            [
                ("voltage_V", pa.float64()),
                (y_name, pa.float64()),
                ("cycle", pa.int64()),
                ("half_cycle", pa.int64()),
                ("step", pa.int64()),
                ("portion", pa.int64()),
                ("direction", pa.string()),
            ]
        )
        expression = "ABS(dqdv_mAh_V)" if self.magnitude else "dqdv_mAh_V"
        where = "cycle=? AND half_cycle=? AND step=? AND portion=?"
        parameters = list(self.key)
        if self.voltage_range is not None:
            where += " AND voltage_V>=? AND voltage_V<=?"
            parameters.extend(self.voltage_range)
        query = (
            f"SELECT voltage_V,{expression},cycle,half_cycle,step,portion,direction "
            f"FROM curve_points WHERE {where} ORDER BY voltage_V"
        )
        with closing(sqlite3.connect(self.path)) as database:
            cursor = database.execute(query, parameters)
            emitted = False
            while rows := cursor.fetchmany(batch_rows):
                emitted = True
                yield pa.RecordBatch.from_pylist(
                    [dict(zip(schema.names, row, strict=True)) for row in rows], schema=schema
                )
            if not emitted:
                yield pa.RecordBatch.from_pylist([], schema=schema)


def dqdv_plot_views(result, *, magnitude=False):
    """Return one lazy plot series per valid readable dQ/dV curve."""
    outcomes = result.outcomes()
    views = []
    for row in outcomes.itertuples(index=False):
        if not row.valid:
            continue
        key = (int(row.cycle), int(row.half_cycle), int(row.step), int(row.portion))
        label = f"cycle {key[0]} | half-cycle {key[1]} | step {key[2]} | portion {key[3]}"
        views.append((label, PreparedDQDVCurveView(result.path, key, row.direction, magnitude)))
    return views


@dataclass(frozen=True)
class CompactResultView:
    """Lazy filtered x/y view over a compact redesigned result."""

    result: object
    x: str
    y: str
    series_column: str | None = None
    series_value: str | None = None
    x_range: tuple[float, float] | None = None

    @property
    def manifest(self):
        """Expose readable display metadata once per series."""
        return {
            "series": self.series_value,
            "identity_free": True,
            "operation": self.result.report.get("operation"),
        }

    def select(self, **filters):
        """Return an x-clipped compact view for progressive viewport rereads."""
        if set(filters) != {self.x}:
            raise ValueError(f"Viewport supports {self.x} only")
        request = filters[self.x]
        if not isinstance(request, dict) or set(request) != {"min", "max"}:
            raise ValueError("Viewport requires min and max")
        return replace(self, x_range=(float(request["min"]), float(request["max"])))

    def iter_batches(self):
        """Filter compact source batches without materializing their complete table."""
        reader = self.result.iter_batches()
        try:
            for batch in reader:
                if self.x not in batch.schema.names or self.y not in batch.schema.names:
                    raise ValueError(f"Plot requires columns {self.x!r} and {self.y!r}")
                mask = None
                if self.series_column is not None:
                    mask = pc.equal(batch.column(self.series_column), self.series_value)
                if self.x_range is not None:
                    values = batch.column(self.x)
                    inside = pc.and_(
                        pc.greater_equal(values, self.x_range[0]),
                        pc.less_equal(values, self.x_range[1]),
                    )
                    mask = inside if mask is None else pc.and_(mask, inside)
                yield batch if mask is None else batch.filter(mask)
        finally:
            reader.close()


@dataclass(frozen=True)
class PreparedSignalView:
    """Lazy time-signal view over original prepared measurements."""

    source: object
    y: str
    time_range: tuple[float, float] | None = None

    @property
    def manifest(self):
        """Expose the readable source filename once, without a generated trace ID."""
        return {"source_file": self.source.report["source_file"], "identity_free": True}

    def select(self, **filters):
        """Return a time-clipped source view for progressive viewport rereads."""
        time_column = self.source.report["time_column"]
        if set(filters) != {time_column}:
            raise ValueError(f"Signal viewport supports {time_column} only")
        request = filters[time_column]
        if not isinstance(request, dict) or set(request) != {"min", "max"}:
            raise ValueError("Time viewport requires min and max")
        return replace(self, time_range=(float(request["min"]), float(request["max"])))

    def iter_batches(self):
        """Yield projected original signal batches with an optional time filter."""
        time_column = self.source.report["time_column"]
        reader = self.source.iter_batches(columns=[time_column, self.y])
        try:
            for batch in reader:
                if self.time_range is None:
                    yield batch
                    continue
                times = batch.column(time_column)
                mask = pc.and_(
                    pc.greater_equal(times, self.time_range[0]),
                    pc.less_equal(times, self.time_range[1]),
                )
                yield batch.filter(mask)
        finally:
            reader.close()


@dataclass(frozen=True)
class CompactPlotRequest:
    """Resolve simple readable series for one compact result plot."""

    result: object
    x: str
    y: str
    series_column: str | None
    series_values: tuple[str, ...]

    def plot_views(self):
        """Return lazy views for series values actually present in the compact result."""
        if self.series_column is None:
            return [(self.y, CompactResultView(self.result, self.x, self.y))]
        present = set()
        reader = self.result.iter_batches()
        try:
            for batch in reader:
                if self.series_column not in batch.schema.names:
                    raise ValueError(f"Plot series column {self.series_column!r} is absent")
                present.update(
                    value
                    for value in batch.column(self.series_column).to_pylist()
                    if value is not None
                )
        finally:
            reader.close()
        return [
            (
                str(value),
                CompactResultView(self.result, self.x, self.y, self.series_column, value),
            )
            for value in self.series_values
            if value in present
        ]


def plot_compact_result(
    result,
    *,
    x,
    y,
    series_column=None,
    series_values=(),
    progressive=False,
    max_points=250000,
    mode="auto",
):
    """Create a final figure or idle progressive session for a compact result."""
    request = CompactPlotRequest(result, x, y, series_column, tuple(series_values))
    spec = PlotSpec(
        x=x,
        y=y,
        plot_mode=mode,
        max_points_per_figure=max_points,
        max_points_per_trace=max_points,
    )
    if progressive:
        from .plotting import plot_progressive

        return plot_progressive(request, spec=spec)
    from .plotting import plot

    return plot(request, spec=spec)


def plot_prepared_signal(
    source,
    *,
    y="voltage_V",
    progressive=False,
    max_points=250000,
    mode="auto",
):
    """Plot one original point signal against its retained measurement time."""
    time_column = source.report["time_column"]
    view = PreparedSignalView(source, y)
    spec = PlotSpec(
        x=time_column,
        y=y,
        plot_mode=mode,
        max_points_per_figure=max_points,
        max_points_per_trace=max_points,
    )
    if progressive:
        from .plotting import plot_progressive

        return plot_progressive(_SinglePlotRequest(source.report["source_file"], view), spec=spec)
    from .plotting import plot

    return plot(_SinglePlotRequest(source.report["source_file"], view), spec=spec)


@dataclass(frozen=True)
class _SinglePlotRequest:
    """Provide one readable lazy series to the shared plotting engine."""

    label: str
    view: object

    def plot_views(self):
        """Return the single configured signal series."""
        return [(self.label, self.view)]
