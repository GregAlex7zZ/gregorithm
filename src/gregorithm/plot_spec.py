"""Backend-independent plotting contracts and bounded deterministic reduction."""

import json
import math
import uuid
from dataclasses import asdict, dataclass, field, replace
from time import monotonic, sleep

import numpy as np

from .diagnostics import Cancelled, PlotBudgetExceeded
from .execution import CancellationToken


@dataclass(frozen=True)
class PlotSpec:
    """Immutable whole-figure display/rendering limits. Default auto reduces only display points; full reports partial on overflow. Use preset() for common axes, retaining explicit scientific table selections."""

    x: str = "time_s"
    y: str = "voltage_V"
    plot_mode: str = "auto"
    render_mode: str = "auto"
    max_points_per_figure: int = 250000
    max_points_per_trace: int = 250000
    max_plot_payload_bytes: int = 64 * 1024**2
    max_update_bytes: int = 4 * 1024**2
    publish_interval_s: float = 0.2
    publish_every_batches: int = 1
    queue_size: int = 2
    generation_id: int = 0

    def __post_init__(self):
        """Validate immutable constructor settings and reject unsupported or scientifically contradictory values."""
        if self.plot_mode not in {"auto", "full", "downsample", "aggregate"}:
            raise ValueError("plot_mode")
        if self.render_mode not in {"auto", "svg", "webgl"}:
            raise ValueError("render_mode")
        if (
            min(
                self.max_points_per_figure,
                self.max_points_per_trace,
                self.max_plot_payload_bytes,
                self.max_update_bytes,
            )
            < 1
        ):
            raise ValueError("Positive plotting budget")
        if self.publish_interval_s < 0.2 or self.publish_every_batches < 1 or self.queue_size < 1:
            raise ValueError("Publication >=0.2s, positive queue/batch threshold")

    def to_dict(self):
        """Return serializable configuration/provenance fields without materializing signal data."""
        return asdict(self)

    @classmethod
    def preset(cls, name, **overrides):
        """Build a plotting specification without altering or analyzing input data.

        Names: voltage_time, current_time, voltage_capacity, capacity_cycle,
        dqdv_voltage, dcir_delay. Pass the appropriate sample/metric/curve/DCIR
        TableView to plot(); select metric, direction, view, and branch explicitly
        before plotting mixed long tables. Overrides replace any PlotSpec field.
        Example: PlotSpec.preset("dqdv_voltage", max_points_per_figure=5000).
        """
        axes = {
            "voltage_time": ("time_s", "voltage_V"),
            "current_time": ("time_s", "current_mA"),
            "voltage_capacity": ("capacity_mAh", "voltage_V"),
            "capacity_cycle": ("group_id", "value"),
            "dqdv_voltage": ("voltage_V", "dqdv_mAh_V"),
            "dcir_delay": ("delay_s", "R_ohm"),
        }
        if name not in axes:
            raise ValueError(f"Unknown plotting preset: {name}")
        x, y = axes[name]
        return cls(**{"x": x, "y": y, **overrides})


@dataclass(frozen=True)
class PlotUpdate:
    """Immutable backend-independent plot data/status record with generation, identity, progress, reduction method, and global budgets. points_rendered stays None without frontend confirmation."""

    plot_id: str
    plot_trace_id: str | None
    trace_id: str | None
    generation_id: int
    index: int
    operation: str
    phase: str
    status: str
    data: dict = field(default_factory=dict)
    rows_read: int = 0
    rows_selected: int = 0
    points_prepared: int = 0
    points_emitted: int = 0
    points_represented: int = 0
    points_rendered: int | None = None
    batches_completed: int = 0
    total: int | None = None
    percent: float | None = None
    elapsed_s: float = 0
    eta_s: float | None = None
    display_method: str = ""
    budget: dict = field(default_factory=dict)
    revision: int = 0
    reason: str | None = None

    def to_dict(self):
        """Return serializable configuration/provenance fields without materializing signal data."""
        return asdict(self)


def plot_views(source):
    """Resolve supported plot sources into separate trace/view pairs; pipeline curve manifests split curve IDs into independent series. No analytical values are recalculated by this adapter.

    Parameters
    ----------
    source
        Trace, selection, collection, or supported lazy result/view; see the return-scope contract."""
    if hasattr(source, "plot_views"):
        return source.plot_views()

    from .io import TableView
    from .model import AnalysisResult, BatchResult, BatteryTrace, TraceCollection, TraceSelection

    if isinstance(source, TableView):
        if (
            source.declared_schema is not None
            and "curve_id" in source.declared_schema.names
            and source.manifest.get("curve_manifests")
            and not source.manifest.get("curve_id")
        ):
            curves = []
            for metadata in source.manifest["curve_manifests"]:
                view = source.select(curve_id=metadata["curve_id"])
                view = replace(
                    view,
                    manifest={
                        **view.manifest,
                        "curve_id": metadata["curve_id"],
                        "curve_manifests": [],
                    },
                )
                curves.append((source.manifest.get("trace_id", "table"), view))
            return curves
        return [(source.manifest.get("trace_id", "table"), source)]
    if isinstance(source, BatteryTrace):
        return list(source.select().iter_traces())
    if isinstance(source, (TraceSelection, TraceCollection)):
        return list(source.iter_traces())
    if isinstance(source, AnalysisResult):
        return [(source.trace.trace_id, source.samples)]
    if isinstance(source, BatchResult):
        return [
            (k, r.samples)
            for k, r in source.by_trace.items()
            if r.status in {"complete", "partial"}
        ]
    if isinstance(source, (list, tuple)):
        out = []
        for s in source:
            out.extend(plot_views(s))
        return out
    raise TypeError("Plot source: view, trace selection, result or collection")


def _finite(x):
    """Convert numerical plotting values to JSON-compatible scalars, retaining nonfinite values as null gaps."""
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    return x


def iter_plot_updates(source, *, spec=None, cancellation=None):
    """Yield serialized-backend-independent progress/data/status updates using two bounded passes. Example: for update in iter_plot_updates(view, spec=PlotSpec()): consume(update). Final reduction is independent of publication cadence; budgets apply to the complete figure and terminal outcomes are explicit.

    Parameters
    ----------
    source
        Trace, selection, collection, or supported lazy result/view; see the return-scope contract.
    spec
        PlotSpec shared across the complete figure, including all series and payload limits.
    cancellation
        CancellationToken checked at cooperative boundaries; None creates a local token."""
    spec = spec or PlotSpec()
    token = cancellation or CancellationToken()
    views = plot_views(source)
    pid = uuid.uuid4().hex
    start = monotonic()
    last = -1e100
    seq = 0
    emitted = 0
    represented = 0
    read = 0
    prepared = 0
    nb = 0
    last_published = -1e100
    counts = []
    mode = spec.plot_mode

    def pace(terminal=False):
        """Enforce the maximum running-update publication rate while checking cancellation."""
        nonlocal last_published
        if not terminal:
            while monotonic() - last_published < spec.publish_interval_s:
                token.check()
                sleep(min(0.02, spec.publish_interval_s - (monotonic() - last_published)))
        last_published = monotonic()

    def status(phase, state="running", reason=None, total=None):
        """Build a serializable progress/terminal record without claiming frontend rendering."""
        nonlocal seq
        pace(state != "running")
        seq += 1
        return PlotUpdate(
            pid,
            None,
            None,
            spec.generation_id,
            seq,
            "status",
            phase,
            state,
            rows_read=read,
            rows_selected=read,
            points_prepared=prepared,
            points_emitted=emitted,
            points_represented=represented,
            batches_completed=nb,
            total=total,
            elapsed_s=monotonic() - start,
            display_method=mode,
            budget=spec.to_dict(),
            reason=reason,
        )

    try:
        if not views:
            yield status("completion", "complete")
            return
        budget = min(
            spec.max_points_per_trace,
            spec.max_points_per_figure // len(views),
            spec.max_plot_payload_bytes // (512 * len(views)),
        )
        if budget < 4:
            raise PlotBudgetExceeded("Too many series for global point/byte budget")
        for tid, view in views:
            count = 0
            for b in view.iter_batches():
                token.check()
                count += b.num_rows
                read += b.num_rows
                nb += 1
                now = monotonic()
                if now - last >= spec.publish_interval_s:
                    yield status("reading")
                    last = monotonic()
            counts.append(count)
        if mode == "auto":
            mode = "full" if all(n <= budget for n in counts) else "downsample"
        yield status("preparation", total=sum(counts))
        represented_by = {}
        for trace_number, ((tid, view), count) in enumerate(zip(views, counts)):
            series = f"{tid}:{trace_number}"
            cap = budget
            bins = max(1, cap // 8)
            width = max(1, int(np.ceil(count / bins)))
            bins_data = {}
            full = []
            anchors = {}
            ordinal = 0
            previous = None
            identity = None
            revision = 0
            full_overflow = False

            def snapshot():
                """Produce a bounded current display envelope or explicit aggregate from retained bin state."""
                rows = []
                if mode == "full":
                    rows = full[:]
                else:
                    for bin_id, state in sorted(bins_data.items()):
                        if mode == "aggregate":
                            first, last, mini, maxi, total, n, valid = state
                            rows.append(
                                (
                                    first[0],
                                    total / valid if valid else None,
                                    None,
                                    first[3],
                                    {
                                        "bin_id": bin_id,
                                        "count": n,
                                        "valid_count": valid,
                                        "min": _finite(mini[1]),
                                        "max": _finite(maxi[1]),
                                        "first_" + (identity or "row_id"): first[2],
                                        "last_" + (identity or "row_id"): last[2],
                                    },
                                )
                            )
                        else:
                            unique = {r[3]: r for r in state[:4]}
                            rows.extend(unique.values())
                    rows.extend(anchors.values())
                    rows = list({r[3]: r for r in rows}.values())
                    rows.sort(key=lambda r: r[3])
                    if len(rows) > cap:
                        # Anchors have a bounded reserve; discarded detail is declared.
                        rows = rows[:cap]
                return rows

            def payload(rows):
                """Serialize display-only points with original identities and reduction metadata."""
                return {
                    "x": [_finite(r[0]) for r in rows],
                    "y": [_finite(r[1]) for r in rows],
                    "ids": [_finite(r[2]) for r in rows],
                    "aggregate": [r[4] for r in rows],
                    "display_reduced": mode != "full",
                    "source_points": count,
                    "selection": view.manifest.get("selection_id"),
                    "result_id": view.manifest.get("result_id"),
                    "source_revision": view.manifest.get("source_revision"),
                    "assignment_revision": view.manifest.get("assignment_revision"),
                    "curve_id": view.manifest.get("curve_id"),
                    "provenance": [r[5] if len(r) > 5 else {} for r in rows],
                    "identity_kind": "bin"
                    if mode == "aggregate"
                    else (
                        "sample"
                        if identity == "sample_id"
                        else "curve_point"
                        if identity == "curve_point_id"
                        else "row"
                    ),
                }

            def send(rows):
                """Publish bounded replacement/append chunks while enforcing whole-figure and update byte limits."""
                nonlocal seq, emitted, represented, revision
                revision += 1
                serialized = len(json.dumps(payload(rows)).encode("utf8"))
                if serialized * 2 > spec.max_plot_payload_bytes // len(views):
                    raise PlotBudgetExceeded(
                        "Serialized figure data and hover exceed payload budget"
                    )
                chunk = max(1, spec.max_update_bytes // 512)
                represented_by[series] = len(rows)
                represented = sum(represented_by.values())
                k = 0
                while k < max(1, len(rows)):
                    pace()
                    block = rows[k : k + chunk]
                    while (
                        len(json.dumps(payload(block)).encode("utf8")) + 2048
                        > spec.max_update_bytes
                    ):
                        if len(block) <= 1:
                            raise PlotBudgetExceeded(
                                "One point plus metadata exceeds update budget"
                            )
                        chunk = max(1, len(block) // 2)
                        block = rows[k : k + chunk]
                    seq += 1
                    emitted += len(block)
                    yield PlotUpdate(
                        pid,
                        series,
                        tid,
                        spec.generation_id,
                        seq,
                        "replace" if k == 0 else "append",
                        "plotting",
                        "running",
                        payload(block),
                        read,
                        read,
                        prepared,
                        emitted,
                        represented,
                        None,
                        nb,
                        sum(counts),
                        None,
                        monotonic() - start,
                        None,
                        mode,
                        spec.to_dict(),
                        revision,
                    )
                    k += max(1, len(block))

            for b in view.iter_batches():
                token.check()
                nb += 1
                if spec.x not in b.schema.names or spec.y not in b.schema.names:
                    raise ValueError("Plot columns absent")
                xs = b.column(spec.x).to_pylist()
                ys = b.column(spec.y).to_pylist()
                identity = (
                    "sample_id"
                    if "sample_id" in b.schema.names
                    else "curve_point_id"
                    if "curve_point_id" in b.schema.names
                    else None
                )
                ids = b.column(identity).to_pylist() if identity else [None] * b.num_rows
                sections = (
                    b.column("time_section_id").to_pylist()
                    if "time_section_id" in b.schema.names
                    else [0] * b.num_rows
                )
                halves = (
                    b.column("half_cycle_id").to_pylist()
                    if "half_cycle_id" in b.schema.names
                    else [0] * b.num_rows
                )
                provenance = [
                    {
                        name: b.column(name)[j].as_py()
                        for name in (
                            "cycle",
                            "half_cycle",
                            "step",
                            "portion",
                            "direction",
                            "trace_id",
                            "result_id",
                            "source_revision",
                            "cycle_id",
                            "half_cycle_id",
                            "step_id",
                            "event_id",
                            "curve_id",
                            "support_left_sample_id",
                            "support_right_sample_id",
                        )
                        if name in b.schema.names
                    }
                    for j in range(b.num_rows)
                ]
                for x, y, ident, section, half, origin in zip(
                    xs, ys, ids, sections, halves, provenance
                ):
                    row = (x, y, ident, ordinal, None, origin)
                    key = (
                        section,
                        half,
                        origin.get("step_id"),
                        origin.get("event_id"),
                        origin.get("curve_id"),
                        x is not None and y is not None and np.isfinite(x) and np.isfinite(y),
                    )
                    if previous is not None and (
                        (
                            key != previous[1]
                            or (
                                spec.x == "time_s"
                                and x is not None
                                and previous[0][0] is not None
                                and x < previous[0][0]
                            )
                        )
                        or (
                            identity == "sample_id"
                            and ident is not None
                            and previous[0][2] is not None
                            and ident != previous[0][2] + 1
                        )
                    ):
                        # A null separator prevents bridges across disjoint samples/branches.
                        gap = (None, None, None, ordinal - 0.5, None)
                        if mode == "full":
                            if len(full) >= cap:
                                full_overflow = True
                                break
                            full.append(gap)
                        elif len(anchors) + 3 <= cap // 2:
                            anchors[ordinal - 0.5] = gap
                            anchors[previous[0][3]] = previous[0]
                            anchors[ordinal] = row
                        else:
                            raise PlotBudgetExceeded(
                                "Too many disjoint branches/gaps for figure budget; select a smaller domain"
                            )
                    if mode == "full":
                        if len(full) >= cap:
                            full_overflow = True
                            break
                        full.append(row)
                    else:
                        bi = ordinal // width
                        state = bins_data.get(bi)
                        if state is None:
                            bins_data[bi] = [
                                row,
                                row,
                                row,
                                row,
                                float(y) if y is not None and np.isfinite(y) else 0.0,
                                1,
                                int(y is not None and np.isfinite(y)),
                            ]
                        else:
                            state[1] = row
                            if y is not None and np.isfinite(y):
                                if (
                                    state[2][1] is None
                                    or not np.isfinite(state[2][1])
                                    or y < state[2][1]
                                ):
                                    state[2] = row
                                if (
                                    state[3][1] is None
                                    or not np.isfinite(state[3][1])
                                    or y > state[3][1]
                                ):
                                    state[3] = row
                                state[4] += y
                                state[6] += 1
                            state[5] += 1
                    previous = (row, key)
                    ordinal += 1
                    prepared += 1
                now = monotonic()
                if now - last >= spec.publish_interval_s and nb % spec.publish_every_batches == 0:
                    yield from send(snapshot())
                    last = monotonic()
                if full_overflow:
                    break
            yield from send(snapshot())
            if full_overflow:
                yield status(
                    "completion",
                    "partial",
                    "PlotBudgetExceeded: full view exceeds global budget",
                    sum(counts),
                )
                return
            if len(anchors) >= cap // 2:
                yield status("plotting", "running", "Some boundaries exceed anchor reserve")
        yield status("completion", "complete", total=sum(counts))
    except Cancelled:
        yield status("completion", "cancelled", "Cancellation requested")
    except Exception as e:
        yield status(
            "completion", "partial" if represented else "failed", f"{type(e).__name__}: {e}"
        )
