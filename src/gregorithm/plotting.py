"""Optional local Plotly adapter. Core never imports Plotly or widgets."""

import json
from dataclasses import dataclass, field, replace
from queue import Empty, Full, Queue
from threading import Lock, Thread, Timer

from .execution import CancellationToken
from .plot_spec import PlotSpec, iter_plot_updates, plot_views


@dataclass
class PlotResult:
    """Final local Plotly figure, its PlotSpec, explicit completion status, and display/provenance metadata. write_html() saves a static snapshot."""

    figure: object
    spec: PlotSpec
    status: str = "running"
    metadata: dict = field(default_factory=dict)

    def write_html(self, path):
        """Save a standalone Plotly snapshot. It contains only exported points and cannot reread data, zoom-query the source, or receive future updates."""
        self.metadata["html_limitation"] = (
            "Standalone HTML contains only exported points; no live rereading or future updates"
        )
        self.figure.update_layout(meta=self.metadata)
        self.figure.write_html(path, include_plotlyjs=True)


def _apply(figure, update, series, spec):
    """Apply one generation-valid update to the existing Plotly series and retain sample/curve/bin identity in hover data."""
    if update.operation == "status":
        return
    import plotly.graph_objects as go

    key = update.plot_trace_id
    if key not in series:
        series[key] = len(figure.data)
        cls = (
            go.Scattergl
            if spec.render_mode == "webgl"
            or (spec.render_mode == "auto" and update.data.get("source_points", 0) > 10000)
            else go.Scatter
        )
        figure.add_trace(cls(x=[], y=[], name=key, mode="lines", connectgaps=False))
    trace = figure.data[series[key]]
    data = update.data
    # Compact source identity remains available on click/hover.
    custom = [
        [
            ident,
            provenance.get("trace_id", update.trace_id),
            update.generation_id,
            data.get("identity_kind", "row"),
            aggregate["bin_id"] if aggregate is not None else ident,
            provenance.get("result_id", data.get("result_id")),
            provenance.get("source_revision", data.get("source_revision")),
            data.get("assignment_revision"),
            json.dumps(provenance, sort_keys=True),
            json.dumps(aggregate, sort_keys=True, allow_nan=False) if aggregate is not None else "",
        ]
        for ident, aggregate, provenance in zip(
            data["ids"], data["aggregate"], data.get("provenance", [{}] * len(data["ids"]))
        )
    ]
    with figure.batch_update():
        if update.operation == "replace":
            trace.x = data["x"]
            trace.y = data["y"]
            trace.customdata = custom
        else:
            trace.x = list(trace.x) + data["x"]
            trace.y = list(trace.y) + data["y"]
            trace.customdata = list(trace.customdata) + custom
        if all(item is None for item in data["ids"]):
            trace.hovertemplate = "x=%{x}<br>y=%{y}<extra>%{fullData.name}</extra>"
        else:
            trace.hovertemplate = "x=%{x}<br>y=%{y}<br>%{customdata[3]} ID=%{customdata[4]}<br>trace=%{customdata[1]}<br>result=%{customdata[5]}<br>source=%{customdata[6]}<br>assignment=%{customdata[7]}<br>%{customdata[8]}<br>%{customdata[9]}<extra>%{fullData.name}</extra>"


def plot(source, *, spec=None):
    """Create a final local Plotly figure from bounded PlotUpdate records. Example: plot(result.samples, spec=PlotSpec.preset("voltage_time")). Return PlotResult with status and reduction metadata; computation never changes scientific values.

    Parameters
    ----------
    source
        Trace, selection, collection, or supported lazy result/view; see the return-scope contract.
    spec
        PlotSpec shared across the complete figure, including all series and payload limits."""
    import plotly.graph_objects as go

    spec = spec or PlotSpec()
    fig = go.Figure()
    series = {}
    last = None
    updates = 0
    for update in iter_plot_updates(source, spec=spec):
        _apply(fig, update, series, spec)
        last = update
        updates += 1
    fig.update_layout(xaxis_title=spec.x, yaxis_title=spec.y, uirevision="preserve")
    return PlotResult(
        fig,
        spec,
        last.status if last else "complete",
        {
            "last_update": last.to_dict() if last else None,
            "update_count": updates,
            "display_reduced": last.display_method != "full" if last else False,
            "rendering_confirmed": False,
        },
    )


class PlotSession:
    """Visible initial widget; background producer and notebook-thread consumer."""

    def __init__(self, source, spec=None):
        """Initialize the state required by this object. Use the public factory or dataclass configuration unless constructing this internal component deliberately."""
        import plotly.graph_objects as go

        self.source = source
        self.spec = spec or PlotSpec()
        self.figure = go.FigureWidget()
        self.figure.update_layout(
            xaxis_title=self.spec.x, yaxis_title=self.spec.y, uirevision="preserve"
        )
        self.status = "idle"
        self.token = CancellationToken()
        self.queue = Queue(self.spec.queue_size)
        self._terminal_update = None
        self._terminal_lock = Lock()
        self.series = {}
        self.closed = False
        self.thread = None
        self.last_update = None
        self.update_count = 0
        self.transferred_bytes = 0
        self.generation_id = 0
        self._timer = None
        self._loop = None
        self.zoom_available = False
        try:
            from IPython import get_ipython

            shell = get_ipython()
            self._loop = shell.kernel.io_loop if shell and getattr(shell, "kernel", None) else None
        except (ImportError, AttributeError):
            pass
        self.zoom_available = self._loop is not None
        if self.zoom_available:
            self.figure.layout.xaxis.on_change(self._zoom, "range")

    def start(self):
        """Start background production after displaying the existing widget. Use poll() outside a notebook kernel event loop."""
        if self.closed:
            raise RuntimeError("Session closed")
        if self.thread and self.thread.is_alive():
            raise RuntimeError("Session already running")
        self.token = CancellationToken()
        with self._terminal_lock:
            self._terminal_update = None
        self.status = "running"
        self._start_producer(self.source)
        return self

    def _start_producer(self, source):
        """Start the bounded background producer with a captured generation and cancellation token; reserve a separate terminal-status slot."""
        token = self.token
        generation = self.generation_id
        spec = replace(self.spec, generation_id=generation)

        def produce():
            """Run background data production with captured generation, bounded queue, and persistent terminal status."""
            try:
                for update in iter_plot_updates(source, spec=spec, cancellation=token):
                    if self.closed or generation != self.generation_id:
                        break
                    if update.operation == "status" and update.status != "running":
                        with self._terminal_lock:
                            if generation == self.generation_id:
                                self._terminal_update = update
                        if self._loop:
                            self._loop.add_callback(self.poll)
                        break
                    while not self.closed and generation == self.generation_id:
                        try:
                            self.queue.put(update, timeout=0.1)
                            break
                        except Full:
                            if token.cancelled:
                                break
                    if self._loop:
                        self._loop.add_callback(self.poll)
            finally:
                if self._loop and not self.closed:
                    self._loop.add_callback(self.poll)

        self.thread = Thread(target=produce, daemon=True, name="gregorithm-plot")
        self.thread.start()

    def poll(self, max_updates=2):
        """Apply at most max_updates pending records on the consumer thread, discarding obsolete generations; terminal status survives a full queue."""
        if self.closed:
            return []
        out = []
        for _ in range(max_updates):
            try:
                update = self.queue.get_nowait()
            except Empty:
                with self._terminal_lock:
                    update = self._terminal_update
                    self._terminal_update = None
                if update is None:
                    break
            if update.generation_id != self.generation_id:
                continue
            if self.token.cancelled and update.status == "running":
                continue
            _apply(self.figure, update, self.series, self.spec)
            self.last_update = update
            self.update_count += 1
            self.status = update.status
            self.transferred_bytes += len(json.dumps(update.data).encode())
            out.append(update)
        if (not self.queue.empty() or self._terminal_update is not None) and self._loop:
            self._loop.add_callback(self.poll)
        return out

    def iter_updates(self):
        """Consume queued updates explicitly without modifying the figure; use this alternative to poll(), not concurrently with it."""
        while not self.closed:
            try:
                update = self.queue.get(timeout=0.1)
            except Empty:
                with self._terminal_lock:
                    update = self._terminal_update
                    self._terminal_update = None
                if update is not None and update.generation_id == self.generation_id:
                    yield update
                    break
                if self.thread and not self.thread.is_alive():
                    break
                continue
            if update.generation_id == self.generation_id:
                yield update

    def query_viewport(self, lo, hi):
        """Cancel the old generation, reread the selected x domain, and return a new generation ID; stale updates cannot overwrite it.

        Usage: call on the PlotSession object; returned selections/revisions are independent of earlier views."""
        if self.closed:
            raise RuntimeError("Session closed")
        self.token.cancel()
        self.generation_id += 1
        with self._terminal_lock:
            self._terminal_update = None
        self.token = CancellationToken()
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                break
        views = [
            v.select(**{self.spec.x: {"min": lo, "max": hi}}) for _, v in plot_views(self.source)
        ]
        self.status = "running"
        self._start_producer(views)
        return self.generation_id

    def _zoom(self, axis, value):
        """Debounce a live notebook axis-range change and schedule the viewport query on the kernel event loop."""
        if not value or self.closed:
            return
        if self._timer:
            self._timer.cancel()
        self._timer = Timer(
            0.2, lambda: self._loop.add_callback(lambda: self.query_viewport(*value))
        )
        self._timer.daemon = True
        self._timer.start()

    def cancel(self):
        """Request cooperative cancellation; completed analytical results remain available and interrupted work is not marked complete."""
        self.token.cancel()
        self.status = "cancelled"

    def close(self):
        """Release owned resources and stop further use. Plot sessions cancel producers/timers and close the widget; summary stores close SQLite handles."""
        self.closed = True
        self.token.cancel()
        if self._timer:
            self._timer.cancel()
        if self.zoom_available:
            self.figure.layout.xaxis.on_change(None, "range")
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                break
        self.figure.close()
        self.status = "closed"

    def write_html(self, path):
        """Save a standalone Plotly snapshot. It contains only exported points and cannot reread data, zoom-query the source, or receive future updates."""
        PlotResult(
            self.figure,
            self.spec,
            self.status,
            {"generation_id": self.generation_id, "zoom_live": False},
        ).write_html(path)


def plot_progressive(source, *, spec=None):
    """Create an idle PlotSession with a FigureWidget before background execution. Example: session=plot_progressive(view); display(session.figure); session.start(). A notebook event loop applies updates; otherwise call poll() explicitly.

    Parameters
    ----------
    source
        Trace, selection, collection, or supported lazy result/view; see the return-scope contract.
    spec
        PlotSpec shared across the complete figure, including all series and payload limits."""
    return PlotSession(source, spec)
