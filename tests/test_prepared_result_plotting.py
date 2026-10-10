"""Simple final/progressive plotting regressions for redesigned compact results."""

import numpy as np
from test_half_cycles import prepared


def test_capacity_and_step_quantity_plots_use_readable_series(tmp_path):
    """Plot compact capacity and step results without manual PlotSpec construction."""
    source = prepared(
        tmp_path,
        range(12),
        [1, 1, 1, 0, -1, -1, -1, 0, 1, 1, 1, 1],
        np.linspace(3, 4, 12),
    )
    capacity = source.capacity()
    steps = source.steps()
    quantities = steps.quantities()
    try:
        voltage_plot = source.plot(y="voltage_V", max_points=6)
        assert voltage_plot.spec.x == "time_s" and voltage_plot.spec.y == "voltage_V"
        assert len(voltage_plot.figure.data[0].x) <= 6
        capacity_plot = capacity.plot(max_points=100)
        assert capacity_plot.spec.x == "half_cycle"
        assert capacity_plot.spec.y == "capacity_mAh"
        assert {trace.name.split(":")[0] for trace in capacity_plot.figure.data} == {
            "charge",
            "discharge",
        }
        step_plot = quantities.plot(y="energy_Wh", max_points=100)
        assert step_plot.spec.x == "step" and step_plot.spec.y == "energy_Wh"
        assert all("ID=" not in trace.hovertemplate for trace in step_plot.figure.data)
    finally:
        quantities.close()
        steps.close()
        capacity.close()


def test_event_and_dcir_plots_and_progressive_updates(tmp_path):
    """Expose event/DCIR figures and running compact-result updates through simple methods."""
    source = prepared(
        tmp_path,
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 0, 10, 10, 0, 0, -10, -10],
        [4.0, 4.0, 3.9, 3.88, 3.95, 3.95, 4.05, 4.06],
    )
    events = source.events()
    dcir = events.dcir(0.5)
    try:
        event_plot = events.plot()
        assert event_plot.spec.y == "observed_pulse_span_s"
        dcir_plot = dcir.plot()
        assert dcir_plot.spec.y == "resistance_ohm"
        session = dcir.plot_progressive(max_points=100)
        try:
            session.start()
            updates = list(session.iter_updates())
            assert any(update.status == "running" for update in updates)
            assert updates[-1].status == "complete"
        finally:
            session.close()
    finally:
        dcir.close()
        events.close()
