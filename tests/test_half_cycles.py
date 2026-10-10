"""Synthetic mathematical/segmentation regressions; not experimental cycle truth labels."""

from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gregorithm import HalfCycleConfig, MemoryBudget, ReadOptions, prepare_source
from gregorithm.point_calculations import EstimatedMeasurementWarning, IncompleteIntegrationWarning


def prepared(tmp_path, times, current, voltage=None, batch_rows=100):
    """Write synthetic point signals and keep a different payload on every original row."""
    path = tmp_path / "source.parquet"
    table = pa.table(
        {
            "time_s": np.array(times, dtype=float),
            "current_mA": np.array(current, dtype=float),
            "voltage_V": np.full(len(times), 4.0)
            if voltage is None
            else np.array(voltage, dtype=float),
            "note": [f"fixture-{k}" for k in range(len(times))],
        }
    )
    pq.write_table(table, path, row_group_size=3)
    return prepare_source(
        path, tmp_path / "prepared.parquet", read_options=ReadOptions(batch_rows=batch_rows)
    )


@pytest.mark.parametrize("batch", [1, 2, 5, 100])
def test_confirmed_zero_crossing_conserves_directional_integrals(tmp_path, batch):
    """Split a +1 A to -1 A boundary at zero and conserve all Q/E across batches."""
    source = prepared(tmp_path, [0, 1, 11, 12], [1000, 1000, -1000, -1000], batch_rows=batch)
    budget = MemoryBudget(8 * 1024**2)
    result = source.capacity(include_energy=True, memory_budget=budget)
    assert budget.used == 0
    frame = result.details().table()
    assert frame.half_cycle.tolist() == [1, 2]
    assert frame.cycle.tolist() == [1, 1]
    assert frame.direction.tolist() == ["charge", "discharge"]
    assert frame.cycle_pair_complete.tolist() == [True, True]
    np.testing.assert_allclose(frame.capacity_mAh, [3500 / 3600, 3500 / 3600])
    np.testing.assert_allclose(frame.start_s, [0, 6])
    np.testing.assert_allclose(frame.end_s, [6, 12])
    np.testing.assert_allclose(frame.first_sample_s, [0, 11])
    assert frame.last_sample_s.tolist() == [1, 12]
    global_totals = source.totals()
    for column in ("charge_mAh", "discharge_mAh", "charge_Wh", "discharge_Wh"):
        assert frame[column].sum() == pytest.approx(global_totals.values[column], abs=1e-12)
    assert not set(frame.columns) & {"sample_id", "trace_id", "result_id", "assignment_revision"}
    result.close()
    assert not result.path.exists()


def test_initial_closing_branch_and_alternate_cycle_pairing(tmp_path):
    """Initial discharge occupies incomplete cycle 1 under charge-first pairing."""
    source = prepared(tmp_path, range(10), [-1, -1, 1, 1, -1, -1, 1, 1, -1, -1])
    result = source.capacity()
    frame = result.table()
    assert frame.cycle.tolist() == [1, 2, 2, 3, 3]
    assert frame.cycle_pair_complete.tolist() == [False, True, True, True, True]
    selected = result.cycles([2]).table()
    assert selected.half_cycle.tolist() == [2, 3]
    assert selected.cycle.tolist() == [2, 2]
    assert result.cycles([]).table().empty
    result.close()
    alternate = source.capacity(config=HalfCycleConfig(cycle_order="discharge_then_charge"))
    assert alternate.table().cycle.tolist() == [1, 1, 2, 2, 3]
    alternate.close()


def test_short_reversal_is_retained_in_net_capacity(tmp_path):
    """Two points spanning only 0.05 s do not confirm a new half-cycle."""
    source = prepared(tmp_path, [0, 1, 2, 2.05, 3, 4], [1, 1, -1, -1, 1, 1])
    result = source.capacity()
    frame = result.details().table()
    assert len(frame) == 1
    assert frame.discharge_mAh.iloc[0] > 0
    assert frame.capacity_mAh.iloc[0] == pytest.approx(
        source.totals(include_energy=False).values["net_mAh"]
    )
    result.close()


def test_rest_onset_and_unassigned_initial_residual_are_conserved(tmp_path):
    """Initial rest residual stays unassigned; last-rest to active onset belongs to the new half."""
    source = prepared(tmp_path, [0, 10, 20, 21], [0.0001, 0, -1, -1])
    result = source.capacity(include_energy=True)
    frame = result.details().table()
    assert frame.start_s.tolist() == [10]
    assert frame.first_sample_s.tolist() == [20]
    assert frame.capacity_mAh.iloc[0] == pytest.approx(6 / 3600)
    total = source.totals()
    for name in ("charge_mAh", "discharge_mAh", "charge_Wh", "discharge_Wh"):
        assert frame[name].sum() + (result.report["unassigned"][name] or 0) == pytest.approx(
            total.values[name]
        )
    result.close()


@pytest.mark.parametrize("batch", [1, 2, 100])
def test_estimates_do_not_count_as_confirmation(tmp_path, batch):
    """Only the two real negative observations confirm, despite an intervening estimate."""
    source = prepared(tmp_path, range(7), [1, 1, -1, np.nan, -1, -1, -1], batch_rows=batch)
    with pytest.warns(EstimatedMeasurementWarning):
        result = source.capacity(include_energy=True)
    frame = result.details().table()
    assert frame.first_sample_s.tolist() == [0, 2]
    assert frame.capacity_quality.tolist() == ["complete", "estimated"]
    result.close()


def test_unresolved_gap_resets_confirmation_and_does_not_invent_root(tmp_path):
    """Unknown-current runs interrupt evidence; a same-direction resumption creates no extra half."""
    source = prepared(tmp_path, range(9), [1, 1, -1, np.nan, np.nan, -1, 1, 1, 1])
    with pytest.warns(IncompleteIntegrationWarning):
        result = source.capacity()
    assert result.report["half_cycles"] == 1
    assert result.table().capacity_quality.tolist() == ["partial"]
    result.close()
    source = prepared(tmp_path, range(7), [1, 1, np.nan, np.nan, -1, -1, -1])
    with pytest.warns(IncompleteIntegrationWarning):
        result = source.capacity()
    frame = result.details().table()
    assert frame.start_s.tolist() == [0, 4]
    assert frame.boundary_quality.iloc[1] == "unknown_transition"
    assert frame.capacity_quality.tolist() == ["partial", "complete"]
    result.close()


def test_explicit_diagnostic_marking_keeps_charge_and_energy(tmp_path):
    """A confirmed physical reversal can be cycle-ignored without subtracting its quantities."""
    source = prepared(tmp_path, range(6), [1, 1, -1, -1, 1, 1])
    ordinary = source.capacity()
    assert len(ordinary.table()) == 3
    ordinary.close()
    result = source.capacity(include_energy=True, diagnostic_windows=[(2.0, 3.0)])
    frame = result.details().table()
    assert len(frame) == 1
    totals = source.totals()
    assert frame.capacity_mAh.iloc[0] == pytest.approx(totals.values["net_mAh"])
    assert frame.energy_Wh.iloc[0] == pytest.approx(totals.values["net_Wh"])
    assert frame.discharge_mAh.iloc[0] > 0
    result.close()
    with pytest.raises(ValueError, match="retained measured"):
        source.capacity(diagnostic_windows=[(2.1, 3.0)])


def test_cycle_selection_keeps_source_numbering_and_capacity_only_has_no_voltage_dependency(
    tmp_path,
):
    """Use the simple selected-source API without a V column or hidden CC/CV classification."""
    source = prepared(tmp_path, range(8), [1, 1, -1, -1, 1, 1, -1, -1])
    source = replace(source, read_options=ReadOptions(batch_rows=1))
    result = source.cycles([2]).capacity()
    assert result.table().half_cycle.tolist() == [3, 4]
    assert result.table().cycle.tolist() == [2, 2]
    result.close()


def test_unknown_or_unconfirmed_initial_actions_do_not_fabricate_half_cycles(tmp_path):
    """Retain known unassigned support when there is no confirmed main direction."""
    source = prepared(tmp_path, [0, 0.01, 0.02], [1, -1, 1])
    result = source.capacity()
    assert result.table().empty
    assert result.report["unassigned"]["charge_mAh"] > 0
    assert result.report["half_cycles"] == 0
    result.close()


def test_selection_materialization_limit_is_explicit(tmp_path):
    """A table limit raises rather than silently truncating scientific rows."""
    from gregorithm import MaterializationLimitError

    source = prepared(tmp_path, range(4), [1, 1, -1, -1])
    result = source.capacity()
    with pytest.raises(MaterializationLimitError):
        result.table(max_rows=1)
    result.close()


def test_inclusive_confirmation_at_large_clock_origin(tmp_path):
    """A one-ULP arithmetic guard preserves inclusive decimal duration thresholds."""
    source = prepared(tmp_path, 1e9 + np.array([0.0, 0.1, 0.2, 0.3]), [1, 1, -1, -1])
    result = source.capacity()
    assert result.table().half_cycle.tolist() == [1, 2]
    result.close()


def test_all_diagnostic_endpoints_are_checked_even_when_windows_overlap(tmp_path):
    """A larger valid union cannot hide an invalid original manual boundary."""
    source = prepared(tmp_path, range(6), [1, 1, -1, -1, 1, 1])
    with pytest.raises(ValueError, match="retained measured"):
        source.capacity(diagnostic_windows=[(2.0, 4.0), (2.1, 3.0)])


def test_summary_reader_budget_and_active_reader_cleanup(tmp_path):
    """Readers reserve/release the shared allowance and prevent deletion while SQLite is open."""
    source = prepared(tmp_path, range(4), [1, 1, -1, -1])
    budget = MemoryBudget(8 * 1024**2)
    result = source.capacity(memory_budget=budget)
    reader = result.iter_batches(batch_rows=10**8)
    assert next(reader).num_rows == 2
    assert budget.used > 0
    with pytest.raises(ValueError, match="active summary"):
        result.close()
    reader.close()
    assert budget.used == 0
    result.table()
    assert budget.used == 0
    result.close()


def test_capacity_without_voltage_and_numpy_cycle_labels(tmp_path):
    """Typical labels extracted from a numpy/pandas column work without an energy dependency."""
    path = tmp_path / "current.parquet"
    pq.write_table(
        pa.table({"time_s": [0.0, 1.0, 2.0, 3.0], "current_mA": [1.0, 1.0, -1.0, -1.0]}), path
    )
    source = prepare_source(path, tmp_path / "prepared.parquet")
    result = source.cycles(np.array([1], dtype=np.int64)).capacity()
    assert result.table().cycle.tolist() == [1, 1]
    assert "energy_Wh" not in result.table()
    result.close()
    with pytest.raises(ValueError, match="positive integers"):
        source.cycles([True])


def test_half_cycle_cancellation_removes_owned_state(tmp_path):
    """Cancellation releases the global reservation and removes unpublished scientific state."""
    from gregorithm import CancellationToken, Cancelled

    source = prepared(tmp_path, range(4), [1, 1, -1, -1])
    budget = MemoryBudget(8 * 1024**2)
    token = CancellationToken()
    token.cancel()
    with pytest.raises(Cancelled):
        source.capacity(memory_budget=budget, cancellation=token)
    assert budget.used == 0
    assert not list((tmp_path / ".gregorithm-work").iterdir())


def test_energy_unavailable_is_reported_not_silently_zero(tmp_path):
    """Finite current with absent V retains Q, unavailable E and an explicit alert."""
    source = prepared(tmp_path, [0, 1], [1, 1], [np.nan, np.nan])
    with pytest.warns(IncompleteIntegrationWarning):
        result = source.capacity(include_energy=True)
    frame = result.table()
    assert frame.capacity_quality.tolist() == ["complete"]
    assert frame.energy_quality.tolist() == ["unavailable"]
    assert frame.energy_Wh.isna().all()
    result.close()
