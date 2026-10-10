"""Synthetic DCIR formula/support tests; values are not experimental validation."""

import numpy as np
import pytest
from test_half_cycles import prepared

from gregorithm import ManualPulse, MemoryBudget


def pulse_source(tmp_path, batch=100):
    """Create rest-charge-rest-discharge-rest point signals with irregular cadence."""
    return prepared(
        tmp_path,
        [0, 1, 2, 3.1, 5, 6, 7, 8, 10.2, 11],
        [0, 0, 100, 100, 100, 0, 0, -200, -200, 0],
        [3.0, 3.0, 3.02, 3.03, 3.04, 3.04, 3.1, 3.08, 3.06, 3.06],
        batch_rows=batch,
    )


@pytest.mark.parametrize("batch", [1, 3, 100])
def test_exact_and_excess_real_points_with_signed_units(tmp_path, batch):
    """Use V/A exactly once and preserve signed resistance for both pulse directions."""
    source = pulse_source(tmp_path, batch)
    events = source.events()
    result = events.dcir(2.1)
    try:
        frame = result.table()
        charge, discharge = frame.iloc[0], frame.iloc[1]
        assert charge.sampling_choice == "exact"
        assert charge.actual_delay_s == pytest.approx(2.1)
        assert charge.delta_current_A == pytest.approx(0.1)
        assert charge.resistance_ohm == pytest.approx(0.3)
        assert discharge.sampling_choice == "excess"
        assert discharge.actual_time_s == pytest.approx(10.2)
        assert discharge.timing_error_s == pytest.approx(1.1)
        assert discharge.delta_current_A == pytest.approx(-0.2)
        assert discharge.resistance_ohm == pytest.approx(0.2)
        assert frame.valid.tolist() == [True, True]
    finally:
        result.close()
        events.close()


def test_defect_end_boundary_timing_limit_and_outside_rest(tmp_path):
    """Allow defect before t3, enforce optional timing error, and reject target at t3."""
    source = pulse_source(tmp_path)
    events = source.events().statuses("supported")
    defect = events.dcir(4.5)
    try:
        row = defect.table().iloc[0]
        assert row.valid
        assert row.sampling_choice == "defect"
        assert row.actual_time_s == 5
        assert row.timing_error_s == pytest.approx(-0.5)
    finally:
        defect.close()
    limited = events.dcir(4.5, max_timing_error_s=0.4)
    try:
        row = limited.table().iloc[0]
        assert not row.valid
        assert "target-time error" in row.reason
    finally:
        limited.close()
    outside = events.dcir(5)
    try:
        assert not outside.table().iloc[0].valid
        assert "following rest" in outside.table().iloc[0].reason
    finally:
        outside.close()
        events.close()


def test_incomplete_pulse_supported_target_but_no_future_defect(tmp_path):
    """Measure an observed target in a terminal pulse but never extrapolate beyond it."""
    source = prepared(
        tmp_path,
        [0, 1, 2, 4, 6],
        [0, 0, -100, -100, -100],
        [3.0, 3.0, 2.9, 2.8, 2.7],
    )
    events = source.events()
    observed = events.dcir(2)
    try:
        row = observed.table().iloc[0]
        assert row.valid and row.quality == "incomplete_event"
        assert row.sampling_choice == "excess" and row.actual_time_s == 4
    finally:
        observed.close()
    beyond = events.dcir(7)
    try:
        row = beyond.table().iloc[0]
        assert not row.valid
        assert "truncated" in row.reason
    finally:
        beyond.close()
        events.close()


def test_manual_exact_boundaries_missing_values_and_current_guard(tmp_path):
    """Reject absent exact boundaries, real-value gaps and sub-threshold delta current."""
    source = prepared(
        tmp_path,
        [0, 1, 2, 3, 4],
        [0, 0, 0.0005, 0.0005, 0],
        [3, 3, 3.1, np.nan, 3.1],
    )
    guarded = source.dcir(1, manual=[ManualPulse(1, 2, 3, 4)])
    try:
        row = guarded.table().iloc[0]
        assert not row.valid
        assert "current difference" in row.reason
    finally:
        guarded.close()
    missing = source.dcir(1, manual=[ManualPulse(0.5, 2, 3, 4)])
    try:
        assert "exact retained timestamp" in missing.table().iloc[0].reason
    finally:
        missing.close()


def test_one_delay_validation_budget_and_export(tmp_path):
    """Reject delay sequences, release shared budget and protect an existing export."""
    source = pulse_source(tmp_path)
    budget = MemoryBudget(8 * 1024**2)
    result = source.dcir(2, memory_budget=budget)
    assert budget.used == 0
    try:
        destination = tmp_path / "dcir.parquet"
        result.write_parquet(destination)
        with pytest.raises(FileExistsError):
            result.write_parquet(destination)
    finally:
        result.close()
    assert budget.used == 0
    with pytest.raises(ValueError, match="one finite scalar"):
        source.dcir([1, 2])
    with pytest.raises(ValueError, match="positive"):
        source.dcir(0)


def test_consecutive_cycle_scope_keeps_all_opportunities(tmp_path):
    """A cycle range narrows the domain but returns every onset inside it."""
    source = prepared(
        tmp_path,
        range(24),
        [0, 0] + [100] * 4 + [0, 0] + [-100] * 4 + [0, 0] + [100] * 4 + [0, 0] + [-100] * 4,
        [3.0 + 0.01 * k for k in range(24)],
    )
    result = source.dcir(1, cycles=[2])
    try:
        frame = result.table()
        assert len(frame) == 2
        assert frame.direction.tolist() == ["charge", "discharge"]
    finally:
        result.close()
    with pytest.raises(ValueError, match="consecutive"):
        source.dcir(1, cycles=[1, 3])
