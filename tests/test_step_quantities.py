"""Synthetic step Q/E accounting tests; no controller-mode truth is implied."""

import numpy as np
import pytest
from test_half_cycles import prepared


@pytest.mark.parametrize("batch", [1, 3, 100])
def test_step_quantities_conserve_source_and_are_batch_invariant(tmp_path, batch):
    """Sum every directional contribution exactly once under variable batch boundaries."""
    source = prepared(
        tmp_path,
        range(12),
        [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0],
        [3.0 + 0.1 * k for k in range(12)],
        batch_rows=batch,
    )
    steps = source.steps()
    quantities = steps.quantities()
    try:
        frame = quantities.table()
        totals = source.totals()
        assert frame.step_type.tolist() == ["rest", "CC", "CC", "rest"]
        assert frame.duration_s.sum() == pytest.approx(11)
        assert frame.charge_mAh.sum() == pytest.approx(totals.values["charge_mAh"])
        assert frame.discharge_mAh.sum() == pytest.approx(totals.values["discharge_mAh"])
        assert frame.charge_Wh.sum() == pytest.approx(totals.values["charge_Wh"])
        assert frame.discharge_Wh.sum() == pytest.approx(totals.values["discharge_Wh"])
        assert not {"sample_id", "trace_id", "result_id"} & set(frame.columns)
    finally:
        quantities.close()
        steps.close()


def test_transition_interval_belongs_to_preceding_step(tmp_path):
    """Assign the complete interval ending at a new step to the preceding step."""
    source = prepared(tmp_path, range(10), [1] * 5 + [2] * 5, [4] * 10)
    steps = source.steps()
    quantities = steps.quantities()
    try:
        frame = quantities.table()
        assert frame.start_s.tolist() == [0, 4, 8]
        assert frame.duration_s.tolist() == [4, 4, 1]
        assert frame.capacity_mAh.iloc[0] == pytest.approx(4 / 3600)
        assert frame.capacity_mAh.iloc[1] == pytest.approx(7.5 / 3600)
        assert frame.capacity_mAh.iloc[2] == pytest.approx(2 / 3600)
    finally:
        quantities.close()
        steps.close()


def test_discharge_orientation_and_missing_quality(tmp_path):
    """Report positive-oriented discharge and independent capacity/energy quality."""
    source = prepared(
        tmp_path,
        range(7),
        [-2, -2, np.nan, -2, -2, -2, -2],
        [4, 4, 4, np.nan, np.nan, 4, 4],
    )
    steps = source.steps()
    with pytest.warns(UserWarning):
        quantities = steps.quantities()
    try:
        row = quantities.table().iloc[0]
        assert row.capacity_mAh > 0
        assert row.net_mAh < 0
        assert row.capacity_quality == "estimated"
        assert row.energy_quality == "partial"
    finally:
        quantities.close()
        steps.close()


def test_capacity_only_and_lifecycle(tmp_path):
    """Permit capacity-only calculation and protect temporary storage from active readers."""
    source = prepared(tmp_path, range(6), [1] * 6)
    steps = source.steps()
    result = steps.quantities(include_energy=False)
    try:
        frame = result.table()
        assert frame.energy_quality.tolist() == ["unavailable"]
        assert frame.energy_Wh.isna().all()
        iterator = result.iter_batches(1)
        next(iterator)
        with pytest.raises(ValueError, match="iterators"):
            result.close()
        iterator.close()
    finally:
        result.close()
        steps.close()
