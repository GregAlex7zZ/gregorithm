"""Synthetic cumulative-coordinate regressions; source measurements remain unchanged."""

import numpy as np
import pandas as pd
import pytest
from test_half_cycles import prepared


@pytest.mark.parametrize("batch", [1, 3, 100])
def test_cumulative_final_values_match_totals_and_preserve_rows(tmp_path, batch):
    """Retain every original value and reach the independently calculated global totals."""
    source = prepared(
        tmp_path,
        [0, 1, 3, 6, 10],
        [2, 2, -1, -1, 3],
        [3.0, 3.1, 3.2, 3.4, 3.5],
        batch_rows=batch,
    )
    view = source.cumulative()
    frame = view.table()
    original = source.preview(100)
    pd.testing.assert_frame_equal(frame[original.columns], original)
    totals = source.totals()
    last = frame.iloc[-1]
    assert last.cumulative_charge_mAh == pytest.approx(totals.values["charge_mAh"])
    assert last.cumulative_discharge_mAh == pytest.approx(totals.values["discharge_mAh"])
    assert last.cumulative_net_mAh == pytest.approx(totals.values["net_mAh"])
    assert last.cumulative_charge_Wh == pytest.approx(totals.values["charge_Wh"])
    assert last.cumulative_discharge_Wh == pytest.approx(totals.values["discharge_Wh"])
    assert last.cumulative_net_Wh == pytest.approx(totals.values["net_Wh"])
    assert not {"sample_id", "trace_id", "result_id"} & set(frame.columns)


def test_missing_interval_breaks_display_and_preserves_original_origin(tmp_path):
    """Show an unavailable interval as a gap and resume with a partial global coordinate."""
    source = prepared(tmp_path, range(7), [1, 1, np.nan, np.nan, 1, 1, 1])
    frame = source.cumulative(include_energy=False).table()
    assert frame.cumulative_net_mAh.iloc[1] == pytest.approx(1 / 3600)
    assert frame.cumulative_net_mAh.iloc[2:5].isna().all()
    assert frame.cumulative_net_mAh.iloc[5] == pytest.approx(2 / 3600)
    assert frame.cumulative_capacity_quality.iloc[5] == "partial"
    assert frame.cumulative_energy_quality.unique().tolist() == ["unavailable"]


def test_projection_preview_export_and_collision(tmp_path):
    """Support explicit projections and protect source/output column names."""
    source = prepared(tmp_path, range(5), [1] * 5)
    view = source.cumulative()
    projected = list(view.iter_batches(columns=["time_s", "cumulative_net_mAh"]))
    assert projected[0].schema.names == ["time_s", "cumulative_net_mAh"]
    assert len(view.preview(2)) == 2
    destination = tmp_path / "cumulative.parquet"
    view.write_parquet(destination, columns=["time_s", "cumulative_net_mAh"])
    assert destination.is_file()
    with pytest.raises(FileExistsError):
        view.write_parquet(destination)
    with pytest.raises(ValueError, match="Unknown"):
        list(view.iter_batches(columns=["missing"]))


def test_isolated_estimate_quality_without_replacing_raw_nan(tmp_path):
    """Use the approved estimate for Q while retaining the original missing current."""
    source = prepared(tmp_path, [0, 1, 2, 3], [1, np.nan, 1, 1])
    frame = source.cumulative(include_energy=False).table()
    assert np.isnan(frame.current_mA.iloc[1])
    assert frame.cumulative_capacity_quality.iloc[-1] == "estimated"
    assert frame.cumulative_net_mAh.iloc[-1] == pytest.approx(3 / 3600)
