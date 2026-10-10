"""Synthetic fixed-reference C-rate tests; not experimental cell calibration."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_half_cycles import prepared

from gregorithm import HalfCycleConfig, MemoryBudget, ReadOptions, UnavailableReferenceWarning


@pytest.mark.parametrize("batch", [1, 2, 100])
def test_manual_signed_current_estimates_and_column_preservation(tmp_path, batch):
    """Estimate one missing point across batches, retain raw NaN and original payload."""
    source = prepared(tmp_path, [0, 1, 2, 3, 4], [2, np.nan, 4, -2, -2], batch_rows=batch)
    rate = source.c_rate(reference_mAh=2)
    with pytest.warns(UserWarning, match="estimates"):
        frame = rate.table()
    original = source.preview(5)
    pd.testing.assert_frame_equal(frame[original.columns], original)
    np.testing.assert_allclose(frame.c_rate_h_inv, [1, 1.5, 2, -1, -1])
    assert not set(frame.columns) & {"trace_id", "sample_id", "reference_mAh"}
    assert rate.reference["capacity_mAh"] == 2


def test_cycle_two_reference_is_closing_half_not_fourth_half(tmp_path):
    """An initial incomplete discharge changes half numbering, not cycle-2 reference."""
    source = prepared(tmp_path, range(10), [-1, -1, 2, 2, -3, -3, 2, 2, -4, -4])
    capacity = source.capacity()
    try:
        rate = source.c_rate(capacity=capacity.cycles([3]))
        assert rate.reference["cycle"] == 2
        assert rate.reference["half_cycle"] == 3
        expected = (
            capacity.cycles([2]).table().query("direction == 'discharge'").capacity_mAh.iloc[0]
        )
        assert rate.reference["capacity_mAh"] == expected
        np.testing.assert_allclose(
            rate.preview().c_rate_h_inv,
            [-1 / expected, -1 / expected, 2 / expected, 2 / expected, -3 / expected],
        )
        custom = source.c_rate(capacity=capacity, reference_half_cycle=2)
        assert custom.reference["direction"] == "charge"
    finally:
        capacity.close()
    # Reference has been resolved to a scalar; reading no longer needs summary storage.
    assert rate.preview(1).shape[0] == 1


def test_reverse_cycle_order_reference_and_custom_cycle(tmp_path):
    """Discharge-first pairs use cycle-2 net charge; explicit cycle selection is readable."""
    source = prepared(tmp_path, range(8), [-1, -1, 2, 2, -3, -3, 4, 4])
    capacity = source.capacity(config=HalfCycleConfig(cycle_order="discharge_then_charge"))
    try:
        rate = source.c_rate(capacity=capacity)
        assert rate.reference["direction"] == "charge"
        assert rate.reference["half_cycle"] == 4
        custom = source.c_rate(capacity=capacity, reference_cycle=1)
        assert custom.reference["half_cycle"] == 2
        assert custom.reference["capacity_mAh"] != rate.reference["capacity_mAh"]
    finally:
        capacity.close()


def test_missing_reference_alert_no_fallback_and_invalid_choices(tmp_path):
    """Missing cycle 2 produces null C-rate, rather than selecting the only observed half."""
    source = prepared(tmp_path, [0, 1], [1, 1])
    with pytest.warns(UnavailableReferenceWarning, match="no fallback"):
        rate = source.c_rate()
    assert rate.table().c_rate_h_inv.isna().all()
    assert rate.reference["reason"] == "missing_reference"
    for settings in (
        {"reference_mAh": 0},
        {"reference_mAh": np.nan},
        {"reference_mAh": True},
        {"reference_cycle": False},
        {"reference_half_cycle": 1.5},
        {"reference_mAh": 1, "reference_cycle": 3},
        {"reference_half_cycle": 2, "reference_cycle": 3},
    ):
        with pytest.raises(ValueError):
            source.c_rate(**settings)


def test_partial_reference_cannot_yield_c_rate(tmp_path):
    """An unsupported interval in the requested reference remains unavailable, not a subtotal denominator."""
    source = prepared(tmp_path, range(11), [1, 1, -1, -1, 1, 1, -1, -1, np.nan, np.nan, -1])
    with pytest.warns(UserWarning):
        capacity = source.capacity()
    try:
        with pytest.warns(UnavailableReferenceWarning):
            rate = source.c_rate(capacity=capacity)
        assert rate.reference["reason"] == "incomplete_reference_support"
        assert rate.preview().c_rate_h_inv.isna().all()
    finally:
        capacity.close()


def test_time_weighted_mean_uses_actual_cadence_and_valid_support(tmp_path):
    """Use 145 mA*s / 100 s, not the arithmetic average of three samples."""
    source = prepared(tmp_path, [0, 10, 100], [1, 1, 2])
    row = source.c_rate(reference_mAh=2).mean().table().iloc[0]
    assert row.mean_current_mA == pytest.approx(1.45)
    assert row.mean_c_rate_h_inv == pytest.approx(0.725)
    assert row.covered_s == 100
    assert row.current_quality == "complete"
    gap = tmp_path / "gap"
    gap.mkdir()
    source = prepared(gap, range(5), [1, 1, np.nan, np.nan, 1])
    with pytest.warns(UserWarning):
        row = source.c_rate(reference_mAh=2).mean().table().iloc[0]
    assert row.mean_current_mA == 1
    assert row.covered_s == 1
    assert row.current_quality == row.c_rate_quality == "partial"


def test_estimated_and_nonpositive_automatic_references(tmp_path):
    """Propagate eligible reference estimation, and reject negative net capacity explicitly."""
    first = tmp_path / "estimated"
    second = tmp_path / "negative"
    first.mkdir()
    second.mkdir()
    source = prepared(first, range(5), [1, 1, np.nan, 1, 1])
    with pytest.warns(UserWarning):
        capacity = source.capacity()
    try:
        rate = source.c_rate(capacity=capacity, reference_half_cycle=1)
        assert rate.reference["quality"] == "estimated"
        with pytest.warns(UserWarning):
            assert rate.mean().values["c_rate_quality"] == "estimated"
    finally:
        capacity.close()
    source = prepared(second, range(7), [1, 1, -100, 1, 1, -1, -1])
    capacity = source.capacity()
    try:
        with pytest.warns(UnavailableReferenceWarning):
            rate = source.c_rate(capacity=capacity, reference_half_cycle=1)
        assert rate.reference["reason"] == "nonpositive_or_unavailable_reference"
    finally:
        capacity.close()


def test_missing_point_bracket_limit_and_real_long_gap(tmp_path):
    """The ten-second limit applies only to estimates, never to finite real measurements."""
    source = prepared(tmp_path, [0, 1, 11, 100], [1, np.nan, 2, 3], batch_rows=1)
    frame = source.c_rate(reference_mAh=1).table()
    assert np.isnan(frame.c_rate_h_inv.iloc[1])
    assert frame.c_rate_h_inv.iloc[2:].tolist() == [2, 3]


def test_bounded_projection_exports_empty_source_and_shared_budget(tmp_path):
    """Exports preserve source columns by default and obey explicit projection/row limits."""
    from gregorithm import MaterializationLimitError

    source = prepared(tmp_path, range(4), [1, 1, -1, -1])
    budget = MemoryBudget(8 * 1024**2)
    rate = source.c_rate(reference_mAh=1, memory_budget=budget)
    with pytest.raises(MaterializationLimitError):
        rate.table(max_rows=2)
    assert budget.used == 0
    iterator = rate.iter_batches(columns=["note"])
    block = next(iterator)
    assert block.schema.names == ["note", "c_rate_h_inv"]
    assert budget.used > 0
    iterator.close()
    assert budget.used == 0
    path = tmp_path / "rates.parquet"
    rate.write_parquet(path)
    assert pq.read_table(path).column_names == source.preview().columns.tolist() + ["c_rate_h_inv"]
    with pytest.raises(ValueError, match="overwrite"):
        rate.write_parquet(source.path)
    with pytest.raises(FileExistsError):
        rate.write_parquet(path)
    assert rate.preview(0).empty
    empty = tmp_path / "empty"
    empty.mkdir()
    # Preparation rejects empty acquisitions; exercise a manually constructed empty handle.
    empty_source = replace(source, path=empty / "empty.parquet")
    pq.write_table(pa.Table.from_batches([], schema=pq.read_schema(source.path)), empty_source.path)
    view = empty_source.c_rate(reference_mAh=1)
    assert view.table().columns.tolist() == [
        "time_s",
        "current_mA",
        "voltage_V",
        "note",
        "c_rate_h_inv",
    ]


def test_reference_source_change_config_conflict_and_cancellation(tmp_path):
    """Reuse only an unchanged matching source and release budget after cancellation."""
    from gregorithm import CancellationToken, Cancelled, SourceChangedError

    source = prepared(tmp_path, range(8), [1, 1, -1, -1, 1, 1, -1, -1])
    capacity = source.capacity()
    try:
        with pytest.raises(ValueError, match="configuration"):
            source.c_rate(capacity=capacity, config=HalfCycleConfig(missing_span_s=5))
        fake = replace(source, path=tmp_path / "other.parquet")
        pq.write_table(pa.table({"time_s": [0.0], "current_mA": [1.0]}), fake.path)
        with pytest.raises(SourceChangedError):
            fake.c_rate(capacity=capacity)
    finally:
        capacity.close()
    token = CancellationToken()
    budget = MemoryBudget(8 * 1024**2)
    rate = source.c_rate(reference_mAh=1, memory_budget=budget, cancellation=token)
    token.cancel()
    with pytest.raises(Cancelled):
        rate.preview()
    assert budget.used == 0
    source = replace(source, read_options=ReadOptions(batch_rows=1))
    rate = source.c_rate(reference_mAh=1)
    pq.write_table(pa.table({"time_s": [0.0], "current_mA": [2.0]}), source.path)
    with pytest.raises(SourceChangedError):
        rate.preview()
