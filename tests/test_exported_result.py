"""Persistent redesigned-result reopening regressions."""

import json

import pandas as pd
import pytest
from test_half_cycles import prepared

from gregorithm import open_result


def test_export_reopens_without_source_or_generated_identifiers(tmp_path):
    """Restore a streamed capacity result and its settings without recalculation."""
    source = prepared(tmp_path, [0, 1, 2, 3], [1, 1, -1, -1])
    capacity = source.capacity()
    destination = tmp_path / "saved-capacity.parquet"
    try:
        expected = capacity.table()
        capacity.write_parquet(destination)
    finally:
        capacity.close()
    reopened = open_result(destination)
    pd.testing.assert_frame_equal(reopened.table(), expected)
    assert reopened.report["operation"] == "half_cycle_capacity"
    assert not {"sample_id", "trace_id", "result_id"} & set(reopened.table().columns)
    plotted = reopened.plot(
        x="half_cycle",
        y="capacity_mAh",
        series_column="direction",
        series_values=("charge", "discharge"),
    )
    assert plotted.status == "complete"


def test_missing_or_invalid_manifest_is_rejected(tmp_path):
    """Never reopen numerical values after silently losing calculation context."""
    source = prepared(tmp_path, [0, 1, 2], [1, 1, 1])
    destination = tmp_path / "totals.parquet"
    source.cumulative(include_energy=False).write_parquet(destination)
    manifest = destination.with_suffix(destination.suffix + ".manifest.json")
    manifest.unlink()
    with pytest.raises(ValueError, match="manifest"):
        open_result(destination)
    manifest.write_text(json.dumps({"wrong": True}), encoding="utf8")
    with pytest.raises(ValueError, match="operation"):
        open_result(destination)
