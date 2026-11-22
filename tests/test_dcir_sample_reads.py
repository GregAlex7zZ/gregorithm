"""Regression: DCIR reads only the requested elaborated rows, with unchanged values and order.

`_iter_m2_samples` used to convert a whole Parquet row group (up to 262,144 rows) to Python
objects for every pulse and for every single reference sample. Publishing a DCIR review measured
each event several times, so it took seconds per event while holding the GIL and froze the GUI's
progress animation. The range is now selected with Arrow compute first; these tests pin that the
selection is exactly the inclusive source-row range, in stored order, across row groups.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from gregorithm.dcir import _iter_m2_samples, _m2_sample


def _write_points(root: Path, rows: int, row_group_size: int) -> pa.Table:
    """Write a minimal `points/elaborated.parquet` with distinct, checkable values per row."""
    index = list(range(rows))
    table = pa.table(
        {
            "source_row": pa.array([i * 2 + 1 for i in index], pa.int64()),  # gaps: odd rows only
            "time_s": pa.array([i * 0.5 for i in index], pa.float64()),
            "current_a": pa.array([(-1.0) ** i * 1e-3 for i in index], pa.float64()),
            "voltage_v": pa.array([3.0 + i * 1e-4 for i in index], pa.float64()),
            "cycle": pa.array([1 + i // 1000 for i in index], pa.int64()),
            "half_cycle": pa.array([1 + i // 500 for i in index], pa.int64()),
            "step": pa.array([i % 3 for i in index], pa.int64()),
        }
    )
    (root / "points").mkdir(parents=True)
    pq.write_table(table, root / "points" / "elaborated.parquet", row_group_size=row_group_size)
    return table


@pytest.mark.parametrize(("first", "last"), [(1, 1), (999, 1203), (2047, 2049), (3999, 5000)])
def test_range_matches_full_scan_across_row_groups(tmp_path: Path, first: int, last: int) -> None:
    table = _write_points(tmp_path, rows=2000, row_group_size=700)
    expected = [row for row in table.to_pylist() if first <= row["source_row"] <= last]
    samples = list(_iter_m2_samples(tmp_path, first, last, "fp"))
    assert [s.source_row for s in samples] == [row["source_row"] for row in expected]
    assert [(s.time_s, s.current_a, s.voltage_v, s.cycle, s.half_cycle) for s in samples] == [
        (r["time_s"], r["current_a"], r["voltage_v"], r["cycle"], r["half_cycle"]) for r in expected
    ]
    assert all(s.elaboration_fingerprint == "fp" for s in samples)


def test_single_reference_row_and_missing_row(tmp_path: Path) -> None:
    _write_points(tmp_path, rows=2000, row_group_size=700)
    sample = _m2_sample(tmp_path, 1401, "fp")  # row index 700: first row of the second group
    assert (sample.source_row, sample.time_s) == (1401, 350.0)
    with pytest.raises(ValueError, match="missing or duplicated"):
        _m2_sample(tmp_path, 1400, "fp")  # even rows do not exist
