"""Synthetic regressions for approved timestamp preparation, not experimental validation."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gregorithm import (
    CancellationToken,
    Cancelled,
    DuplicateTimestampWarning,
    MemoryBudget,
    ReadOptions,
    SchemaError,
    TimestampOrderError,
    prepare_source,
    prepare_sources,
)


def write_source(path, times):
    """Write distinct row payloads so duplicate removal must preserve the whole first row."""
    table = pa.table(
        {
            "time_s": times,
            "current_mA": [float(i + 1) for i in range(len(times))],
            "voltage_V": [float(i + 3) for i in range(len(times))],
            "operator_note": [f"original-{i}" for i in range(len(times))],
        }
    )
    pq.write_table(table, path, row_group_size=2)
    return table


@pytest.mark.parametrize("batch_rows", [1, 2, 7, 1024])
def test_first_complete_rows_and_nonconsecutive_repeats(tmp_path, batch_rows):
    """Retain identical original rows across decoder batches and Parquet row groups."""
    source = tmp_path / "measurement.parquet"
    original = write_source(source, [0.0, 1.0, 1.0, 2.0, 3.0, 0.0, 1.0, 4.0])
    budget = MemoryBudget(8 * 1024**2)
    updates = []
    with pytest.warns(DuplicateTimestampWarning) as messages:
        prepared = prepare_source(
            source,
            tmp_path / "prepared.parquet",
            read_options=ReadOptions(batch_rows=batch_rows),
            memory_budget=budget,
            on_progress=updates.append,
        )
    assert len(messages) == 1
    expected = original.take(pa.array([0, 1, 3, 4, 7]))
    assert pq.read_table(prepared.path).equals(expected)
    assert pq.read_table(source).equals(original)
    assert prepared.report["rows_read"] == 8
    assert prepared.report["rows_retained"] == 5
    assert prepared.report["duplicates_removed"] == 3
    assert prepared.report["generated_measurement_identifiers"] is False
    assert set(prepared.preview().columns) == set(original.column_names)
    assert prepared.preview(0).empty
    assert updates[-1].status == "complete"
    assert budget.used == 0
    sidecar = json.loads(Path(str(prepared.path) + ".manifest.json").read_text())
    assert sidecar == prepared.report
    assert not set(sidecar) & {"trace_id", "sample_id", "result_id", "selection_id"}


@pytest.mark.parametrize("batch_rows", [1, 2, 100])
def test_backward_time_fails_without_replacing_output(tmp_path, batch_rows):
    """A distinct backwards time fails, including at a batch/row-group boundary."""
    source = tmp_path / "bad.parquet"
    write_source(source, [0.0, 10.0, 20.0, 3.0])
    output = tmp_path / "prepared.parquet"
    output.write_bytes(b"existing-output")
    budget = MemoryBudget(8 * 1024**2)
    with pytest.raises(TimestampOrderError, match="20 s to 3 s"):
        prepare_source(
            source, output, read_options=ReadOptions(batch_rows=batch_rows), memory_budget=budget
        )
    assert output.read_bytes() == b"existing-output"
    assert budget.used == 0
    assert not list(tmp_path.glob("*.tmp"))
    assert not Path(str(output) + ".manifest.json").exists()


def test_original_column_projection_and_alias_time(tmp_path):
    """Validation time is internal when omitted from an explicit original-column projection."""
    source = tmp_path / "source.csv"
    source.write_text("seconds,note\n0,first\n1,keep\n1,remove\n2,last\n")
    with pytest.warns(DuplicateTimestampWarning):
        result = prepare_source(
            source,
            tmp_path / "projected.parquet",
            time_column="seconds",
            columns=["note"],
            read_options=ReadOptions(batch_rows=1),
        )
    assert result.preview().to_dict("list") == {"note": ["first", "keep", "last"]}


def test_same_names_rejected_before_any_analysis(tmp_path):
    """Case-insensitive basename collisions fail before reading sources or creating outputs."""
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="same basename"):
        prepare_sources([tmp_path / "a/File.parquet", tmp_path / "b/file.PARQUET"], output)
    assert not output.exists()


def test_multifile_lazy_handles_and_shared_budget(tmp_path):
    """A dictionary stores handles rather than dataframes; each source releases shared buffers."""
    paths = [tmp_path / "first.parquet", tmp_path / "second.parquet"]
    for path in paths:
        write_source(path, [0.0, 1.0])
    budget = MemoryBudget(8 * 1024**2)
    results = prepare_sources(paths, tmp_path / "prepared", memory_budget=budget)
    assert list(results) == [path.name for path in paths]
    assert all(handle.report["rows_retained"] == 2 for handle in results.values())
    assert budget.used == 0


@pytest.mark.parametrize("times", [[0.0, float("nan")], [0.0, float("inf")], [2**53 + 1]])
def test_invalid_timestamp_support(tmp_path, times):
    """Reject invalid or lossily representable time instead of deduplicating rounded values."""
    source = tmp_path / "invalid.parquet"
    write_source(source, times)
    with pytest.raises(SchemaError):
        prepare_source(source, tmp_path / "prepared.parquet")
    assert not (tmp_path / "prepared.parquet").exists()


def test_cancellation_releases_budget_and_keeps_source(tmp_path):
    """Cancellation cannot publish unfinished preparation or retain reserved buffers."""
    source = tmp_path / "source.parquet"
    original = write_source(source, [0.0, 1.0])
    token = CancellationToken()
    token.cancel()
    budget = MemoryBudget(8 * 1024**2)
    with pytest.raises(Cancelled):
        prepare_source(
            source, tmp_path / "prepared.parquet", cancellation=token, memory_budget=budget
        )
    assert budget.used == 0
    assert pq.read_table(source).equals(original)
    assert not (tmp_path / "prepared.parquet").exists()
