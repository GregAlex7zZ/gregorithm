"""Regression: buffering the elaboration scratch files changes no published byte.

The S4/S5 stages used to write one Parquet row group per classified region (thousands per real
file), which dominated the elaboration time. `_BufferedStagingSink` now queues those tables and
writes large row groups. These tests elaborate a pulse-rich synthetic file with the former
unbuffered sink and with the buffered one and require the same fingerprint and byte-identical
published points and indexes, while the scratch files receive far fewer writes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import gregorithm
import gregorithm.elaboration as elaboration


def _write_source(path: Path) -> None:
    """Rests and short charge/discharge pulses: many classified regions, a few hundred rows."""
    lines = ["time_s,current_mA,voltage_V"]
    t = 0.0
    for pulse in range(60):
        current = (1.0, -1.0, 0.5)[pulse % 3]
        for _ in range(4):
            lines.append(f"{t},0,{3.6 + 0.001 * pulse:.6f}")
            t += 1.0
        for k in range(5):
            lines.append(f"{t},{current},{3.6 + 0.001 * pulse + 0.002 * k * current:.6f}")
            t += 1.0
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _elaborate(
    root: Path, source: Path, order: gregorithm.CycleOrder
) -> tuple[str, dict[str, str]]:
    """Import and elaborate ``source``; return the fingerprint and sha256 of published data."""
    inspection = gregorithm.inspect_source(source)
    imported = gregorithm.import_source(
        gregorithm.ImportRequest(
            source=source,
            caller_root=root,
            sample_id="s",
            job_id="j",
            generation_id="g",
            mapping=inspection.suggested_mapping,
        )
    )
    result = gregorithm.elaborate(
        gregorithm.ElaborationRequest(
            canonical_parquet=imported.ref.path / "points" / "canonical.parquet",
            canonical_content_hash=imported.report.content_hash,
            caller_root=root,
            sample_id="e",
            job_id="j2",
            generation_id="g2",
            parameters=gregorithm.ElaborationParameters(cycle_order=order),
            batch_size=64,
        )
    )
    files = {
        p.relative_to(result.ref.path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(result.ref.path.rglob("*.parquet"))
    }
    return result.fingerprint, files


@pytest.mark.parametrize(
    "order", [gregorithm.CycleOrder.CHARGE_DISCHARGE, gregorithm.CycleOrder.DISCHARGE_CHARGE]
)
def test_buffered_scratch_writes_publish_identical_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, order: gregorithm.CycleOrder
) -> None:
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    source = tmp_path / "pulses.csv"
    _write_source(source)
    scratch = {"provisional.parquet", "half-meta.parquet"}
    writes = {"count": 0}
    original_write = pq.ParquetWriter.write_table

    def counted(self: pq.ParquetWriter, table, *args, **kwargs) -> None:
        if Path(str(self.where)).name in scratch:
            writes["count"] += 1
        original_write(self, table, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetWriter, "write_table", counted)
    buffered = _elaborate(tmp_path / "buffered", source, order)
    buffered_writes = writes["count"]

    writes["count"] = 0
    monkeypatch.setattr(elaboration, "_BufferedStagingSink", elaboration._BoundedParquetSink)
    unbuffered = _elaborate(tmp_path / "unbuffered", source, order)
    unbuffered_writes = writes["count"]

    assert buffered[0] == unbuffered[0]
    assert buffered[1] == unbuffered[1]
    assert {"points/elaborated.parquet", "indexes/segments.parquet"} <= set(buffered[1])
    assert unbuffered_writes > 50
    assert buffered_writes * 5 < unbuffered_writes


def test_buffered_sink_writes_rows_in_order_and_flushes_on_close(tmp_path: Path) -> None:
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    schema = pa.schema([("x", pa.int64())])
    sink = elaboration._BufferedStagingSink(tmp_path / "s.parquet", schema, row_limit=10)
    for start in range(0, 25, 3):
        sink.write_table(pa.table({"x": list(range(start, min(start + 3, 25)))}, schema=schema))
    sink.close()
    stored = pq.ParquetFile(tmp_path / "s.parquet")
    assert pq.read_table(tmp_path / "s.parquet")["x"].to_pylist() == list(range(25))
    assert stored.num_row_groups < 9  # 9 tables queued, fewer row groups written
