"""Compact source/row-group inventory, without materializing the input."""

import pyarrow as pa
import pyarrow.parquet as pq


def batches(trace, rows, result_id):
    """Yield bounded rows from the indexed segment/source table, retaining explicit scientific identity.

    Parameters
    ----------
    trace
        BatteryTrace source handle; raw signals are never rewritten.
    rows
        Bounded operation-specific value described by the function contract and API reference.
    result_id
        Bounded operation-specific value described by the function contract and API reference."""
    base = {
        "trace_id": trace.trace_id,
        "source_revision": trace.source_revision,
        "result_id": result_id,
    }
    if trace.path.suffix.lower() != ".parquet":
        yield pa.RecordBatch.from_pylist(
            [
                {
                    **base,
                    "row_group": None,
                    "sample_start": 0,
                    "sample_end": rows - 1,
                    "rows": rows,
                    "time_min": None,
                    "time_max": None,
                    "access": "sequential_text",
                }
            ]
        )
        return
    f = pq.ParquetFile(trace.path)
    try:
        original = next(
            (
                n
                for n in f.schema_arrow.names
                if trace.read_options.column_mapping.get(n, n) == "time_s"
            ),
            "time_s",
        )
        column = f.schema_arrow.get_field_index(original)
        offset = 0
        records = []
        for g in range(f.num_row_groups):
            group = f.metadata.row_group(g)
            st = group.column(column).statistics
            records.append(
                {
                    **base,
                    "row_group": g,
                    "sample_start": offset,
                    "sample_end": offset + group.num_rows - 1,
                    "rows": group.num_rows,
                    "time_min": float(st.min) if st and st.has_min_max else None,
                    "time_max": float(st.max) if st and st.has_min_max else None,
                    "access": "parquet_row_group",
                }
            )
            offset += group.num_rows
            if len(records) == 4096:
                yield pa.RecordBatch.from_pylist(records)
                records = []
        if records:
            yield pa.RecordBatch.from_pylist(records)
    finally:
        f.close()
