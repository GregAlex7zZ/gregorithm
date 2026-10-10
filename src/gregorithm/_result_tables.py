"""Shared guarded pandas materialization for bounded result and measurement views."""

import pyarrow as pa

from .diagnostics import MaterializationLimitError


def bounded_frame(view, rows, *, preview=False):
    """Collect bounded batches under the view's shared budget; preview explicitly truncates.

    Views must emit a typed empty batch for empty selections. Caller-owned pandas
    memory is released from managed accounting after return; it is not a RAM limit.
    """
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
        raise ValueError("Nonnegative row limit required")
    allowance = getattr(view, "_materialize_bytes", min(64 * 1024**2, view._budget.limit // 4))
    view._budget.reserve(allowance)
    reader, blocks, count, size = view.iter_batches(), [], 0, 0
    try:
        for block in reader:
            if preview:
                block = block.slice(0, max(0, rows - count))
            count += block.num_rows
            size += block.nbytes
            if count > rows or size * 3 > allowance:
                raise MaterializationLimitError(
                    "Table exceeds materialization limits; use iter_batches()"
                )
            if block.num_rows:
                if blocks and not blocks[0].num_rows:
                    blocks.clear()
                blocks.append(block)
            elif not blocks:
                blocks.append(block)
            if preview and count >= rows:
                break
        return pa.Table.from_batches(blocks).to_pandas()
    finally:
        reader.close()
        view._budget.release(allowance)
