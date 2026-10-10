"""Optional bounded notebook-input adapters; pandas is never imported by the core."""

import tempfile
import uuid
from pathlib import Path

import pyarrow as pa

from .diagnostics import MaterializationLimitError
from .io import atomic_parquet
from .model import open_trace


def from_pandas(
    frame,
    *,
    directory=None,
    metadata=None,
    read_options=None,
    max_rows=1_000_000,
    max_bytes=512 * 1024**2,
):
    """Persist a small pandas DataFrame as a trace without retaining it in the handle.

    Parameters
    ----------
    frame : pandas.DataFrame
        Standardized time_s/current_mA/voltage_V columns, or aliases supplied
        through read_options. Extra columns are preserved; the pandas index is
        not a signal column. Pandas is an optional dependency.
    directory : path-like, optional
        Directory for the persistent input copy. If omitted, create a temporary
        directory; callers should remove it after all trace/results handles are
        no longer needed. The source is not deleted automatically.
    metadata : dict, optional
        Cell/protocol annotations passed to open_trace.
    read_options : ReadOptions, optional
        Column aliases and reader configuration for the persisted source.
    max_rows, max_bytes : int
        Explicit input limits. A conservative four-copy estimate is checked
        before Arrow conversion; exceeding either limit raises
        MaterializationLimitError without truncating rows.

    Returns
    -------
    BatteryTrace
        Disk-backed input usable with analyze(), select(), and iter_batches().
        Example: trace = from_pandas(df, directory="notebook_inputs").
    """
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("A pandas DataFrame is required")
    if max_rows < 0 or max_bytes < 1:
        raise ValueError("Nonnegative row and positive byte limits are required")
    estimate = int(frame.memory_usage(index=True, deep=True).sum()) * 4
    if len(frame) > max_rows or estimate > max_bytes:
        raise MaterializationLimitError("Input exceeds bounded pandas adapter limits")
    folder = (
        Path(directory) if directory is not None else Path(tempfile.mkdtemp(prefix="gregorithm-"))
    )
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ("pandas-" + uuid.uuid4().hex + ".parquet")

    def batches():
        """Convert bounded slices while retaining only one Arrow batch at a time."""
        if not len(frame):
            yield pa.RecordBatch.from_pandas(frame, preserve_index=False)
        for start in range(0, len(frame), 32768):
            yield pa.RecordBatch.from_pandas(
                frame.iloc[start : start + 32768], preserve_index=False
            )

    atomic_parquet(batches(), path, {"input_adapter": "pandas", "index_preserved": False})
    return open_trace(path, metadata=metadata, read_options=read_options)
