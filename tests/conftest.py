"""Synthetic regression tests; run with tools/run_tests.py. Passing verifies software behavior, not experimental electrochemistry."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gregorithm import open_trace


@pytest.fixture
def make_trace(tmp_path):
    """Provide the pytest synthetic-trace factory, writing point-signal arrays and extra columns to small Parquet row groups in the test temporary directory."""

    def make(t, i, v=None, **extra):
        """Write one synthetic trace with standardized seconds/mA/V signals and return open_trace(); no real data are used."""
        p = tmp_path / (str(len(list(tmp_path.glob("*.parquet")))) + ".parquet")
        data = {
            "time_s": np.array(t, dtype=float),
            "current_mA": np.array(i, dtype=float),
            "voltage_V": np.array(v if v is not None else np.full(len(t), 4.0), dtype=float),
            **extra,
        }
        pq.write_table(pa.table(data), p, row_group_size=7)
        return open_trace(p)

    return make
