"""Fresh-process checks that the new default path does not initialize legacy identities."""

import subprocess
import sys


def test_new_calculations_do_not_load_legacy_identity_modules(tmp_path):
    """Run a complete tiny preparation/global calculation in an isolated fresh interpreter."""
    script = """
import sys
from pathlib import Path
import gregorithm
assert not any(name.startswith("gregorithm.") for name in sys.modules)
from gregorithm import prepare_source
assert "gregorithm.provenance" not in sys.modules
import pyarrow as pa
import pyarrow.parquet as pq
pq.write_table(pa.table({"time_s":[0.,3600.], "current_mA":[1.,1.], "voltage_V":[4.,4.]}), "source.parquet")
data = prepare_source("source.parquet", "prepared.parquet")
result = data.totals()
assert result.values["charge_mAh"] == 1.
assert result.values["charge_Wh"] == .004
half_cycles = data.capacity(include_energy=True)
half_rows = list(half_cycles.iter_batches())
assert half_rows[0].column("capacity_mAh")[0].as_py() == 1.
summary = list(half_cycles.summary().iter_batches())
assert summary[0].column("coulombic_efficiency_reason").to_pylist() == ["missing_counterpart"]
normalized = list(half_cycles.normalize(mass_g=.5).iter_batches())
assert normalized[0].column("capacity_mAh_g")[0].as_py() == 2.
rate = data.c_rate(reference_mAh=1.)
rate_rows = list(rate.iter_batches())
assert rate_rows[0].column("c_rate_h_inv").to_pylist() == [1.,1.]
assert rate.mean().values["mean_current_mA"] == 1.
assert "gregorithm.step_classification" not in sys.modules
half_cycles.close()
for name in ("gregorithm.provenance", "gregorithm.model", "gregorithm.analysis", "plotly", "ipywidgets"):
    assert name not in sys.modules, name
assert set(gregorithm.__all__).issubset(dir(gregorithm))
"""
    subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, check=True, capture_output=True, text=True
    )
