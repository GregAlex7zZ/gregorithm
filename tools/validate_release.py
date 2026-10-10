"""Build, inspect, and install local review distributions in an isolated environment.

Run after the final reports are written. Dependency wheels must already exist in
.cache/release-wheelhouse. Everything stays inside the authorized workspace;
this script does not publish a package or distribute a real dataset.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
CACHE = ROOT / ".cache/release-validation"
CACHE.mkdir(parents=True, exist_ok=True)
ENV = ROOT / ".cache/release-venv"
BUILD_ENV = {**os.environ, "SOURCE_DATE_EPOCH": "1791331200"}


def run(arguments, *, cwd=ROOT, env=None):
    """Execute one reproducible local check and fail immediately on a nonzero exit."""
    completed = subprocess.run(
        arguments, cwd=cwd, env=env or BUILD_ENV, capture_output=True, text=True
    )
    if completed.returncode:
        detail = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
        raise RuntimeError(f"Release command failed ({completed.returncode}):\n{detail}")
    return completed.stdout


def digest(path):
    """Compute the complete artifact SHA-256 in bounded chunks."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def inspect_entries(names):
    """Reject private/cache/dataset assets and unsafe archive member paths."""
    forbidden = (".venv", ".cache", ".demo", ".test-runs", "__pycache__", "benchmark-results")
    for name in names:
        parts = PurePosixPath(name).parts
        if ".." in parts or PurePosixPath(name).is_absolute():
            raise ValueError("Unsafe archive member: " + name)
        if any(part in forbidden for part in parts):
            raise ValueError("Private/cache member included: " + name)
        if name.endswith((".parquet", ".csv", ".executed.ipynb", ".pyc", ".nbc", ".nbi")):
            raise ValueError("Dataset/runtime output included: " + name)


def main():
    """Check local wheel/sdist, rebuild equality, and genuine isolated core usage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--private-input",
        action="append",
        type=Path,
        default=[],
        help="Reject this owner's input basename/stem in actual release payloads",
    )
    args = parser.parse_args()
    private_tokens = tuple(
        token.casefold() for path in args.private_input for token in (path.name, path.stem) if token
    )
    print("Building local wheel and source distribution", flush=True)
    run([sys.executable, "-m", "build", "--no-isolation"])
    wheel = DIST / "gregorithm-0.1.0-py3-none-any.whl"
    source = DIST / "gregorithm-0.1.0.tar.gz"
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = archive.namelist()
        inspect_entries(wheel_names)
        for name in wheel_names:
            payload = archive.read(name).decode("utf8", errors="ignore").casefold()
            if any(token in payload or token in name.casefold() for token in private_tokens):
                raise ValueError("Private input reference found in wheel payload")
    with tarfile.open(source) as archive:
        source_names = archive.getnames()
        inspect_entries(source_names)
        for member in archive.getmembers():
            if member.isfile():
                payload = (
                    archive.extractfile(member).read().decode("utf8", errors="ignore").casefold()
                )
                if any(
                    token in payload or token in member.name.casefold() for token in private_tokens
                ):
                    raise ValueError("Private input reference found in source archive payload")
        for member in archive.getmembers():
            if member.name.endswith(".ipynb"):
                notebook = json.loads(archive.extractfile(member).read())
                if any(
                    cell.get("outputs") or cell.get("execution_count") is not None
                    for cell in notebook["cells"]
                    if cell["cell_type"] == "code"
                ):
                    raise ValueError("Executed notebook output included in source distribution")
    extraction = Path(tempfile.mkdtemp(dir=CACHE, prefix="source-"))
    with tarfile.open(source) as archive:
        archive.extractall(extraction, filter="data")
    extracted = extraction / "gregorithm-0.1.0"
    rebuilt = extraction / "rebuilt"
    run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(rebuilt),
            str(extracted),
        ]
    )
    rebuilt_matches = digest(wheel) == digest(rebuilt / wheel.name)
    if not rebuilt_matches:
        raise ValueError("Wheel rebuilt from source distribution differs")

    print("Installing wheel in an isolated environment, offline", flush=True)
    if not ENV.exists():
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(ENV)
    interpreter = ENV / "Scripts/python.exe" if os.name == "nt" else ENV / "bin/python"
    install_output = run(
        [
            str(interpreter),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(ROOT / ".cache/release-wheelhouse"),
            "--force-reinstall",
            str(wheel),
        ]
    )
    smoke_folder = Path(tempfile.mkdtemp(dir=CACHE, prefix="smoke-"))
    script = r"""
import json, sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import gregorithm
from gregorithm import (
    AnalysisConfig, DQDVConfig, analyze, normalize, open_result, open_trace, prepare_source
)
assert Path(gregorithm.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert not any(name in sys.modules for name in ("plotly","ipywidgets","pandas","matplotlib"))
assert DQDVConfig(max_auto_points=None).max_auto_points is None
preparation_input=Path("dedupe.parquet")
pq.write_table(pa.table({"time_s":[0.,1.,1.,2.],"current_mA":[1.,2.,99.,3.],
                         "voltage_V":[4.,4.,99.,4.],"note":["first","keep","remove","last"]}),preparation_input)
prepared=prepare_source(preparation_input,Path("prepared.parquet"))
prepared_table=pa.Table.from_batches(list(prepared.iter_batches()))
assert prepared_table.column("note").to_pylist()==["first","keep","last"]
assert prepared.report["duplicates_removed"]==1
global_totals=prepared.totals(include_energy=False)
assert np.isclose(global_totals.values["charge_mAh"],4./3600.)
rates=prepared.c_rate(reference_mAh=2.)
rate_rows=pa.Table.from_batches(list(rates.iter_batches()))
assert rate_rows.column("c_rate_h_inv").to_pylist()==[.5,1.,1.5]
assert np.isclose(rates.mean().values["mean_current_mA"],2.)
steps=prepared.steps()
try:
    inferred=pa.Table.from_batches(list(steps.iter_batches()))
    assert inferred.column("step_type").to_pylist()==["CV"]
    labelled=pa.Table.from_batches(list(steps.measurements().iter_batches()))
    assert labelled.column("note").to_pylist()==["first","keep","last"]
    assert "step_type" not in prepared_table.column_names
finally:
    steps.close()
halves=prepared.capacity(include_energy=True)
try:
    summary=pa.Table.from_batches(list(halves.iter_batches()))
    assert summary.column("half_cycle").to_pylist()==[1]
    assert np.isclose(summary.column("capacity_mAh")[0].as_py(),4./3600.)
    assert np.isclose(summary.column("energy_Wh")[0].as_py(),16./3600000.)
    paired=pa.Table.from_batches(list(halves.summary().iter_batches()))
    assert paired.column("coulombic_efficiency_reason").to_pylist()==["missing_counterpart"]
    normalized=pa.Table.from_batches(list(halves.normalize(mass_g=.005,area_cm2=2).iter_batches()))
    assert np.isclose(normalized.column("capacity_mAh_g")[0].as_py(),4./3600./.005)
    assert np.isclose(normalized.column("capacity_mAh_cm2")[0].as_py(),4./3600./2)
    exported=Path("capacity.parquet")
    halves.write_parquet(exported)
    reopened=open_result(exported)
    reopened_table=pa.Table.from_batches(list(reopened.iter_batches()))
    assert reopened.report["operation"]=="half_cycle_capacity"
    assert np.isclose(reopened_table.column("capacity_mAh")[0].as_py(),4./3600.)
finally:
    halves.close()
assert not set(prepared_table.column_names)&{"trace_id","result_id","sample_id","selection_id"}
source=Path("synthetic.parquet")
pq.write_table(pa.table({"time_s":[0.,1800.,3600.],"current_mA":[1.,1.,1.],
                         "voltage_V":[4.,4.,4.],"note":["synthetic"]*3}),source)
result=analyze(open_trace(source),config=AnalysisConfig(backend="numpy"),export_samples=True)
block=pa.Table.from_batches(list(result.samples.iter_batches()))
q=block.column("global_charge_mAh")[-1].as_py()
e=block.column("global_charge_Wh")[-1].as_py()
assert np.isclose(q,1.) and np.isclose(e,.004)
assert block.column("note").to_pylist()==["synthetic"]*3
normalized=pa.Table.from_batches(list(normalize(result.samples,mass_g=.5).iter_batches()))
assert np.isclose(normalized.column("capacity_mAh_g")[-1].as_py(),2.)
print(json.dumps({"version":gregorithm.__version__,"isolated_installed_package":True,
                  "core_imports_graphics":False,"capacity_mAh":q,"energy_Wh":e,
                  "source_extras_preserved":True,"normalization_verified":True,
                  "identity_free_preparation_verified":True,"configurable_grid_cap_verified":True,
                  "identity_free_point_totals_verified":True,
                  "identity_free_half_cycle_capacity_verified":True,
                  "identity_free_cycle_summary_normalization_verified":True,
                  "identity_free_point_c_rate_verified":True,
                  "optional_step_classification_verified":True,
                  "persistent_result_reopening_verified":True,
                  "status":result.status}))
"""
    smoke = json.loads(
        run([str(interpreter), "-I", "-c", script], cwd=smoke_folder).strip().splitlines()[-1]
    )
    report = {
        "wheel": {
            "name": wheel.name,
            "bytes": wheel.stat().st_size,
            "sha256": digest(wheel),
            "members": len(wheel_names),
        },
        "sdist": {
            "name": source.name,
            "bytes": source.stat().st_size,
            "sha256": digest(source),
            "members": len(source_names),
        },
        "private_assets_excluded": True,
        "private_input_references_checked": bool(private_tokens),
        "unexecuted_notebooks_only": True,
        "rebuilt_wheel_matches": rebuilt_matches,
        "isolated_smoke": smoke,
        "dependencies_installed_offline": "Successfully installed" in install_output,
        "python": sys.version,
        "public_publication_performed": False,
    }
    (DIST / "VERIFICATION.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
