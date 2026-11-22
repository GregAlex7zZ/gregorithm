# Researcher notebook route

These notebooks introduce the public `gregorithm` API through a tiny synthetic cycling dataset. They
are written for battery researchers who are new to Python. Every notebook is independent: open one,
choose **Run All**, and read from top to bottom.

## Install and launch

From the project directory:

```console
python -m venv .venv
# Activate .venv using the command for your operating system.
python -m pip install --upgrade pip
python -m pip install -e ".[notebooks]"
python -m jupyter lab
```

The `notebooks` extra is for examples only. It does not add Jupyter or plotting libraries to the
core runtime dependencies or distribution package.

## Suggested order

1. [`01_getting_started_import.ipynb`](notebooks/01_getting_started_import.ipynb) — import a CSV and
   inspect canonical seconds, amperes, and volts.
2. [`02_data_elaboration_and_capacity.ipynb`](notebooks/02_data_elaboration_and_capacity.ipynb) —
   inspect cycles, half-cycles, steps, Capacity, and efficiency.
3. [`03_dcir_detection_and_review.ipynb`](notebooks/03_dcir_detection_and_review.ipynb) — detect a
   compact synthetic pulse, review it explicitly, and calculate DCIR.
4. [`04_dqdv_and_figures.ipynb`](notebooks/04_dqdv_and_figures.ipynb) — calculate and plot raw dQ/dV.
5. [`05_export_and_reproducibility.ipynb`](notebooks/05_export_and_reproducibility.ipynb) — export
   result tables and record fingerprints and software version.
6. [`06_gitt_ici_diffusion.ipynb`](notebooks/06_gitt_ici_diffusion.ipynb) — GITT diffusion
   coefficients on a synthetic record with a known answer: analysis, review and export.

All default data are synthetic: notebook 06 writes its own GITT record; the others use [`data/synthetic_cycling.csv`](data/synthetic_cycling.csv).
No external download is used. Notebooks 01–05 include an optional `LOCAL_FILE` setting. Leave it as
`None` for the synthetic example; if you choose a local file, keep it outside the repository and
confirm its column roles, units, and positive-current convention before analysis.

## Troubleshooting

- **Jupyter cannot find the kernel:** activate the same environment used for installation, then run
  `python -m ipykernel install --user --name gregorithm-local` if your local Jupyter setup requires it.
- **Ambiguous columns or units:** use the explicit public `ColumnMapping` API described in
  [user guide](../docs/user-guide.md#1-inspect-and-import-a-source). Never guess units or invert current silently.
- **A result is unavailable:** inspect its diagnostics. Missing values are scientifically distinct
  from numerical zero.
- **Your data are private:** keep the path local. The notebooks never upload files or use network
  access.
- **Large files:** begin with a copy or small synthetic file to learn the workflow. The examples are
  demonstrations, not performance claims.
