# gregorithm

Streaming analysis of battery cycling data with bounded memory use. The Python core is
independent of any graphical interface and keeps scientific calculations explicit.

```python
from gregorithm import prepare_source

cell = prepare_source("cell.csv", "cell.prepared.parquet")
capacity = cell.capacity(include_energy=True)
capacity.table()
capacity.plot()
```

Start with the [getting-started notebook](examples/getting_started.ipynb) or the
[user guide](docs/user-guide/index.md).

## Install

Python 3.12 or later is required.

```bash
python -m pip install .
python -m pip install ".[notebook,plot]"
```

## Main workflow

`prepare_source()` validates a standardized CSV/TXT or Parquet source and creates a
reusable prepared Parquet file. Standardized inputs contain `time_s`, `current_mA` and
`voltage_V`. Use the sibling `battread` library first for vendor exports and noncanonical
column names. Duplicate timestamps retain the first complete row and raise a warning;
distinct timestamps that move backwards raise an error. Existing source columns are
preserved.

```python
from gregorithm import PulseProfile, open_result, prepare_source

cell = prepare_source("cell.csv", "cell.prepared.parquet")
cell.preview()
cell.totals(include_energy=True).table()

capacity = cell.capacity(include_energy=True)
capacity.table()
capacity.summary().table()
capacity.normalize(mass_g=0.005).table()

cell.c_rate(capacity=capacity).preview()
cell.cumulative().preview()
cell.time_window(120, 180).table()

steps = cell.steps()
steps.table()
steps.quantities().table()

events = cell.events()
events.table()
events.dcir(delay_s=10).table()

profile = PulseProfile(rest_before_s=600, pulse_s=30, rest_after_s=600)
cell.events(profile=profile).table()

curves = cell.dqdv(cycles=[2])
curves.table()
curves.plot()

capacity.write_parquet("capacity.parquet")
saved = open_result("capacity.parquet")
saved.table()
```

For large point-level outputs, use `iter_batches()` or `write_parquet()` rather than
materializing the complete result.

## Scope

The current release covers capacity, energy, cycles, normalization, C-rate, optional
step classification, generic pulse discovery, DCIR, dQ/dV, plotting and multi-file
execution. Specialist GITT and ICI calculations are outside scope.

Scientific definitions and point-selection rules are documented in
[Scientific methods](docs/scientific/methods.md). See [API recipes](docs/api/index.md),
[large-file guidance](docs/user-guide/large-files.md) and [development](CONTRIBUTING.md).

Original gregorithm code is available under the [MIT License](LICENSE). Dependencies
retain their own licenses; see [third-party notices](THIRD_PARTY_NOTICES.md). Real
measurement files and internal project records are not part of this repository.

The software is provided without warranty. Users remain responsible for checking input
quality, configuration and scientific results before relying on them experimentally.
