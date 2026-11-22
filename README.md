# gregorithm

**gregorithm** is a Python library for analysing galvanostatic battery-cycling data. It imports
time, current and voltage measurements, identifies half-cycles, cycles and step types (rest, CC,
CV), and computes per-cycle Capacity, energy and efficiency, C-rate, DC internal resistance (DCIR)
and differential capacity (dQ/dV) by interpolated gradient or the
derivative-free voltage histograms of Flores and Clark (2026), with voltage reconstruction as an
objective quality check. Its GITT/ICI module computes solid-state diffusion coefficients from pulse
experiments (GITT_VA, GITT_DELTA, ICI and the analytical spherical Delacourt fit) with explicit
window selection, advisory quality indicators and operator review; cycle numbering can be corrected
after the fact without re-importing data. Every result is written as an immutable, checksummed and
fingerprinted set of Parquet/JSON files, so analyses are reproducible and safe to share.

- **Streaming and bounded memory**: files larger than RAM are processed in batches.
- **Explicit science**: no silent unit guessing, no polarity inversion, no interpolated samples;
  unavailable values are `null` with a stable diagnostic code, never a fabricated zero.
- **Reproducible**: each result records the library version, parameters and input fingerprints.
- **Lightweight**: the only runtime dependencies are NumPy, PyArrow and SciPy. No GUI, no network
  access.

> **Status: alpha (0.x).** The API and result schemas may still change between minor versions.
> Results are verified against analytic test cases, which establishes software and mathematical
> consistency; it does not validate any particular instrument or laboratory method.

## Installation

Python 3.12 or newer is required.

Clone or download this repository, then from its folder:

```console
python -m pip install .
python -m pip install ".[notebooks]"   # optional: JupyterLab and Matplotlib for the examples
python -m pip install -e ".[dev]"      # development: tests, linters, type checker
```

## Quick start

```python
from pathlib import Path

import pyarrow.parquet as pq

from gregorithm import (
    CapacityParameters,
    CapacityRequest,
    ElaborationRequest,
    ImportRequest,
    analyze_capacity,
    elaborate,
    import_source,
)

store = Path("results")  # any directory you own; results are written below it
cell = "cell-01"  # one identifier per sample (cell)

imported = import_source(ImportRequest(Path("cell-01.csv"), store, cell, "import-1", "import-1"))
elaborated = elaborate(
    ElaborationRequest(
        imported.ref.path / "points" / "canonical.parquet",
        imported.report.content_hash,
        store,
        cell,
        "elaboration-1",
        "elaboration-1",
    )
)
capacity = analyze_capacity(
    CapacityRequest(
        elaborated.ref.path,
        elaborated.fingerprint,
        store,
        cell,
        "capacity-1",
        "capacity-1",
        CapacityParameters(reference_cycle=2, active_material_mass_mg=5.0),
    )
)
print(pq.read_table(capacity.ref.path / "tables" / "capacity.parquet"))
```

The input CSV above has the recognized header `time_s,current_a,voltage_v` (or
`time_s,current_mA,voltage_V`). Any other layout needs an explicit
[`ColumnMapping`](docs/user-guide.md#1-inspect-and-import-a-source).

## Conventions

| Quantity | Unit | Notes |
|---|---|---|
| time | s | elapsed time, strictly increasing after cleaning |
| current | A | **positive = charge, negative = discharge** (never inverted) |
| voltage | V | |
| capacity / energy | Ah / Wh | cumulative values reset at every half-cycle |
| resistance | Ω | DCIR magnitude; signed ΔV and ΔI are kept |
| dQ/dV | Ah/V | |

## Documentation

- [Getting started](docs/getting-started.md) – installation and the example notebooks
- [User guide](docs/user-guide.md) – the complete workflow, step by step
- [Scientific methods](docs/methods/README.md) – exactly how every quantity is computed
- [Results and storage](docs/results-and-storage.md) – output files, columns, manifests,
  fingerprints and diagnostic codes
- [API reference](docs/api-reference.md) – overview of the public API (every object also has a
  detailed docstring: `help(gregorithm.elaborate)`)
- [Example notebooks](examples/README.md) – six Run-All notebooks on synthetic data

## Graphical interface

**SparkBox** is a desktop application built on gregorithm's public API. The
library itself has no GUI dependency and can be used from scripts and notebooks.

## Contributing, security and citation

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md) and
[CITATION.cff](CITATION.cff). Changes are listed in [CHANGELOG.md](CHANGELOG.md); authorship in
[AUTHORS.md](AUTHORS.md).

If you use the histogram dQ/dV methods or the voltage reconstruction, please also cite:

> E. Flores and S. Clark, "Simple and Comparable Computation of Incremental Capacity Curves",
> *J. Electrochem. Soc.* **173**, 120520 (2026).
> [doi:10.1149/1945-7111/ae7e5c](https://doi.org/10.1149/1945-7111/ae7e5c)

## License

gregorithm is released under the [MIT License](LICENSE). Its dependencies keep their own licenses;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
