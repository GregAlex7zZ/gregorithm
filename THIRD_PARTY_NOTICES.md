# Third-party notices

gregorithm's own source code is released under the [MIT License](LICENSE). It does not bundle or
vendor third-party code: the packages below are installed separately by `pip` and remain under
their own licenses. This inventory is provided for transparency and is not legal advice.

## Runtime dependencies

| Package | Version range | License |
|---|---|---|
| [NumPy](https://numpy.org/) | `>=2.5,<3` | BSD-3-Clause (its wheels also contain components under 0BSD, MIT, Zlib and CC0-1.0) |
| [PyArrow](https://arrow.apache.org/) | `>=25,<27` | Apache-2.0 |
| [SciPy](https://scipy.org/) | `>=1.16,<2` | BSD-3-Clause |

NumPy, PyArrow and SciPy binary wheels include compiled third-party libraries; their license texts
are shipped inside those wheels.

## Optional `notebooks` extra (examples only)

| Package | License |
|---|---|
| JupyterLab | BSD-3-Clause |
| ipykernel | BSD-3-Clause |
| nbclient, nbformat | BSD-3-Clause |
| Matplotlib | Matplotlib License (PSF-based, BSD-compatible) |

## Development and build tools (not distributed)

| Package | License |
|---|---|
| Hatchling, build | MIT |
| pytest | MIT |
| Ruff | MIT |
| mypy | MIT |
| pre-commit | MIT |

All of these licenses are permissive and compatible with distributing gregorithm under MIT.

## Algorithms

The Savitzky–Golay smoothing in `gregorithm.dqdv` is an original NumPy implementation of the
published method (A. Savitzky and M. J. E. Golay, *Anal. Chem.* 36 (1964) 1627–1639); no SciPy
code is used.

The `histogram` and `histogram_weighted` dQ/dV methods, the voltage reconstruction and its APE and
RMSE metrics in `gregorithm.dqdv` are original NumPy implementations of the method published by
E. Flores and S. Clark, "Simple and Comparable Computation of Incremental Capacity Curves",
*J. Electrochem. Soc.* 173 (2026) 120520, doi:10.1149/1945-7111/ae7e5c (open access, CC BY 4.0).
The authors' MIT-licensed reference implementation was consulted to match its conventions; no
code was copied. The Gaussian filter
reproduces the behaviour of `scipy.ndimage.gaussian_filter1d` with NumPy only.
