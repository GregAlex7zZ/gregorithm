# Third-party notices

gregorithm is MIT-licensed. Its dependencies are separate works distributed under their
own licenses. The package does not bundle dependency wheels, Python or measurement data.

| Project | Role | License family |
|---|---|---|
| NumPy | numerical arrays | BSD-3-Clause and compatible bundled notices |
| PyArrow | streaming tables and Parquet | Apache-2.0 |
| SciPy | smoothing and numerical utilities | BSD-3-Clause and bundled runtime notices |
| pandas | optional notebook tables | BSD-3-Clause |
| Plotly.py | optional interactive plots | MIT |
| anywidget | optional notebook widget transport | MIT |
| ipywidgets | optional notebook controls | BSD-3-Clause |
| Matplotlib | optional static plots | PSF-based |
| Numba / llvmlite | optional acceleration | BSD-2-Clause / Apache-2.0 with LLVM exception |

These licenses permit an MIT license for gregorithm's original source while their own
copyright and license terms remain in force. Binary redistributors must inspect the exact
dependency builds they ship, including native libraries bundled by numerical packages.
