# Contributing to gregorithm

Thank you for your interest in improving gregorithm. Bug reports, documentation fixes and
focused pull requests are welcome.

## Development setup

Use CPython 3.12 or newer in a virtual environment:

```console
python -m pip install -e ".[dev]"
pre-commit install
```

## Quality checks

Run these before opening a pull request (the CI runs the same checks):

```console
ruff format --check .
ruff check .
mypy
python -m pytest
python -m build
```

## Conventions

- Code, docstrings, comments and documentation are written in English.
- Every module, class and function has a docstring explaining its purpose, usage and limits.
- Production data paths are vectorized and bounded in memory: process data in batches, never with
  a Python loop over individual samples.
- The runtime depends only on NumPy, PyArrow and SciPy (the bounded scalar optimizer and
  special functions of the GITT/ICI analytical fit). No GUI, network or telemetry code belongs in the
  library; applications build on its public API.
- Unavailable results are `null` with a diagnostic code, never a fabricated number.

## Scientific changes

The rules in [docs/methods](docs/methods/README.md) define the library's results. A change to a
scientific rule, default or tolerance must:

1. be discussed in an issue first, with references or a worked example;
2. update the relevant method page and the changelog;
3. add or update regression tests with independently computed expected values.

Every bug fix in parsing, numerical, memory or processing behaviour must include reproducible
verification appropriate to the change.

## Notebooks

If you change an example notebook, run it from a clean environment with **Run All**, then commit
it with outputs and execution counts cleared.

## Data and privacy

Do not commit experimental datasets, files with identifying metadata, local absolute paths,
credentials or generated result stores. Use small synthetic data in tests and examples.

## Pull requests

Keep pull requests focused. Describe the change, the tests you ran (platform and Python version)
and any remaining uncertainty.
