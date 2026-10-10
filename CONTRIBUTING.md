# Contributing

Use English for code, docstrings, tests, documentation and examples. Keep the public API
small, notebook-friendly and independent of a GUI. Scientific behavior changes require
tests and an update to `docs/scientific/methods.md`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,notebook,plot]"
.\.venv\Scripts\python.exe tools/run_tests.py
.\.venv\Scripts\python.exe -m ruff check src tests tools
.\.venv\Scripts\python.exe tools/check_documentation.py
```

Do not commit real measurements, prepared outputs, executed notebooks, caches or local
validation reports. Synthetic tests must never be described as experimental validation.
Preserve third-party notices and document incorporated external code before merging.
By contributing original code, documentation or tests, you agree to provide that
contribution under the repository's MIT License.
