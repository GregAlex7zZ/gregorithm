"""Local run tests verification helper. Run with the project interpreter; outputs remain inside the workspace."""

import tempfile
from pathlib import Path

import pytest

root = Path(__file__).resolve().parents[1]
(root / ".test-runs").mkdir(exist_ok=True)
if __name__ == "__main__":
    import sys

    tmp = tempfile.mkdtemp(dir=root / ".test-runs", prefix="run-")
    raise SystemExit(
        pytest.main(
            [
                "-q",
                "--tb=short",
                "--basetemp",
                tmp,
                "--junitxml",
                str(root / "docs/test-results.xml"),
                *sys.argv[1:],
            ]
        )
    )
