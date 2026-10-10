"""Capture installed runtime versions and source-file hashes without GUI imports."""

import hashlib
import importlib.metadata as metadata
import sys
from pathlib import Path


def _capture():
    """Snapshot code files at module import so a later file edit does not alter records."""
    versions = {}
    for name in (
        "gregorithm",
        "numpy",
        "pyarrow",
        "scipy",
        "numba",
        "pandas",
        "plotly",
        "anywidget",
    ):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    hashes = {}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(65536), b""):
                digest.update(block)
        hashes[path.name] = digest.hexdigest()
    return {
        "library_version": versions.get("gregorithm") or "0.1.0",
        "runtime_versions": versions,
        "python_version": sys.version,
        "source_file_sha256": hashes,
    }


_SNAPSHOT = _capture()


def runtime_provenance():
    """Return an independent small runtime/code snapshot for a result manifest.

    Optional dependency metadata is queried without importing those libraries.
    Hashes describe installed Python source files captured at module import;
    they do not identify compiled native wheels, whose versions/notices are
    recorded separately. This is provenance, not experimental validation.
    """
    return {
        **_SNAPSHOT,
        "runtime_versions": dict(_SNAPSHOT["runtime_versions"]),
        "source_file_sha256": dict(_SNAPSHOT["source_file_sha256"]),
    }
