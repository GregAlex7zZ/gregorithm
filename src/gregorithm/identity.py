"""Deterministic hashing of analysis parameters and dependencies.

Fingerprints must not depend on dictionary order, float formatting or the difference between `1`
and `1.0`. Values are therefore encoded with explicit type tags, floats as `float.hex()` (keeping
`+0.0` and `-0.0` distinct) and mapping keys sorted. NaN and infinity are rejected.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import DependencyRef, JsonValue


def _normalized(value: JsonValue) -> list[Any]:
    """Convert a JSON-like value into a type-tagged list structure."""
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Canonical values cannot contain NaN or infinity")
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Canonical mapping keys must be strings")
        return ["map", [[key, _normalized(value[key])] for key in sorted(value)]]
    if isinstance(value, Sequence):
        return ["list", [_normalized(item) for item in value]]
    raise TypeError(f"Unsupported canonical value: {type(value).__name__}")


def canonical_parameter_bytes(value: JsonValue) -> bytes:
    """Encode a JSON-like value as unambiguous, type-tagged UTF-8 bytes.

    Raises `ValueError` for NaN/infinity and `TypeError` for unsupported value types or
    non-string mapping keys.
    """
    return json.dumps(
        _normalized(value),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _domain_hash(domain: bytes, payload: bytes) -> str:
    """Return the SHA-256 hex digest of `domain`, a NUL separator and `payload`."""
    return hashlib.sha256(domain + b"\0" + payload).hexdigest()


def hash_parameters(parameters: Mapping[str, JsonValue]) -> str:
    """Return the SHA-256 fingerprint of a parameter mapping.

    The digest uses the domain ``gregorithm-parameters-v1`` and `canonical_parameter_bytes`.
    """
    return _domain_hash(
        b"gregorithm-parameters-v1",
        canonical_parameter_bytes(dict(parameters)),
    )


def hash_dependencies(dependencies: Sequence[DependencyRef]) -> str:
    """Return the SHA-256 fingerprint of a set of dependency references.

    Dependencies are sorted by identifier, so input order does not matter. Raises `ValueError`
    when two dependencies share an identifier.
    """
    ordered = sorted(dependencies, key=lambda dependency: dependency.id)
    if len({dependency.id for dependency in ordered}) != len(ordered):
        raise ValueError("Dependency identifiers must be unique")
    payload: JsonValue = [
        {"id": dependency.id, "fingerprint": dependency.fingerprint} for dependency in ordered
    ]
    return _domain_hash(b"gregorithm-dependencies-v1", canonical_parameter_bytes(payload))
