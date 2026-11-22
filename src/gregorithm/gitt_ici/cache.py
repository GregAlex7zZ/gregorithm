"""Small persistent JSON cache of technical intermediates with dependency-aware identities.

The cache stores only JSON intermediates (for example selected window bounds). It never stores
review or acceptance state and is never a second list of exportable results. Every entry
records its dependency fingerprints; a lookup with different fingerprints is a miss.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from gregorithm.contracts import JsonValue
from gregorithm.identity import canonical_parameter_bytes


def cache_key(namespace: str, dependencies: Mapping[str, str], parameters: JsonValue) -> str:
    """Build a deterministic cache identity from all relevant scientific dependencies."""
    if not namespace:
        raise ValueError("cache namespace must not be empty")
    digest = hashlib.sha256(b"gregorithm-gitt-ici-cache-v1\0")
    digest.update(namespace.encode())
    digest.update(b"\0")
    digest.update(json.dumps(dict(sorted(dependencies.items())), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(canonical_parameter_bytes(parameters))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CacheWrite:
    """One staged cache entry, committed only after its batch completes."""

    key: str
    dependencies: tuple[tuple[str, str], ...]
    value: JsonValue


class IntermediateCache:
    """Caller-owned cache; acceptance/review state is deliberately never stored here."""

    def __init__(self, root: Path) -> None:
        """Create or open the cache directory ``root``."""
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, dependencies: Mapping[str, str]) -> JsonValue | None:
        """Return the cached value when key and every dependency fingerprint match, else None.

        Unreadable, corrupt or foreign-schema files are treated as misses.
        """
        path = self.root / f"{key}.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            return None
        if data.get("key") != key or data.get("dependencies") != dict(sorted(dependencies.items())):
            return None
        return cast("JsonValue", data.get("value"))

    def commit(self, writes: tuple[CacheWrite, ...]) -> None:
        """Atomically commit staged technical intermediates after a successful batch."""
        for write in writes:
            target = self.root / f"{write.key}.json"
            temporary = self.root / f".{write.key}.{uuid.uuid4().hex}.tmp"
            payload = {
                "schema_version": 1,
                "key": write.key,
                "dependencies": dict(write.dependencies),
                "value": write.value,
            }
            temporary.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
                encoding="utf-8",
            )
            os.replace(temporary, target)

    def invalidate_changed(self, dependencies: Mapping[str, str]) -> tuple[Path, ...]:
        """Remove only entries carrying a supplied dependency with a different fingerprint."""
        removed: list[Path] = []
        for path in self.root.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                stored = data["dependencies"]
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                continue
            if any(
                name in stored and stored[name] != value for name, value in dependencies.items()
            ):
                path.unlink()
                removed.append(path)
        return tuple(removed)
