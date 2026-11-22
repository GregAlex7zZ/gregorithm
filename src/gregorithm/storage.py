"""Immutable result generations and atomic publication.

Every operation writes into a caller-owned root directory with this layout::

    <root>/sample-<sample_id>/
        current.json                    pointer to the latest generation
        generations/<generation_id>/    immutable published results + manifest.json
        jobs/<job_id>.staging/          in-progress output (removed or published)

A manifest lists every file with its size and SHA-256 together with the library version,
operation, parameter and dependency fingerprints and diagnostics. Publication renames a fully
validated staging directory into place and then atomically replaces `current.json`, so readers
never observe a partially written result. Identifiers are limited to 1-128 ASCII letters, digits,
dots, underscores or hyphens, and symbolic links or junctions are rejected.

Limitation: on Windows, directory metadata cannot be flushed (fsynced) with the standard library,
so atomic renames are guaranteed but power-loss durability of the rename is not.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePath
from typing import Any, BinaryIO

from .__about__ import __version__
from .contracts import CleanupReport, DependencyRef, OperationId, ResultInfo, ResultRef
from .diagnostics import Diagnostic, DiagnosticCode, DiagnosticKind
from .progress import CancelToken, check_cancelled

MANIFEST_SCHEMA_VERSION = 1
POINTER_SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTROL_FILES = {".active.lock", "manifest.json"}
FaultHook = Callable[[str], None]


class StorageError(RuntimeError):
    """Structured storage failure; `diagnostic` holds the reason code.

    Codes include path validation errors, `RESULT_JOB_EXISTS`, `RESULT_GENERATION_EXISTS`,
    manifest/checksum errors, `RESULT_PUBLICATION_CONFLICT` and `STORAGE_IO_ERROR`. After any
    failure the previously published generation remains current.
    """

    def __init__(self, diagnostic: Diagnostic) -> None:
        """Store the diagnostic and use its message as text."""
        self.diagnostic = diagnostic
        super().__init__(diagnostic.message)


def _storage_error(
    code: DiagnosticCode, message: str, *, operation: str | None = None
) -> StorageError:
    """Build a `StorageError` with an error diagnostic."""
    return StorageError(
        Diagnostic(code=code, kind=DiagnosticKind.ERROR, message=message, operation=operation)
    )


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(data: Mapping[str, Any]) -> bytes:
    """Serialize a mapping as sorted, compact UTF-8 JSON with a trailing newline."""
    return (
        json.dumps(data, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _validate_identifier(value: str, label: str) -> None:
    """Raise `PATH_INVALID` unless `value` is a safe 1-128 character identifier."""
    if not _IDENTIFIER.fullmatch(value):
        raise _storage_error(
            DiagnosticCode.PATH_INVALID,
            f"Invalid {label}; use 1-128 ASCII letters, digits, dots, underscores or hyphens.",
        )


def _resolve_contained(root: Path, candidate: Path) -> Path:
    """Resolve `candidate` and raise `PATH_OUTSIDE_ROOT` if it escapes `root`."""
    resolved_root = root.resolve(strict=False)
    resolved_candidate = candidate.resolve(strict=False)
    if not resolved_candidate.is_relative_to(resolved_root):
        raise _storage_error(DiagnosticCode.PATH_OUTSIDE_ROOT, "Path escapes the caller root.")
    return resolved_candidate


def _safe_relative_path(value: str) -> Path:
    """Validate a relative result path; reject absolute, `..` and reserved names."""
    path = PurePath(value)
    if not value or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise _storage_error(
            DiagnosticCode.PATH_INVALID, "Result file path is not safe and relative."
        )
    if path.as_posix() in _CONTROL_FILES or path.name in _CONTROL_FILES:
        raise _storage_error(DiagnosticCode.PATH_INVALID, "Result file path is reserved.")
    return Path(*path.parts)


def _is_link_like(path: Path) -> bool:
    """Return True for symbolic links and Windows junctions."""
    return path.is_symlink() or path.is_junction()


def _fsync_file(stream: BinaryIO) -> None:
    """Flush and fsync an open binary file."""
    stream.flush()
    os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    """Fsync a directory entry on POSIX; a no-op on Windows (not supported)."""
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class _FileLock:
    """Exclusive, non-blocking, OS-backed lock on a single file.

    Uses `msvcrt.locking` on Windows and `fcntl.flock` elsewhere. Acquisition never waits: if
    another process holds the lock, `RESULT_PUBLICATION_CONFLICT` is raised.
    """

    def __init__(self, path: Path) -> None:
        """Prepare a lock for `path` without acquiring it."""
        self.path = path
        self._stream: BinaryIO | None = None

    def acquire(self) -> None:
        """Create the lock file if needed and lock it, failing immediately if held."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream: BinaryIO | None = None
        try:
            stream = self.path.open("a+b")
            stream.seek(0)
            if stream.read(1) == b"":
                stream.seek(0)
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if stream is not None:
                stream.close()
            raise _storage_error(
                DiagnosticCode.RESULT_PUBLICATION_CONFLICT,
                "Another process owns the result publication lock.",
            ) from error
        self._stream = stream

    def release(self) -> None:
        """Unlock and close the lock file; safe to call when not held."""
        if self._stream is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                self._stream.seek(0)
                msvcrt.locking(self._stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        finally:
            self._stream.close()
            self._stream = None

    def __enter__(self) -> _FileLock:
        """Acquire the lock for a `with` block."""
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        """Release the lock at the end of a `with` block."""
        self.release()


@dataclass(frozen=True, slots=True)
class FileRecord:
    """Relative path, size in bytes and SHA-256 of one file listed in a manifest."""

    path: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        """Reject unsafe paths, negative sizes and malformed checksums."""
        _safe_relative_path(self.path)
        if self.bytes < 0 or not _SHA256.fullmatch(self.sha256):
            raise ValueError("Invalid file record")

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-compatible dictionary."""
        return {"path": self.path, "bytes": self.bytes, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class GenerationManifest:
    """Contents of `manifest.json`, the identity card of a result generation.

    Attributes:
        schema_version: Manifest schema version (currently 1).
        library_version: gregorithm version that produced the result.
        generation_id: Caller-chosen identifier of the generation.
        operation: `OperationId` of the producing operation.
        created_utc: ISO 8601 UTC creation time.
        parameter_hash: Fingerprint of the operation parameters.
        dependency_hash: Fingerprint of the sorted dependencies.
        dependencies: Inputs actually used, as `DependencyRef` records.
        content_hash: Content fingerprint of the result, when defined.
        files: Sorted `FileRecord` entries for every result file.
        diagnostics: Diagnostics recorded during the operation.
        row_counts: Row counts of the produced tables.
        logical_schemas: Logical schema identifiers of the produced tables.
    """

    schema_version: int
    library_version: str
    generation_id: str
    operation: OperationId
    created_utc: str
    parameter_hash: str
    dependency_hash: str
    dependencies: tuple[DependencyRef, ...]
    content_hash: str | None
    files: tuple[FileRecord, ...]
    diagnostics: tuple[Diagnostic, ...]
    row_counts: tuple[tuple[str, int], ...] = ()
    logical_schemas: tuple[tuple[str, str], ...] = ()

    def validate(self) -> None:
        """Check versions, hashes, sorting and uniqueness; raise ValueError."""
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError("Unsupported manifest schema version")
        _validate_identifier(self.generation_id, "generation identifier")
        if not self.library_version:
            raise ValueError("Library version is required")
        for value in (self.parameter_hash, self.dependency_hash):
            if not _SHA256.fullmatch(value):
                raise ValueError("Manifest hash is invalid")
        if self.content_hash is not None and not _SHA256.fullmatch(self.content_hash):
            raise ValueError("Content hash is invalid")
        file_paths = [record.path for record in self.files]
        dependency_ids = [dependency.id for dependency in self.dependencies]
        if file_paths != sorted(file_paths) or len(file_paths) != len(set(file_paths)):
            raise ValueError("Manifest file records must be sorted and unique")
        if dependency_ids != sorted(dependency_ids) or len(dependency_ids) != len(
            set(dependency_ids)
        ):
            raise ValueError("Manifest dependencies must be sorted and unique")
        if any(
            not dependency.id or not _SHA256.fullmatch(dependency.fingerprint)
            for dependency in self.dependencies
        ):
            raise ValueError("Manifest dependency identity is invalid")
        if any(count < 0 for _, count in self.row_counts):
            raise ValueError("Manifest row counts must not be negative")
        datetime.fromisoformat(self.created_utc.replace("Z", "+00:00"))

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest as a JSON-compatible dictionary."""
        return {
            "schema_version": self.schema_version,
            "library_version": self.library_version,
            "generation_id": self.generation_id,
            "operation": self.operation.value,
            "created_utc": self.created_utc,
            "parameter_hash": self.parameter_hash,
            "dependency_hash": self.dependency_hash,
            "dependencies": [
                {"id": dependency.id, "fingerprint": dependency.fingerprint}
                for dependency in self.dependencies
            ],
            "content_hash": self.content_hash,
            "files": [record.to_dict() for record in self.files],
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
            "row_counts": dict(self.row_counts),
            "logical_schemas": dict(self.logical_schemas),
        }

    def canonical_bytes(self) -> bytes:
        """Validate and return the exact bytes written to disk."""
        self.validate()
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerationManifest:
        """Parse and validate a manifest dictionary; unknown fields fail."""
        expected = {
            "schema_version",
            "library_version",
            "generation_id",
            "operation",
            "created_utc",
            "parameter_hash",
            "dependency_hash",
            "dependencies",
            "content_hash",
            "files",
            "diagnostics",
            "row_counts",
            "logical_schemas",
        }
        if set(data) != expected:
            raise ValueError("Manifest fields do not match schema")
        manifest = cls(
            schema_version=int(data["schema_version"]),
            library_version=str(data["library_version"]),
            generation_id=str(data["generation_id"]),
            operation=OperationId(data["operation"]),
            created_utc=str(data["created_utc"]),
            parameter_hash=str(data["parameter_hash"]),
            dependency_hash=str(data["dependency_hash"]),
            dependencies=tuple(
                DependencyRef(id=str(item["id"]), fingerprint=str(item["fingerprint"]))
                for item in data["dependencies"]
            ),
            content_hash=data["content_hash"],
            files=tuple(
                FileRecord(
                    path=str(item["path"]), bytes=int(item["bytes"]), sha256=str(item["sha256"])
                )
                for item in data["files"]
            ),
            diagnostics=tuple(Diagnostic.from_dict(item) for item in data["diagnostics"]),
            row_counts=tuple(
                sorted((str(key), int(value)) for key, value in data["row_counts"].items())
            ),
            logical_schemas=tuple(
                sorted((str(key), str(value)) for key, value in data["logical_schemas"].items())
            ),
        )
        manifest.validate()
        return manifest


@dataclass(slots=True)
class StagingGeneration:
    """A staging directory owned by one running job.

    Obtained from `ResultStore.begin_job`. Write files with `write_bytes` (or directly below
    `path`), build the manifest with `build_manifest` and hand both to `ResultStore.publish`. The
    job holds an active lock until it is closed, which protects it from cleanup.
    """

    caller_root: Path
    sample_root: Path
    path: Path
    generation_path: Path
    generation_id: str
    _lease: _FileLock | None

    def close(self) -> None:
        """Release the job's active lock; publication calls this automatically."""
        if self._lease is not None:
            self._lease.release()
            self._lease = None

    def __enter__(self) -> StagingGeneration:
        """Return the staging generation for a `with` block."""
        return self

    def __exit__(self, *_: object) -> None:
        """Release the active lock at the end of a `with` block."""
        self.close()

    def write_bytes(self, relative_path: str, data: bytes) -> Path:
        """Write `data` to `relative_path` inside the staging directory.

        The file is fsynced. Unsafe or reserved paths and existing links raise `StorageError`.
        Returns the absolute path written.
        """
        relative = _safe_relative_path(relative_path)
        destination = _resolve_contained(self.path, self.path / relative)
        if destination.exists() and _is_link_like(destination):
            raise _storage_error(
                DiagnosticCode.PATH_OUTSIDE_ROOT, "Symlink result files are forbidden."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        _resolve_contained(self.path, destination.parent)
        with destination.open("wb") as stream:
            stream.write(data)
            _fsync_file(stream)
        return destination

    def build_manifest(
        self,
        *,
        operation: OperationId,
        parameter_hash: str,
        dependency_hash: str,
        dependencies: Sequence[DependencyRef] = (),
        content_hash: str | None = None,
        diagnostics: Sequence[Diagnostic] = (),
        row_counts: Mapping[str, int] | None = None,
        logical_schemas: Mapping[str, str] | None = None,
        created_utc: str | None = None,
    ) -> GenerationManifest:
        """Build a validated manifest over the files currently staged.

        Every regular staged file (except control files) is hashed. The library version and,
        unless supplied, the current UTC time are filled in automatically.
        """
        files = tuple(
            FileRecord(
                path=path.relative_to(self.path).as_posix(),
                bytes=path.stat().st_size,
                sha256=_sha256(path),
            )
            for path in sorted(self.path.rglob("*"))
            if path.is_file() and path.name not in _CONTROL_FILES and not path.is_symlink()
        )
        manifest = GenerationManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            library_version=__version__,
            generation_id=self.generation_id,
            operation=operation,
            created_utc=created_utc
            or datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z"),
            parameter_hash=parameter_hash,
            dependency_hash=dependency_hash,
            dependencies=tuple(sorted(dependencies, key=lambda value: value.id)),
            content_hash=content_hash,
            files=files,
            diagnostics=tuple(diagnostics),
            row_counts=tuple(sorted((row_counts or {}).items())),
            logical_schemas=tuple(sorted((logical_schemas or {}).items())),
        )
        manifest.validate()
        return manifest


class ResultStore:
    """Low-level access to a caller-owned result root.

    Analysis functions use this class internally; applications normally need only the result
    references they return plus `open_result`, `open_current`, `cleanup_job` and `sweep_stale`.
    Use it directly only to publish custom files with the same atomic guarantees.
    """

    def __init__(self, caller_root: Path) -> None:
        """Open (and create if needed) the result root `caller_root`."""
        self.root = caller_root.resolve(strict=False)
        self.root.mkdir(parents=True, exist_ok=True)

    def begin_job(self, sample_id: str, job_id: str, generation_id: str) -> StagingGeneration:
        """Create the staging directory for a new job and lock it.

        Raises `StorageError` with `RESULT_JOB_EXISTS` if the job identifier was already used for
        this sample, or `PATH_INVALID` for unsafe identifiers.
        """
        for value, label in (
            (sample_id, "sample identifier"),
            (job_id, "job identifier"),
            (generation_id, "generation identifier"),
        ):
            _validate_identifier(value, label)
        sample_root = _resolve_contained(self.root, self.root / f"sample-{sample_id}")
        jobs = sample_root / "jobs"
        generations = sample_root / "generations"
        jobs.mkdir(parents=True, exist_ok=True)
        generations.mkdir(parents=True, exist_ok=True)
        staging = _resolve_contained(self.root, jobs / f"{job_id}.staging")
        try:
            staging.mkdir()
        except FileExistsError as error:
            raise _storage_error(
                DiagnosticCode.RESULT_JOB_EXISTS, "The requested staging job already exists."
            ) from error
        lease = _FileLock(staging / ".active.lock")
        lease.acquire()
        return StagingGeneration(
            caller_root=self.root,
            sample_root=sample_root,
            path=staging,
            generation_path=generations / generation_id,
            generation_id=generation_id,
            _lease=lease,
        )

    def publish(
        self,
        staging: StagingGeneration,
        manifest: GenerationManifest,
        *,
        cancel: CancelToken | None = None,
        fault_hook: FaultHook | None = None,
    ) -> ResultRef:
        """Validate a staging directory and publish it atomically.

        Steps: verify ownership and checksums, take the per-sample publication lock (without
        waiting), write and fsync `manifest.json`, rename staging to
        `generations/<generation_id>` and atomically replace `current.json`. Returns the
        `ResultRef` of the new generation.

        Raises `StorageError` (`RESULT_GENERATION_EXISTS`, `RESULT_PUBLICATION_CONFLICT`,
        checksum or I/O codes) or `OperationCancelled`. A failure before the pointer replacement
        leaves the previous generation current.
        """
        operation = manifest.operation.value
        self._validate_staging_ownership(staging)
        staging.close()
        active_lock = staging.path / ".active.lock"
        active_lock.unlink(missing_ok=True)
        if manifest.generation_id != staging.generation_id:
            raise _storage_error(
                DiagnosticCode.RESULT_MANIFEST_INVALID,
                "Manifest and staging generation identifiers differ.",
                operation=operation,
            )
        self._validate_staging(staging.path, manifest)
        lock = _FileLock(staging.sample_root / ".publish.lock")
        pointer_temp: Path | None = None
        try:
            with lock:
                self._checkpoint("lock_acquired", cancel, operation, fault_hook)
                if staging.generation_path.exists():
                    raise _storage_error(
                        DiagnosticCode.RESULT_GENERATION_EXISTS,
                        "An immutable generation with this identifier already exists.",
                        operation=operation,
                    )
                self._checkpoint("before_manifest_write", cancel, operation, fault_hook)
                manifest_path = staging.path / "manifest.json"
                with manifest_path.open("xb") as stream:
                    stream.write(manifest.canonical_bytes())
                    _fsync_file(stream)
                self._checkpoint("after_manifest_fsync", cancel, operation, fault_hook)
                self._validate_generation_directory(staging.path)
                self._checkpoint("before_generation_rename", cancel, operation, fault_hook)
                os.replace(staging.path, staging.generation_path)
                _fsync_directory(staging.generation_path.parent)
                self._checkpoint("after_generation_rename", cancel, operation, fault_hook)

                manifest_sha256 = _sha256(staging.generation_path / "manifest.json")
                pointer = {
                    "schema_version": POINTER_SCHEMA_VERSION,
                    "generation_id": staging.generation_id,
                    "path": f"generations/{staging.generation_id}",
                    "manifest_sha256": manifest_sha256,
                }
                pointer_temp = staging.sample_root / f"current.{uuid.uuid4().hex}.tmp"
                with pointer_temp.open("xb") as stream:
                    stream.write(_canonical_json(pointer))
                    _fsync_file(stream)
                self._checkpoint("before_pointer_replace", cancel, operation, fault_hook)
                os.replace(pointer_temp, staging.sample_root / "current.json")
                pointer_temp = None
                _fsync_directory(staging.sample_root)
                self._checkpoint("after_pointer_replace", None, operation, fault_hook)
                return ResultRef(
                    path=staging.generation_path,
                    generation_id=staging.generation_id,
                    manifest_sha256=manifest_sha256,
                )
        except OSError as error:
            raise _storage_error(
                DiagnosticCode.STORAGE_IO_ERROR,
                "Result publication failed because of a filesystem error.",
                operation=operation,
            ) from error
        finally:
            if pointer_temp is not None:
                pointer_temp.unlink(missing_ok=True)

    @staticmethod
    def _checkpoint(
        name: str,
        cancel: CancelToken | None,
        operation: str,
        fault_hook: FaultHook | None,
    ) -> None:
        """Check cancellation and call the testing fault hook at `name`."""
        check_cancelled(cancel, operation)
        if fault_hook is not None:
            fault_hook(name)

    def _validate_staging_ownership(self, staging: StagingGeneration) -> None:
        """Ensure the staging object belongs to this store."""
        caller_root = staging.caller_root.resolve(strict=True)
        if caller_root != self.root:
            raise _storage_error(
                DiagnosticCode.PATH_OUTSIDE_ROOT,
                "Staging generation belongs to a different caller root.",
            )
        sample_root = _resolve_contained(caller_root, staging.sample_root)
        staging_path = _resolve_contained(caller_root, staging.path)
        generation_path = _resolve_contained(caller_root, staging.generation_path)
        if (
            not sample_root.name.startswith("sample-")
            or staging_path.parent != sample_root / "jobs"
            or not staging_path.name.endswith(".staging")
            or generation_path.parent != sample_root / "generations"
            or generation_path.name != staging.generation_id
        ):
            raise _storage_error(
                DiagnosticCode.PATH_INVALID,
                "Staging generation does not match the caller-owned store layout.",
            )

    @staticmethod
    def _validate_staging(path: Path, manifest: GenerationManifest) -> None:
        """Compare the staged file set with the manifest, byte by byte."""
        try:
            manifest.validate()
        except (KeyError, TypeError, ValueError) as error:
            raise _storage_error(
                DiagnosticCode.RESULT_MANIFEST_INVALID, "The result manifest is invalid."
            ) from error
        expected = {record.path: record for record in manifest.files}
        actual: dict[str, Path] = {}
        for candidate in path.rglob("*"):
            if _is_link_like(candidate):
                raise _storage_error(
                    DiagnosticCode.PATH_OUTSIDE_ROOT, "Symlinks are forbidden in staged results."
                )
            if candidate.is_file() and candidate.name not in _CONTROL_FILES:
                actual[candidate.relative_to(path).as_posix()] = candidate
        missing = set(expected) - set(actual)
        unexpected = set(actual) - set(expected)
        if missing:
            raise _storage_error(
                DiagnosticCode.RESULT_FILE_MISSING, "A file declared by the manifest is missing."
            )
        if unexpected:
            raise _storage_error(
                DiagnosticCode.RESULT_FILE_UNEXPECTED,
                "The staging directory contains an undeclared file.",
            )
        for relative, record in expected.items():
            candidate = actual[relative]
            if candidate.stat().st_size != record.bytes or _sha256(candidate) != record.sha256:
                raise _storage_error(
                    DiagnosticCode.RESULT_FILE_CHECKSUM_MISMATCH,
                    "A staged result file does not match its manifest checksum.",
                )

    @staticmethod
    def _validate_generation_directory(path: Path) -> GenerationManifest:
        """Load and fully verify a generation directory."""
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise _storage_error(
                DiagnosticCode.RESULT_MANIFEST_INVALID, "The generation manifest is missing."
            )
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Manifest root must be an object")
            manifest = GenerationManifest.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise _storage_error(
                DiagnosticCode.RESULT_MANIFEST_INVALID, "The generation manifest is corrupt."
            ) from error
        ResultStore._validate_staging(path, manifest)
        return manifest


def open_result(path: Path) -> ResultInfo:
    """Open a published generation directory and verify every file.

    Returns a `ResultInfo` with the result reference, operation, library version, diagnostics and
    file list. Raises `StorageError` if the manifest is missing or corrupt or any checksum differs.
    """
    generation = path.resolve(strict=True)
    manifest = ResultStore._validate_generation_directory(generation)
    manifest_hash = _sha256(generation / "manifest.json")
    return ResultInfo(
        ref=ResultRef(
            path=generation,
            generation_id=manifest.generation_id,
            manifest_sha256=manifest_hash,
        ),
        operation=manifest.operation,
        library_version=manifest.library_version,
        diagnostics=manifest.diagnostics,
        files=tuple(generation / record.path for record in manifest.files),
    )


def open_current(sample_root: Path) -> ResultInfo:
    """Open the generation referenced by `<sample_root>/current.json`.

    `sample_root` is the ``sample-<id>`` directory. The pointer and the generation are both
    validated; any inconsistency raises `StorageError` with `RESULT_POINTER_INVALID`.
    """
    root = sample_root.resolve(strict=True)
    pointer_path = root / "current.json"
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        expected = {"schema_version", "generation_id", "path", "manifest_sha256"}
        if not isinstance(pointer, dict) or set(pointer) != expected:
            raise ValueError("Pointer fields do not match schema")
        if pointer["schema_version"] != POINTER_SCHEMA_VERSION:
            raise ValueError("Unsupported pointer schema")
        _validate_identifier(str(pointer["generation_id"]), "generation identifier")
        if not _SHA256.fullmatch(str(pointer["manifest_sha256"])):
            raise ValueError("Pointer manifest hash is invalid")
        generation = _resolve_contained(root / "generations", root / str(pointer["path"]))
        if generation.parent != root / "generations":
            raise ValueError("Pointer does not name a direct generation")
        info = open_result(generation)
        if (
            info.ref.generation_id != pointer["generation_id"]
            or info.ref.manifest_sha256 != pointer["manifest_sha256"]
        ):
            raise ValueError("Pointer does not match generation")
        return info
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, StorageError) as error:
        raise _storage_error(
            DiagnosticCode.RESULT_POINTER_INVALID, "The current-generation pointer is invalid."
        ) from error


def _recognized_staging(caller_root: Path, job_path: Path) -> Path:
    """Validate that `job_path` is a staging job inside `caller_root`."""
    root = caller_root.resolve(strict=True)
    _assert_trusted_ancestors(root, job_path)
    job = _resolve_contained(root, job_path)
    if (
        not job.name.endswith(".staging")
        or job.parent.name != "jobs"
        or not job.parent.parent.name.startswith("sample-")
    ):
        raise _storage_error(DiagnosticCode.PATH_INVALID, "Path is not a recognized staging job.")
    _validate_identifier(job.parent.parent.name.removeprefix("sample-"), "sample identifier")
    _validate_identifier(job.name.removesuffix(".staging"), "job identifier")
    return job


def _assert_trusted_ancestors(root: Path, candidate: Path) -> None:
    """Reject lexical escapes and every link/reparse component below an owned root."""
    owned = root.resolve(strict=True)
    try:
        relative = candidate.absolute().relative_to(owned.absolute())
    except ValueError as error:
        raise _storage_error(
            DiagnosticCode.PATH_OUTSIDE_ROOT, "Path escapes the caller root."
        ) from error
    current = owned
    for part in relative.parts:
        if part in {".", ".."}:
            raise _storage_error(DiagnosticCode.PATH_OUTSIDE_ROOT, "Path escapes the caller root.")
        current = current / part
        if _is_link_like(current):
            raise _storage_error(
                DiagnosticCode.PATH_OUTSIDE_ROOT,
                "Linked or reparse-point cleanup paths are forbidden.",
            )
    _resolve_contained(owned, current)


def _recognized_pointer_temp(root: Path, candidate: Path) -> tuple[Path, Path]:
    """Validate that `candidate` is a temporary pointer file of a sample."""
    _assert_trusted_ancestors(root, candidate)
    pointer = _resolve_contained(root, candidate)
    sample = pointer.parent
    if (
        not sample.name.startswith("sample-")
        or not pointer.name.startswith("current.")
        or not pointer.name.endswith(".tmp")
        or pointer.name.count(".") != 2
    ):
        raise _storage_error(DiagnosticCode.PATH_INVALID, "Path is not a pointer temporary.")
    _validate_identifier(sample.name.removeprefix("sample-"), "sample identifier")
    _validate_identifier(
        pointer.name.removeprefix("current.").removesuffix(".tmp"), "pointer identifier"
    )
    return sample, pointer


def _assert_cleanup_tree_safe(path: Path) -> None:
    """Raise if a staging tree contains links that cleanup could follow."""
    for parent, directories, files in os.walk(path, topdown=True, followlinks=False):
        parent_path = Path(parent)
        for name in (*directories, *files):
            candidate = parent_path / name
            if _is_link_like(candidate):
                raise _storage_error(
                    DiagnosticCode.PATH_OUTSIDE_ROOT,
                    "A staging tree containing links cannot be cleaned.",
                )


def _job_is_live(job: Path) -> bool:
    """Return True if another process still holds the job's active lock."""
    active = job / ".active.lock"
    if not active.exists():
        return False
    lock = _FileLock(active)
    try:
        lock.acquire()
    except StorageError as error:
        if error.diagnostic.code is DiagnosticCode.RESULT_PUBLICATION_CONFLICT:
            return True
        raise
    else:
        lock.release()
        return False


def cleanup_job(caller_root: Path, job_path: Path) -> CleanupReport:
    """Remove an abandoned staging job, for example after a cancelled operation.

    Only a directory matching ``<root>/sample-<id>/jobs/<job>.staging`` can be removed. Live jobs
    (still locked) raise `RESULT_JOB_ACTIVE`; published generations are never touched. Returns a
    `CleanupReport`.
    """
    job = _recognized_staging(caller_root, job_path)
    if not job.exists():
        return CleanupReport(removed=(), skipped=(job,))
    cleanup_lock = _FileLock(job.parent / f".{job.name}.cleanup.lock")
    try:
        cleanup_lock.acquire()
        if _job_is_live(job):
            raise _storage_error(
                DiagnosticCode.RESULT_JOB_ACTIVE, "A live staging job cannot be removed."
            )
        _assert_cleanup_tree_safe(job)
        tombstone = job.parent / f".{job.name}.{uuid.uuid4().hex}.deleting"
        job.rename(tombstone)
    except StorageError:
        raise
    except OSError as error:
        raise _storage_error(
            DiagnosticCode.STORAGE_IO_ERROR, "The staging job could not be isolated for cleanup."
        ) from error
    finally:
        cleanup_lock.release()
        with suppress(OSError):
            cleanup_lock.path.unlink(missing_ok=True)
    try:
        shutil.rmtree(tombstone)
    except OSError as error:
        raise _storage_error(
            DiagnosticCode.STORAGE_IO_ERROR,
            "The isolated staging job could not be removed safely.",
        ) from error
    return CleanupReport(removed=(job,), skipped=())


def sweep_stale(
    caller_root: Path,
    *,
    older_than_s: float,
    now_s: float | None = None,
) -> CleanupReport:
    """Remove abandoned staging jobs and pointer temporaries older than `older_than_s`.

    Useful after crashes or forced process termination. Young, live or link-like entries are
    skipped and reported; published generations are never removed. `now_s` overrides the current
    time (mainly for testing).
    """
    if older_than_s < 0:
        raise ValueError("Stale age must not be negative")
    root = caller_root.resolve(strict=True)
    now = time.time() if now_s is None else now_s
    removed: list[Path] = []
    skipped: list[Path] = []
    for job in sorted(root.glob("sample-*/jobs/*.staging")):
        if _is_link_like(job) or now - job.stat().st_mtime < older_than_s or _job_is_live(job):
            skipped.append(job)
            continue
        try:
            report = cleanup_job(root, job)
        except StorageError as error:
            if error.diagnostic.code in {
                DiagnosticCode.RESULT_JOB_ACTIVE,
                DiagnosticCode.PATH_INVALID,
                DiagnosticCode.PATH_OUTSIDE_ROOT,
            }:
                skipped.append(job)
                continue
            raise
        removed.extend(report.removed)
        skipped.extend(report.skipped)
    for pointer_temp in sorted(root.glob("sample-*/current.*.tmp")):
        try:
            sample, pointer = _recognized_pointer_temp(root, pointer_temp)
        except (OSError, StorageError):
            skipped.append(pointer_temp)
            continue
        publish_lock = _FileLock(sample / ".publish.lock")
        try:
            publish_lock.acquire()
        except StorageError as error:
            if error.diagnostic.code is DiagnosticCode.RESULT_PUBLICATION_CONFLICT:
                skipped.append(pointer_temp)
                continue
            raise
        try:
            # Revalidate identity, ownership and age while holding the exact publisher lock.
            _, pointer = _recognized_pointer_temp(root, pointer)
            if not pointer.is_file() or now - pointer.stat().st_mtime < older_than_s:
                skipped.append(pointer_temp)
                continue
            pointer.unlink()
            removed.append(pointer_temp)
        except (FileNotFoundError, StorageError):
            skipped.append(pointer_temp)
        except OSError as error:
            raise _storage_error(
                DiagnosticCode.STORAGE_IO_ERROR,
                "A stale pointer temporary could not be removed safely.",
            ) from error
        finally:
            publish_lock.release()
    return CleanupReport(removed=tuple(removed), skipped=tuple(skipped))
