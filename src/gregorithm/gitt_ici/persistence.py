"""Atomic persistence and long-table export source for GITT/ICI snapshots."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from gregorithm.contracts import DependencyRef, JsonValue, OperationId, ResultRef
from gregorithm.identity import hash_dependencies, hash_parameters
from gregorithm.progress import CancelToken, check_cancelled
from gregorithm.storage import GenerationManifest, ResultStore, open_result

from .contracts import DelacourtResult, DiffusionMethod, DiffusionResult, OcvSource
from .quality import IndicatorColor, QualityAssessment
from .review import AnalysisRecord, EventCoordinates, ResultIdentity, ResultSnapshot, ResultState

GITT_ICI_RESULTS_SCHEMA = "gregorithm.gitt_ici.results.long.v1"
GITT_ICI_TABLE_PATH = "tables/gitt_ici_results.parquet"
GITT_ICI_ACCEPTED_TABLE_PATH = "tables/gitt_ici_accepted.parquet"

_ARROW_SCHEMA = pa.schema(
    [
        ("event_id", pa.string()),
        ("method", pa.string()),
        ("variant", pa.string()),
        ("ocv_source", pa.string()),
        ("state", pa.string()),
        ("stale", pa.bool_()),
        ("diffusion_m2_s", pa.float64()),
        ("standard_uncertainty_m2_s", pa.float64()),
        ("cycle", pa.int64()),
        ("half_cycle", pa.int64()),
        ("polarity", pa.int64()),
        ("capacity_start_ah", pa.float64()),
        ("capacity_end_ah", pa.float64()),
        ("quality_overall", pa.string()),
        ("quality_statistical", pa.string()),
        ("quality_physical", pa.string()),
        ("quality_rule_version", pa.string()),
        ("calculation_fingerprint", pa.string()),
        ("error", pa.string()),
        ("dependencies_json", pa.string()),
        ("result_json", pa.string()),
        ("quality_json", pa.string()),
        ("details_json", pa.string()),
    ]
)


@dataclass(frozen=True, slots=True)
class SnapshotPublicationRequest:
    """Identifiers, provenance and dependencies for one immutable snapshot generation."""

    caller_root: Path
    sample_id: str
    job_id: str
    generation_id: str
    parameters: Mapping[str, JsonValue]
    dependencies: tuple[DependencyRef, ...]


def _json_value(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _dumps(value: object) -> str:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _row(record: AnalysisRecord) -> dict[str, object]:
    coordinates = record.coordinates
    quality = record.quality
    return {
        "event_id": record.identity.event_id,
        "method": record.identity.method.value,
        "variant": record.identity.variant,
        "ocv_source": record.identity.ocv_source.value,
        "state": record.state.value,
        "stale": record.stale,
        "diffusion_m2_s": record.diffusion_m2_s,
        "standard_uncertainty_m2_s": None
        if record.result is None
        else record.result.standard_uncertainty_m2_s,
        "cycle": None if coordinates is None else coordinates.cycle,
        "half_cycle": None if coordinates is None else coordinates.half_cycle,
        "polarity": None if coordinates is None else coordinates.polarity,
        "capacity_start_ah": None if coordinates is None else coordinates.capacity_start_ah,
        "capacity_end_ah": None if coordinates is None else coordinates.capacity_end_ah,
        "quality_overall": None if quality is None else quality.overall.value,
        "quality_statistical": None if quality is None else quality.statistical.value,
        "quality_physical": None if quality is None else quality.physical.value,
        "quality_rule_version": None if quality is None else quality.rule_version,
        "calculation_fingerprint": record.calculation_fingerprint,
        "error": record.error,
        "dependencies_json": _dumps(record.dependencies),
        "result_json": None if record.result is None else _dumps(record.result),
        "quality_json": None if quality is None else _dumps(quality),
        "details_json": _dumps(record.details),
    }


def _table(records: Sequence[AnalysisRecord], *, export_safe: bool = False) -> pa.Table:
    """Build the stable long table, preserving technical records as well as accepted D values."""
    rows = [_row(record) for record in records]
    if export_safe:
        string_names = {field.name for field in _ARROW_SCHEMA if pa.types.is_string(field.type)}
        for row in rows:
            for name in string_names:
                if row[name] is None:
                    row[name] = ""
    return pa.Table.from_pylist(rows, schema=_ARROW_SCHEMA)


def publish_snapshot(
    snapshot: ResultSnapshot,
    request: SnapshotPublicationRequest,
    *,
    cancel: CancelToken | None = None,
) -> ResultRef:
    """Publish a complete snapshot atomically; cancellation never changes ``current``."""
    check_cancelled(cancel, OperationId.GITT_ICI.value)
    store = ResultStore(request.caller_root)
    staging = store.begin_job(request.sample_id, request.job_id, request.generation_id)
    try:
        table_path = staging.path / GITT_ICI_TABLE_PATH
        table_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(_table(snapshot.records), table_path, compression="zstd")
        accepted_path = staging.path / GITT_ICI_ACCEPTED_TABLE_PATH
        pq.write_table(
            _table(
                tuple(record for record in snapshot.exportable_records() if record.has_diffusion),
                export_safe=True,
            ),
            accepted_path,
            compression="zstd",
        )
        check_cancelled(cancel, OperationId.GITT_ICI.value)
        manifest = staging.build_manifest(
            operation=OperationId.GITT_ICI,
            parameter_hash=hash_parameters(request.parameters),
            dependency_hash=hash_dependencies(request.dependencies),
            dependencies=request.dependencies,
            content_hash=snapshot.fingerprint,
            row_counts={
                GITT_ICI_TABLE_PATH: len(snapshot.records),
                GITT_ICI_ACCEPTED_TABLE_PATH: sum(
                    record.exportable and record.has_diffusion for record in snapshot.records
                ),
            },
            logical_schemas={
                GITT_ICI_TABLE_PATH: GITT_ICI_RESULTS_SCHEMA,
                GITT_ICI_ACCEPTED_TABLE_PATH: GITT_ICI_RESULTS_SCHEMA,
            },
        )
        return store.publish(staging, manifest, cancel=cancel)
    finally:
        staging.close()


def _tuple_rows(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise ValueError("dependencies payload must be a list")
    return tuple((str(item[0]), str(item[1])) for item in value)


def _manifest(path: Path) -> GenerationManifest:
    payload = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be an object")
    return GenerationManifest.from_dict(payload)


def _result(method: DiffusionMethod, payload: dict[str, Any]) -> DiffusionResult | DelacourtResult:
    if method is DiffusionMethod.GITT_SPHERE_FULL:
        for name in ("source_rows", "relative_times_s", "fitted_voltage_v", "residuals_v"):
            payload[name] = tuple(payload[name])
        return DelacourtResult(**payload)
    payload["method"] = DiffusionMethod(payload["method"])
    return DiffusionResult(**payload)


def load_snapshot(result_path: Path) -> ResultSnapshot:
    """Reopen and checksum-verify a published GITT/ICI snapshot."""
    info = open_result(result_path)
    if info.operation is not OperationId.GITT_ICI:
        raise ValueError("result generation is not a GITT/ICI snapshot")
    manifest = _manifest(info.ref.path)
    schemas = dict(manifest.logical_schemas)
    if schemas.get(GITT_ICI_TABLE_PATH) != GITT_ICI_RESULTS_SCHEMA:
        raise ValueError("unsupported GITT/ICI logical schema")
    records: list[AnalysisRecord] = []
    for row in pq.read_table(info.ref.path / GITT_ICI_TABLE_PATH).to_pylist():
        method = DiffusionMethod(row["method"])
        quality_payload = None if row["quality_json"] is None else json.loads(row["quality_json"])
        quality = None
        if quality_payload is not None:
            quality_payload["method"] = DiffusionMethod(quality_payload["method"])
            for name in ("statistical", "physical", "overall"):
                quality_payload[name] = IndicatorColor(quality_payload[name])
            quality_payload["metrics"] = tuple(tuple(item) for item in quality_payload["metrics"])
            quality = QualityAssessment(**quality_payload)
        coordinates = None
        if row["cycle"] is not None:
            coordinates = EventCoordinates(
                row["cycle"],
                row["half_cycle"],
                row["polarity"],
                row["capacity_start_ah"],
                row["capacity_end_ah"],
            )
        result_payload = None if row["result_json"] is None else json.loads(row["result_json"])
        records.append(
            AnalysisRecord(
                ResultIdentity(
                    row["event_id"], method, row["variant"], OcvSource(row["ocv_source"])
                ),
                row["calculation_fingerprint"],
                ResultState(row["state"]),
                None if result_payload is None else _result(method, result_payload),
                quality,
                error=row["error"],
                dependencies=_tuple_rows(json.loads(row["dependencies_json"])),
                stale=row["stale"],
                details=json.loads(row["details_json"]),
                coordinates=coordinates,
            )
        )
    snapshot = ResultSnapshot(tuple(records))
    if manifest.content_hash != snapshot.fingerprint:
        raise ValueError("snapshot fingerprint does not match the manifest")
    return snapshot
