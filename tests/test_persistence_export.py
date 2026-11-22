"""Persistence, manifest, export and export-destination regression tests."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from gregorithm import (
    CancellationToken,
    DependencyRef,
    ExportFormat,
    NoAcceptedResultsError,
    OperationCancelled,
    SnapshotPublicationRequest,
    export_accepted_results,
    load_snapshot,
    open_current,
    publish_snapshot,
)
from gregorithm.gitt_ici import (
    AnalysisRecord,
    BatchTask,
    ComputeOutput,
    DelacourtResult,
    DiffusionMethod,
    DiffusionResult,
    EventCoordinates,
    IndicatorColor,
    OcvSource,
    QualityAssessment,
    ResultIdentity,
    ResultSnapshot,
    ResultState,
    run_batch,
)


def _record(event: str, state: ResultState, diffusion: float | None) -> AnalysisRecord:
    method = DiffusionMethod.GITT_VA
    return AnalysisRecord(
        ResultIdentity(event, method, "default", OcvSource.REST_SAMPLE),
        f"calculation-{event}",
        state,
        DiffusionResult(method, diffusion, 1e-16, 0.02, 10.0, 0.003, 0.01, None),
        QualityAssessment(
            method,
            IndicatorColor.GREEN,
            IndicatorColor.ORANGE,
            IndicatorColor.ORANGE,
            1.0,
            6.0,
            metrics=(("rmse_v", 1e-6),),
        ),
        dependencies=(("elaboration", "e" * 64),),
        details={"window": [1.0, 10.0], "rows": [1, 2, 3]},
        coordinates=EventCoordinates(1, 2, 1, 0.1, 0.2),
    )


def _request(root: Path, generation: str) -> SnapshotPublicationRequest:
    return SnapshotPublicationRequest(
        root,
        "sample",
        f"job-{generation}",
        generation,
        {"protocol": "gitt", "schema": 1},
        (DependencyRef("elaboration", "e" * 64),),
    )


def test_snapshot_round_trip_manifest_and_current(tmp_path: Path) -> None:
    snapshot = ResultSnapshot(
        (
            _record("accepted", ResultState.ACCEPTED, 2e-14),
            _record("draft", ResultState.PROVISIONAL, 3e-14),
        )
    )
    ref = publish_snapshot(snapshot, _request(tmp_path / "store", "generation-1"))

    assert load_snapshot(ref.path) == snapshot
    current = open_current(tmp_path / "store" / "sample-sample")
    assert current.ref.path == ref.path
    manifest = (ref.path / "manifest.json").read_text(encoding="utf-8")
    assert "gregorithm.gitt_ici.results.long.v1" in manifest
    assert '"tables/gitt_ici_accepted.parquet":1' in manifest


def test_export_long_csv_and_parquet_have_equivalent_schema(tmp_path: Path) -> None:
    snapshot = ResultSnapshot(
        (
            _record("accepted", ResultState.ACCEPTED, 2e-14),
            _record("draft", ResultState.PROVISIONAL, 3e-14),
        )
    )
    ref = publish_snapshot(snapshot, _request(tmp_path / "store", "generation-1"))
    csv_path = tmp_path / "accepted.csv"
    parquet_path = tmp_path / "accepted.parquet"

    csv_receipt = export_accepted_results(ref.path, csv_path, ExportFormat.CSV)
    parquet_receipt = export_accepted_results(ref.path, parquet_path, ExportFormat.PARQUET)

    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    parquet = pq.read_table(parquet_path)
    assert csv_receipt.rows == parquet_receipt.rows == len(rows) == parquet.num_rows == 1
    assert tuple(rows[0]) == parquet_receipt.columns == csv_receipt.columns
    assert rows[0]["event_id"] == "accepted"
    assert rows[0]["state"] == "accepted"
    assert float(rows[0]["diffusion_m2_s"]) == 2e-14


def test_d80_creates_no_destination_or_parent(tmp_path: Path) -> None:
    snapshot = ResultSnapshot((_record("draft", ResultState.PROVISIONAL, 3e-14),))
    ref = publish_snapshot(snapshot, _request(tmp_path / "store", "generation-1"))
    destination = tmp_path / "not-created" / "empty.csv"

    with pytest.raises(NoAcceptedResultsError, match="no accepted current"):
        export_accepted_results(ref.path, destination, ExportFormat.CSV)

    assert not destination.exists()
    assert not destination.parent.exists()


def test_cancelled_publication_preserves_current(tmp_path: Path) -> None:
    store = tmp_path / "store"
    first = publish_snapshot(
        ResultSnapshot((_record("accepted", ResultState.ACCEPTED, 2e-14),)),
        _request(store, "generation-1"),
    )
    token = CancellationToken()
    token.cancel()

    with pytest.raises(OperationCancelled):
        publish_snapshot(
            ResultSnapshot((_record("other", ResultState.ACCEPTED, 4e-14),)),
            _request(store, "generation-2"),
            cancel=token,
        )

    assert open_current(store / "sample-sample").ref.path == first.path


def _delacourt(r_squared: float | None = 0.99) -> DelacourtResult:
    return DelacourtResult(
        1e-14,
        None,
        -1.0,
        0.1,
        5.0,
        0.0,
        (1, 2),
        (0.0, 1.0),
        (3.7, 3.71),
        (0.0, 0.0),
        1e-4,
        r_squared,
        1e-8,
        False,
        True,
        0.5,
        42,
    )


def test_delacourt_snapshot_round_trip(tmp_path: Path) -> None:
    method = DiffusionMethod.GITT_SPHERE_FULL
    record = AnalysisRecord(
        ResultIdentity("pulse", method, "fixed", OcvSource.REST_SAMPLE),
        "calculation-delacourt",
        ResultState.ACCEPTED,
        _delacourt(),
        None,
    )
    snapshot = ResultSnapshot((record,))
    ref = publish_snapshot(snapshot, _request(tmp_path / "store", "generation-1"))

    assert load_snapshot(ref.path) == snapshot


def test_tuple_details_reopen_equal_to_published_snapshot(tmp_path: Path) -> None:
    record = AnalysisRecord(
        ResultIdentity("pulse", DiffusionMethod.GITT_VA, "default", OcvSource.REST_SAMPLE),
        "calculation-pulse",
        ResultState.ACCEPTED,
        DiffusionResult(DiffusionMethod.GITT_VA, 2e-14, None, 0.02, 10.0, 0.003, 0.01, None),
        None,
        details={"window": (1.0, 10.0), "nested": [{"rows": (1, 2)}]},  # type: ignore[arg-type]
    )
    assert record.details == {"window": [1.0, 10.0], "nested": [{"rows": [1, 2]}]}
    snapshot = ResultSnapshot((record,))
    ref = publish_snapshot(snapshot, _request(tmp_path / "store", "generation-1"))

    assert load_snapshot(ref.path) == snapshot


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_values_cannot_enter_a_record(value: float) -> None:
    identity = ResultIdentity("pulse", DiffusionMethod.GITT_VA, "default", OcvSource.REST_SAMPLE)
    with pytest.raises(ValueError, match=r"result\.diffusion_m2_s must be finite"):
        _record("pulse", ResultState.REJECTED, value)
    with pytest.raises(ValueError, match=r"result\.r_squared must be finite"):
        AnalysisRecord(
            ResultIdentity(
                "pulse", DiffusionMethod.GITT_SPHERE_FULL, "fixed", OcvSource.REST_SAMPLE
            ),
            "calculation-delacourt",
            ResultState.PROVISIONAL,
            _delacourt(value),
            None,
        )
    with pytest.raises(ValueError, match=r"details\.stats\[1\] must be finite"):
        AnalysisRecord(
            identity,
            "calculation-pulse",
            ResultState.PROVISIONAL,
            DiffusionResult(DiffusionMethod.GITT_VA, 2e-14, None, 0.02, 10.0, 0.003, 0.01, None),
            None,
            details={"stats": [1.0, value]},
        )


def test_non_finite_task_output_becomes_error_record_and_snapshot_publishes(
    tmp_path: Path,
) -> None:
    def task(event: str, diffusion: float) -> BatchTask:
        result = DiffusionResult(
            DiffusionMethod.GITT_VA, diffusion, None, 0.02, 10.0, 0.003, 0.01, None
        )
        return BatchTask(
            ResultIdentity(event, DiffusionMethod.GITT_VA, "default", OcvSource.REST_SAMPLE),
            (("generation", "g" * 64),),
            {"event": event},
            lambda _cached: ComputeOutput(result),
        )

    outcome = run_batch(
        (task("finite", 2e-14), task("not-finite", math.nan)), previous=ResultSnapshot()
    )
    states = {record.identity.event_id: record for record in outcome.snapshot.records}

    assert outcome.recoverable_errors == 1
    assert states["finite"].state is ResultState.PROVISIONAL
    assert states["not-finite"].state is ResultState.ERROR
    assert "must be finite" in (states["not-finite"].error or "")
    ref = publish_snapshot(outcome.snapshot, _request(tmp_path / "store", "generation-1"))
    assert load_snapshot(ref.path) == outcome.snapshot
