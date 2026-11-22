"""Per-cycle Capacity, energy, efficiency, normalization and C-rate.

`analyze_capacity` reads an immutable elaboration result (never the original source) and
publishes a Capacity generation with:

- ``tables/capacity.parquet``: one row per cycle with the columns of `CAPACITY_SCHEMA`;
- ``tables/capacity_diagnostics.parquet``: for every null cell, the first causal
  `DiagnosticCode` (columns ``cycle``, ``column``, ``code``);
- ``reports/capacity.json``: scan counts and the selected C-rate reference capacity.

Phase capacity/energy is the net value of the labelled half-cycle (for a charge half-cycle,
charge minus discharge integral). Coulombic and energy efficiency are ``100 * second / first``
phase in the selected cycle order and are null when a phase is missing or the denominator is
exactly zero; values below 0 % or above 100 % are reported as computed. Cycle 0 (leading rest, if
any) reports its directional rest contributions and has null efficiencies.

Normalization uses only caller-supplied positive values: mAh/g and Wh/kg from the active mass in
mg, mAh/cm² and mWh/cm² from the electrode area in cm². C-rate is ``|I_CC| / Q_ref`` where
``Q_ref`` is the net second-phase capacity of `reference_cycle` (default 2) and ``I_CC`` is the
duration-weighted mean of the arithmetic mean current of every CC portion in the phase. A missing
or non-positive reference, or a phase without CC portions, gives a null C-rate. Missing values
are never replaced by zero.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sqlite3
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from . import __version__
from .contracts import (
    AnalysisDescriptor,
    AvailabilityDescriptor,
    DependencyRef,
    OperationId,
    OutputDescriptor,
    ParameterDescriptor,
    ParameterGroup,
    ParameterLevel,
    ResultRef,
)
from .diagnostics import DiagnosticCode
from .elaboration import CycleOrder, Step
from .identity import hash_dependencies, hash_parameters
from .progress import CancelToken, ProgressEvent, ProgressSink, check_cancelled, report_progress
from .storage import GenerationManifest, ResultStore, open_result

CAPACITY_SCHEMA = pa.schema(
    [
        ("cycle", pa.int32()),
        ("charge_capacity_ah", pa.float64()),
        ("discharge_capacity_ah", pa.float64()),
        ("coulombic_efficiency_pct", pa.float64()),
        ("charge_energy_wh", pa.float64()),
        ("discharge_energy_wh", pa.float64()),
        ("energy_efficiency_pct", pa.float64()),
        ("charge_c_rate", pa.float64()),
        ("discharge_c_rate", pa.float64()),
        ("charge_capacity_mah_g", pa.float64()),
        ("discharge_capacity_mah_g", pa.float64()),
        ("charge_capacity_mah_cm2", pa.float64()),
        ("discharge_capacity_mah_cm2", pa.float64()),
        ("charge_energy_wh_kg", pa.float64()),
        ("discharge_energy_wh_kg", pa.float64()),
        ("charge_energy_mwh_cm2", pa.float64()),
        ("discharge_energy_mwh_cm2", pa.float64()),
    ]
)
BASE_CAPACITY_SCHEMA = pa.schema(list(CAPACITY_SCHEMA)[:7])

DIAGNOSTIC_SCHEMA = pa.schema(
    [("cycle", pa.int32()), ("column", pa.string()), ("code", pa.string())]
)

_BASE_COLUMNS = (
    "cycle",
    "charge_capacity_ah",
    "discharge_capacity_ah",
    "coulombic_efficiency_pct",
    "charge_energy_wh",
    "discharge_energy_wh",
    "energy_efficiency_pct",
)
_OUTPUT_BATCH_ROWS = 4_096


@dataclass(frozen=True, slots=True)
class CapacityParameters:
    """Settings of the Capacity analysis.

    Attributes:
        cycle_order: Must equal the order used by the elaboration; it defines which phase is
            first for efficiencies and which phase is the C-rate reference.
        reference_cycle: Cycle (>= 1) whose second-phase capacity is the C-rate reference.
        active_material_mass_mg: Optional positive active-material mass in mg.
        electrode_area_cm2: Optional positive electrode area in cm².
    """

    cycle_order: CycleOrder = CycleOrder.CHARGE_DISCHARGE
    reference_cycle: int = 2
    active_material_mass_mg: float | None = None
    electrode_area_cm2: float | None = None

    def validate(self) -> None:
        """Raise ValueError for a reference cycle < 1 or invalid mass/area."""
        if self.reference_cycle < 1:
            raise ValueError("reference_cycle must be a positive cycle identifier")
        for name, value in (
            ("active_material_mass_mg", self.active_material_mass_mg),
            ("electrode_area_cm2", self.electrode_area_cm2),
        ):
            if value is not None and (
                not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{name} must be absent or a positive finite number")

    def as_dict(self) -> dict[str, Any]:
        """Return the parameters as a JSON-compatible dictionary."""
        return {
            "cycle_order": self.cycle_order.value,
            "reference_cycle": self.reference_cycle,
            "active_material_mass_mg": self.active_material_mass_mg,
            "electrode_area_cm2": self.electrode_area_cm2,
        }


@dataclass(frozen=True, slots=True)
class CapacityRequest:
    """Parameters of `analyze_capacity`.

    Attributes:
        elaboration_path: Elaboration generation directory (`ElaborationResult.ref.path`).
        elaboration_fingerprint: `ElaborationResult.fingerprint`; it must match the manifest.
        caller_root, sample_id, job_id, generation_id: Destination in the result store.
        parameters: `CapacityParameters`.
        batch_size: Rows per streaming batch.
    """

    elaboration_path: Path
    elaboration_fingerprint: str
    caller_root: Path
    sample_id: str
    job_id: str
    generation_id: str
    parameters: CapacityParameters = CapacityParameters()
    batch_size: int = 65_536

    def validate(self) -> None:
        """Validate parameters, batch size and the fingerprint format."""
        self.parameters.validate()
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len(self.elaboration_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.elaboration_fingerprint
        ):
            raise ValueError("elaboration_fingerprint must be lowercase SHA-256 hex")


@dataclass(frozen=True, slots=True)
class CapacityResult:
    """Published Capacity result.

    Attributes:
        ref: Reference to the Capacity generation.
        cycles: Number of cycle rows.
        fingerprint: Fingerprint of the whole result (input plus all parameters).
        base_fingerprint: Fingerprint of capacity/energy/efficiency (independent of mass, area
            and reference cycle).
        mass_fingerprint, area_fingerprint: Fingerprints of the normalized values, or None.
        c_rate_fingerprint: Fingerprint of the C-rate values (depends on the reference cycle).
    """

    ref: ResultRef
    cycles: int
    fingerprint: str
    base_fingerprint: str
    mass_fingerprint: str | None
    area_fingerprint: str | None
    c_rate_fingerprint: str


@dataclass(slots=True)
class _CCPortion:
    """Running sum, count and first/last time of one contiguous CC portion."""

    total: np.longdouble
    count: int
    first_time: float
    last_time: float


@dataclass(slots=True)
class _CCPhase:
    """Duration-weighted accumulation of CC portion mean currents within one phase."""

    weighted_current: np.longdouble = field(default_factory=lambda: np.longdouble(0.0))
    duration: np.longdouble = field(default_factory=lambda: np.longdouble(0.0))
    portion: _CCPortion | None = None

    def close_portion(self) -> None:
        """Add the open CC portion (mean current times duration), if any."""
        if self.portion is None:
            return
        duration = np.longdouble(self.portion.last_time - self.portion.first_time)
        mean = self.portion.total / np.longdouble(self.portion.count)
        self.weighted_current += duration * mean
        self.duration += duration
        self.portion = None

    def value(self) -> float | None:
        """Return the duration-weighted CC current, or None without CC duration."""
        self.close_portion()
        if self.duration <= 0:
            return None
        return float(self.weighted_current / self.duration)


def _fingerprint(domain: bytes, dependency: str, parameters: dict[str, Any]) -> str:
    """Hash a domain, the library version, a dependency and parameters."""
    digest = hashlib.sha256(domain + b"\0")
    digest.update(__version__.encode("utf-8"))
    digest.update(b"\0")
    digest.update(dependency.encode("ascii"))
    digest.update(b"\0")
    digest.update(hash_parameters(parameters).encode("ascii"))
    return digest.hexdigest()


def _safe_normalize(value: float | None, factor: int, basis: float | None) -> float | None:
    """Return value * factor / basis with exact intermediates, or None."""
    if value is None or basis is None:
        return None
    with localcontext() as context:
        context.prec = 80
        exact = Decimal.from_float(value) * Decimal(factor) / Decimal.from_float(basis)
        try:
            result = float(exact)
        except OverflowError:
            return None
    return result if math.isfinite(result) else None


def _ratio(second: float | None, first: float | None) -> float | None:
    """Return 100 * second / first with exact intermediates; None if undefined."""
    if first is None or second is None or first == 0:
        return None
    with localcontext() as context:
        context.prec = 100
        exact = Decimal.from_float(second) / Decimal.from_float(first) * Decimal(100)
        try:
            value = float(exact)
        except OverflowError:
            return None
    return value if math.isfinite(value) else None


def _validate_upstream(request: CapacityRequest, root: Path) -> None:
    """Check the elaboration operation, fingerprint and cycle order."""
    info = open_result(root)
    if info.operation is not OperationId.INGEST_AND_ELABORATE:
        raise ValueError("Capacity requires an immutable elaboration result")
    manifest_data = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest_data, dict):
        raise ValueError("Elaboration manifest root must be an object")
    manifest = GenerationManifest.from_dict(manifest_data)
    if manifest.content_hash != request.elaboration_fingerprint:
        raise ValueError("elaboration_fingerprint does not match the elaboration manifest")
    report_data = json.loads((root / "reports" / "elaboration.json").read_text(encoding="utf-8"))
    if not isinstance(report_data, dict) or "cycle_order" not in report_data:
        raise ValueError("Elaboration report lacks cycle_order provenance; run elaborate() again")
    try:
        upstream_order = CycleOrder(str(report_data["cycle_order"]))
    except ValueError as error:
        raise ValueError("Elaboration report has invalid cycle_order provenance") from error
    if upstream_order is not request.parameters.cycle_order:
        raise ValueError("Capacity cycle_order conflicts with the elaboration result")


def _direction(half_cycle: int, first_direction: int) -> int:
    """Return +1 (charge) or -1 (discharge) for a half-cycle number."""
    return first_direction if half_cycle % 2 else -first_direction


def _phase_from_half(record: dict[str, Any]) -> tuple[str, float, float | None]:
    """Return the phase name and net capacity/energy of one half-cycle record."""
    direction = _direction(int(record["half_cycle"]), int(record["first_direction"]))
    if direction > 0:
        capacity = float(record["charge_capacity_ah"]) - float(record["discharge_capacity_ah"])
        energy = (
            None
            if record["charge_energy_wh"] is None or record["discharge_energy_wh"] is None
            else float(record["charge_energy_wh"]) - float(record["discharge_energy_wh"])
        )
        return "charge", capacity, energy
    capacity = float(record["discharge_capacity_ah"]) - float(record["charge_capacity_ah"])
    energy = (
        None
        if record["charge_energy_wh"] is None or record["discharge_energy_wh"] is None
        else float(record["discharge_energy_wh"]) - float(record["charge_energy_wh"])
    )
    return "discharge", capacity, energy


def _base_row(cycle: int, halves: list[dict[str, Any]], order: CycleOrder) -> dict[str, Any]:
    """Build the capacity/energy/efficiency row of one cycle from its half-cycles."""
    if cycle == 0:
        half = halves[0]
        return {
            "cycle": 0,
            "charge_capacity_ah": float(half["charge_capacity_ah"]),
            "discharge_capacity_ah": float(half["discharge_capacity_ah"]),
            "coulombic_efficiency_pct": None,
            "charge_energy_wh": half["charge_energy_wh"],
            "discharge_energy_wh": half["discharge_energy_wh"],
            "energy_efficiency_pct": None,
        }
    values: dict[str, tuple[float, float | None]] = {}
    for half in halves:
        phase, capacity, energy = _phase_from_half(half)
        values[phase] = (capacity, energy)
    charge = values.get("charge")
    discharge = values.get("discharge")
    charge_capacity = None if charge is None else charge[0]
    discharge_capacity = None if discharge is None else discharge[0]
    charge_energy = None if charge is None else charge[1]
    discharge_energy = None if discharge is None else discharge[1]
    first_capacity, second_capacity = (
        (charge_capacity, discharge_capacity)
        if order is CycleOrder.CHARGE_DISCHARGE
        else (discharge_capacity, charge_capacity)
    )
    first_energy, second_energy = (
        (charge_energy, discharge_energy)
        if order is CycleOrder.CHARGE_DISCHARGE
        else (discharge_energy, charge_energy)
    )
    return {
        "cycle": cycle,
        "charge_capacity_ah": charge_capacity,
        "discharge_capacity_ah": discharge_capacity,
        "coulombic_efficiency_pct": _ratio(second_capacity, first_capacity),
        "charge_energy_wh": charge_energy,
        "discharge_energy_wh": discharge_energy,
        "energy_efficiency_pct": _ratio(second_energy, first_energy),
    }


def capacity_rows_from_halves(
    halves: list[dict[str, Any]], order: CycleOrder
) -> list[dict[str, Any]]:
    """Aggregate half-cycle totals into per-cycle Capacity rows (pure).

    `halves` must be ordered by half-cycle and contain ``cycle``, ``half_cycle``,
    ``first_direction`` and the four directional capacity/energy totals, as in
    ``indexes/half_cycles.parquet`` of an elaboration result. Returns dictionaries with the seven
    base Capacity columns. Useful for verification and small in-memory analyses.
    """
    rows: list[dict[str, Any]] = []
    current_cycle: int | None = None
    group: list[dict[str, Any]] = []
    for half in halves:
        cycle = int(half["cycle"])
        if current_cycle is not None and cycle != current_cycle:
            rows.append(_base_row(current_cycle, group, order))
            group = []
        current_cycle = cycle
        group.append(half)
    if current_cycle is not None:
        rows.append(_base_row(current_cycle, group, order))
    return rows


def _write_base_rows(
    half_path: Path,
    output: Path,
    order: CycleOrder,
    batch_size: int,
    cancel: CancelToken | None,
    reference_cycle: int,
) -> tuple[int, float | None, int]:
    """Stream half-cycle totals into base Capacity rows and find the reference."""
    writer: pq.ParquetWriter | None = None
    count = 0
    selected_reference: float | None = None
    scanned = 0
    current_cycle: int | None = None
    group: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        """Write pending rows to the output Parquet file."""
        nonlocal writer
        if not pending:
            return
        table = pa.Table.from_pylist(pending, schema=BASE_CAPACITY_SCHEMA)
        if writer is None:
            writer = pq.ParquetWriter(output, table.schema, compression="zstd")
        writer.write_table(table)
        pending.clear()

    def emit() -> None:
        """Finish the current cycle row and record the C-rate reference."""
        nonlocal count, selected_reference
        if current_cycle is None:
            return
        row = _base_row(current_cycle, group, order)
        pending.append(row)
        if len(pending) >= _OUTPUT_BATCH_ROWS:
            flush()
        count += 1
        if current_cycle == reference_cycle:
            selected_reference = row[
                "discharge_capacity_ah"
                if order is CycleOrder.CHARGE_DISCHARGE
                else "charge_capacity_ah"
            ]

    try:
        file = pq.ParquetFile(half_path)
        for batch in file.iter_batches(batch_size=batch_size):
            check_cancelled(cancel, OperationId.CAPACITY.value)
            scanned += batch.num_rows
            for row in pa.Table.from_batches([batch]).to_pylist():
                cycle = int(row["cycle"])
                if current_cycle is not None and cycle != current_cycle:
                    emit()
                    group = []
                current_cycle = cycle
                group.append(row)
        emit()
        flush()
    finally:
        if writer is not None:
            writer.close()
    return count, selected_reference, scanned


def _write_cc_rows(
    points_path: Path,
    output: Path,
    batch_size: int,
    cancel: CancelToken | None,
    progress: ProgressSink | None,
    total_rows: int,
    first_direction: int,
) -> tuple[int, int]:
    """Stream elaborated points and compute per-cycle CC currents for C-rate.

    Per-phase partial results are spilled to a private temporary SQLite file so memory stays
    bounded for any number of cycles; the scratch directory is removed afterwards.
    """
    import os
    import shutil
    import tempfile

    schema = pa.schema(
        [("cycle", pa.int32()), ("charge_cc_a", pa.float64()), ("discharge_cc_a", pa.float64())]
    )
    writer: pq.ParquetWriter | None = None
    current_cycle: int | None = None
    current_half: int | None = None
    phase = _CCPhase()
    scanned = 0
    written = 0
    pending: list[dict[str, Any]] = []
    scratch_directory: Path | None = None
    database_path: Path | None = None
    database: sqlite3.Connection | None = None
    sequence = 0

    def flush() -> None:
        """Write pending CC rows to the output Parquet file."""
        nonlocal writer
        if not pending:
            return
        table = pa.Table.from_pylist(pending, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(output, schema, compression="zstd")
        writer.write_table(table)
        pending.clear()

    def close_half() -> None:
        """Store the finished half-cycle CC aggregate in the scratch database."""
        nonlocal phase, sequence
        phase.close_portion()
        if current_cycle is not None and current_half is not None and phase.duration > 0:
            assert database is not None
            direction = first_direction * (1 if current_half % 2 else -1)
            database.execute(
                "INSERT INTO parts VALUES (?, ?, ?, ?, ?)",
                (
                    sequence,
                    current_cycle,
                    direction,
                    str(phase.weighted_current),
                    str(phase.duration),
                ),
            )
            sequence += 1
        phase = _CCPhase()

    def record_cycle(cycle: int) -> None:
        """Remember that a cycle exists, even without CC portions."""
        assert database is not None
        database.execute("INSERT OR IGNORE INTO cycles VALUES (?)", (cycle,))

    previous_cc = False
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        scratch_directory = Path(tempfile.mkdtemp(prefix=".gregorithm-cc-", dir=output.parent))
        if (
            scratch_directory.is_symlink()
            or scratch_directory.resolve().parent != output.parent.resolve()
        ):
            raise RuntimeError("Private Capacity scratch escaped its authorized directory")
        database_path = scratch_directory / "parts.sqlite3"
        descriptor = os.open(database_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        os.close(descriptor)
        database = sqlite3.connect(database_path)
        database.execute("CREATE TABLE cycles (cycle INTEGER PRIMARY KEY)")
        database.execute(
            "CREATE TABLE parts (sequence INTEGER PRIMARY KEY, cycle INTEGER, direction INTEGER, "
            "weighted TEXT, duration TEXT)"
        )
        file = pq.ParquetFile(points_path)
        for batch in file.iter_batches(
            batch_size=batch_size, columns=["time_s", "current_a", "cycle", "half_cycle", "step"]
        ):
            check_cancelled(cancel, OperationId.CAPACITY.value)
            table = pa.Table.from_batches([batch])
            time = table["time_s"].to_numpy(zero_copy_only=False)
            current = table["current_a"].to_numpy(zero_copy_only=False)
            cycles = table["cycle"].to_numpy(zero_copy_only=False)
            halves = table["half_cycle"].to_numpy(zero_copy_only=False)
            cc = table["step"].to_numpy(zero_copy_only=False) == int(Step.CC)
            starts = np.r_[
                0,
                np.flatnonzero(
                    (cycles[1:] != cycles[:-1]) | (halves[1:] != halves[:-1]) | (cc[1:] != cc[:-1])
                )
                + 1,
            ]
            ends = np.r_[starts[1:], len(time)]
            for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
                cycle = int(cycles[start])
                half = int(halves[start])
                record_cycle(cycle)
                if current_cycle is not None and cycle != current_cycle:
                    close_half()
                    current_half = None
                    previous_cc = False
                elif current_half is not None and half != current_half:
                    close_half()
                    previous_cc = False
                current_cycle, current_half = cycle, half
                if bool(cc[start]):
                    total = np.sum(current[start:end], dtype=np.longdouble)
                    if previous_cc and phase.portion is not None:
                        phase.portion.total += total
                        phase.portion.count += end - start
                        phase.portion.last_time = float(time[end - 1])
                    else:
                        phase.portion = _CCPortion(
                            total, end - start, float(time[start]), float(time[end - 1])
                        )
                    previous_cc = True
                else:
                    phase.close_portion()
                    previous_cc = False
            scanned += batch.num_rows
            report_progress(
                progress,
                ProgressEvent(
                    OperationId.CAPACITY,
                    "scan-cc",
                    total_rows + scanned,
                    total_rows * 3,
                    "work_rows",
                ),
            )
        close_half()
        database.commit()
        part_rows = iter(
            database.execute(
                "SELECT c.cycle, p.direction, p.weighted, p.duration "
                "FROM cycles c LEFT JOIN parts p ON p.cycle = c.cycle "
                "ORDER BY c.cycle, p.sequence"
            )
        )
        row = next(part_rows, None)
        while row is not None:
            cycle = int(row[0])
            weighted = {1: np.longdouble(0), -1: np.longdouble(0)}
            duration = {1: np.longdouble(0), -1: np.longdouble(0)}
            while row is not None and int(row[0]) == cycle:
                if row[1] is not None:
                    direction = int(row[1])
                    weighted[direction] += np.longdouble(row[2])
                    duration[direction] += np.longdouble(row[3])
                row = next(part_rows, None)
            pending.append(
                {
                    "cycle": cycle,
                    "charge_cc_a": (None if duration[1] <= 0 else float(weighted[1] / duration[1])),
                    "discharge_cc_a": (
                        None if duration[-1] <= 0 else float(weighted[-1] / duration[-1])
                    ),
                }
            )
            written += 1
            if len(pending) >= _OUTPUT_BATCH_ROWS:
                flush()
        flush()
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            try:
                if database is not None:
                    database.close()
            finally:
                if scratch_directory is not None and scratch_directory.exists():
                    shutil.rmtree(scratch_directory)
    return written, scanned


class _RowCursor:
    """Row-by-row reader over a Parquet file, loading one batch at a time."""

    def __init__(self, path: Path, batch_size: int) -> None:
        """Open `path` for batched iteration."""
        self._iterator = iter(pq.ParquetFile(path).iter_batches(batch_size=batch_size))
        self._rows: list[dict[str, Any]] = []
        self._position = 0

    def next(self) -> dict[str, Any] | None:
        """Return the next row as a dictionary, or None at the end."""
        if self._position == len(self._rows):
            try:
                self._rows = pa.Table.from_batches([next(self._iterator)]).to_pylist()
            except StopIteration:
                return None
            self._position = 0
        row = self._rows[self._position]
        self._position += 1
        return row


def _reason(
    row: dict[str, Any],
    column: str,
    reference_valid: bool,
    mass: float | None,
    area: float | None,
    order: CycleOrder,
) -> DiagnosticCode | None:
    """Return the first causal diagnostic code explaining a null Capacity cell."""
    if row[column] is not None:
        return None
    if column.endswith("_c_rate"):
        if not reference_valid:
            return DiagnosticCode.C_RATE_REFERENCE_UNAVAILABLE
        if int(row["cycle"]) == 0:
            return DiagnosticCode.C_RATE_CC_UNAVAILABLE
        phase = column.removesuffix("_c_rate")
        if row[f"{phase}_capacity_ah"] is None:
            return DiagnosticCode.CAPACITY_MISSING_PHASE
        return DiagnosticCode.C_RATE_CC_UNAVAILABLE
    if "mah_g" in column or "wh_kg" in column:
        source = column.replace("_mah_g", "_ah").replace("_wh_kg", "_wh")
        if row.get(source) is None:
            phase = source.split("_")[0]
            if source.endswith("energy_wh") and row[f"{phase}_capacity_ah"] is not None:
                return DiagnosticCode.INTEGRATION_ENERGY_UNAVAILABLE
            return DiagnosticCode.CAPACITY_MISSING_PHASE
        if mass is None:
            return DiagnosticCode.NORMALIZATION_BASIS_MISSING
        return DiagnosticCode.NUMERIC_RESULT_UNREPRESENTABLE
    if "mah_cm2" in column or "mwh_cm2" in column:
        source = column.replace("_mah_cm2", "_ah").replace("_mwh_cm2", "_wh")
        if row.get(source) is None:
            phase = source.split("_")[0]
            if source.endswith("energy_wh") and row[f"{phase}_capacity_ah"] is not None:
                return DiagnosticCode.INTEGRATION_ENERGY_UNAVAILABLE
            return DiagnosticCode.CAPACITY_MISSING_PHASE
        if area is None:
            return DiagnosticCode.NORMALIZATION_BASIS_MISSING
        return DiagnosticCode.NUMERIC_RESULT_UNREPRESENTABLE
    if column in ("charge_energy_wh", "discharge_energy_wh"):
        phase = column.split("_")[0]
        return (
            DiagnosticCode.CAPACITY_MISSING_PHASE
            if row[f"{phase}_capacity_ah"] is None
            else DiagnosticCode.INTEGRATION_ENERGY_UNAVAILABLE
        )
    if column in ("charge_capacity_ah", "discharge_capacity_ah"):
        return DiagnosticCode.CAPACITY_MISSING_PHASE
    if int(row["cycle"]) == 0:
        return DiagnosticCode.CAPACITY_NO_ACTIVE_PAIR_CYCLE_ZERO
    first = "charge" if order is CycleOrder.CHARGE_DISCHARGE else "discharge"
    second = "discharge" if first == "charge" else "charge"
    if column == "coulombic_efficiency_pct":
        if row[f"{first}_capacity_ah"] is None or row[f"{second}_capacity_ah"] is None:
            return DiagnosticCode.CAPACITY_MISSING_PHASE
        denominator = row[f"{first}_capacity_ah"]
    else:
        if row[f"{first}_capacity_ah"] is None or row[f"{second}_capacity_ah"] is None:
            return DiagnosticCode.CAPACITY_MISSING_PHASE
        if row[f"{first}_energy_wh"] is None or row[f"{second}_energy_wh"] is None:
            return DiagnosticCode.INTEGRATION_ENERGY_UNAVAILABLE
        denominator = row[f"{first}_energy_wh"]
    if denominator == 0:
        return DiagnosticCode.CAPACITY_ZERO_DENOMINATOR
    return DiagnosticCode.NUMERIC_RESULT_UNREPRESENTABLE


def _finalize_rows(
    base_path: Path,
    cc_path: Path,
    output: Path,
    diagnostics_path: Path,
    parameters: CapacityParameters,
    reference: float | None,
    batch_size: int,
    cancel: CancelToken | None,
    progress: ProgressSink | None,
    total_rows: int,
) -> int:
    """Merge base and CC rows, add C-rate and normalization, write tables."""
    base = _RowCursor(base_path, batch_size)
    cc = _RowCursor(cc_path, batch_size)
    writer = pq.ParquetWriter(output, CAPACITY_SCHEMA, compression="zstd")
    diagnostic_writer = pq.ParquetWriter(diagnostics_path, DIAGNOSTIC_SCHEMA, compression="zstd")
    reference_valid = reference is not None and reference > 0 and math.isfinite(reference)
    count = 0
    pending_rows: list[dict[str, Any]] = []
    pending_diagnostics: list[dict[str, Any]] = []

    def flush() -> None:
        """Write pending Capacity and diagnostic rows."""
        if pending_rows:
            writer.write_table(pa.Table.from_pylist(pending_rows, schema=CAPACITY_SCHEMA))
            pending_rows.clear()
        if pending_diagnostics:
            diagnostic_writer.write_table(
                pa.Table.from_pylist(pending_diagnostics, schema=DIAGNOSTIC_SCHEMA)
            )
            pending_diagnostics.clear()

    try:
        while (base_row := base.next()) is not None:
            check_cancelled(cancel, OperationId.CAPACITY.value)
            cc_row = cc.next()
            if cc_row is None or cc_row["cycle"] != base_row["cycle"]:
                raise RuntimeError("Capacity base and CC streams are not aligned")
            cycle = int(base_row["cycle"])
            charge_cc = cc_row["charge_cc_a"]
            discharge_cc = cc_row["discharge_cc_a"]
            row = dict(base_row)
            row["charge_c_rate"] = (
                None
                if cycle == 0 or not reference_valid or charge_cc is None
                else abs(charge_cc) / reference
            )
            row["discharge_c_rate"] = (
                None
                if cycle == 0 or not reference_valid or discharge_cc is None
                else abs(discharge_cc) / reference
            )
            for phase in ("charge", "discharge"):
                q = row[f"{phase}_capacity_ah"]
                e = row[f"{phase}_energy_wh"]
                row[f"{phase}_capacity_mah_g"] = _safe_normalize(
                    q, 1_000_000, parameters.active_material_mass_mg
                )
                row[f"{phase}_capacity_mah_cm2"] = _safe_normalize(
                    q, 1_000, parameters.electrode_area_cm2
                )
                row[f"{phase}_energy_wh_kg"] = _safe_normalize(
                    e, 1_000_000, parameters.active_material_mass_mg
                )
                row[f"{phase}_energy_mwh_cm2"] = _safe_normalize(
                    e, 1_000, parameters.electrode_area_cm2
                )
            pending_rows.append(row)
            diagnostic_rows = [
                {"cycle": cycle, "column": column, "code": reason.value}
                for column in CAPACITY_SCHEMA.names[1:]
                if (
                    reason := _reason(
                        row,
                        column,
                        reference_valid,
                        parameters.active_material_mass_mg,
                        parameters.electrode_area_cm2,
                        parameters.cycle_order,
                    )
                )
                is not None
            ]
            pending_diagnostics.extend(diagnostic_rows)
            if len(pending_rows) >= _OUTPUT_BATCH_ROWS:
                flush()
            count += 1
            if count % _OUTPUT_BATCH_ROWS == 0:
                report_progress(
                    progress,
                    ProgressEvent(
                        OperationId.CAPACITY,
                        "finalize-capacity",
                        2 * total_rows + count,
                        3 * total_rows,
                        "work_rows",
                    ),
                )
        flush()
    finally:
        writer.close()
        diagnostic_writer.close()
    return count


def capacity_descriptor(order: CycleOrder) -> AnalysisDescriptor:
    """Describe the Capacity analysis for user interfaces.

    The output order depends on `order`: the first phase of the cycle order is listed first.
    """
    always = AvailabilityDescriptor("always", "", "Available for every emitted cycle row.")
    nullable = AvailabilityDescriptor(
        "scientific", "see_diagnostics", "See per-cell diagnostic table."
    )
    labels_units = {
        "cycle": ("Cycle", None, "int32"),
        "charge_capacity_ah": ("Charge capacity", "Ah", "float64"),
        "discharge_capacity_ah": ("Discharge capacity", "Ah", "float64"),
        "coulombic_efficiency_pct": ("Coulombic efficiency", "%", "float64"),
        "charge_energy_wh": ("Charge energy", "Wh", "float64"),
        "discharge_energy_wh": ("Discharge energy", "Wh", "float64"),
        "energy_efficiency_pct": ("Energy efficiency", "%", "float64"),
        "charge_c_rate": ("Charge C-rate", "C", "float64"),
        "discharge_c_rate": ("Discharge C-rate", "C", "float64"),
        "charge_capacity_mah_g": ("Charge specific capacity", "mAh/g", "float64"),
        "discharge_capacity_mah_g": ("Discharge specific capacity", "mAh/g", "float64"),
        "charge_capacity_mah_cm2": ("Charge areal capacity", "mAh/cm2", "float64"),
        "discharge_capacity_mah_cm2": ("Discharge areal capacity", "mAh/cm2", "float64"),
        "charge_energy_wh_kg": ("Charge specific energy", "Wh/kg", "float64"),
        "discharge_energy_wh_kg": ("Discharge specific energy", "Wh/kg", "float64"),
        "charge_energy_mwh_cm2": ("Charge areal energy", "mWh/cm2", "float64"),
        "discharge_energy_mwh_cm2": ("Discharge areal energy", "mWh/cm2", "float64"),
    }
    first, second = (
        ("discharge", "charge") if order is CycleOrder.CHARGE_DISCHARGE else ("charge", "discharge")
    )
    presentation = (
        "cycle",
        f"{first}_capacity_ah",
        f"{second}_capacity_ah",
        "coulombic_efficiency_pct",
        f"{first}_energy_wh",
        f"{second}_energy_wh",
        "energy_efficiency_pct",
    )
    remaining = tuple(name for name in CAPACITY_SCHEMA.names if name not in presentation)
    outputs = tuple(
        OutputDescriptor(
            name,
            labels_units[name][0],
            labels_units[name][2],
            labels_units[name][1],
            True,
            "capacity",
            always if name == "cycle" else nullable,
        )
        for name in presentation + remaining
    )
    descriptor = AnalysisDescriptor(
        1,
        OperationId.CAPACITY,
        "Capacity",
        "Per-cycle capacity, energy, efficiency, normalization and C-rate.",
        ("elaboration",),
        (ParameterGroup("capacity", "Capacity", 0),),
        (
            ParameterDescriptor(
                "reference_cycle",
                "Reference cycle",
                "Cycle whose second-phase capacity is the C-rate reference.",
                "integer",
                None,
                2,
                1,
                None,
                None,
                ParameterLevel.BASIC,
                "capacity",
            ),
            ParameterDescriptor(
                "active_material_mass_mg",
                "Active material mass",
                "Optional positive finite mass.",
                "number",
                "mg",
                None,
                None,
                None,
                None,
                ParameterLevel.BASIC,
                "capacity",
            ),
            ParameterDescriptor(
                "electrode_area_cm2",
                "Electrode area",
                "Optional positive finite area.",
                "number",
                "cm2",
                None,
                None,
                None,
                None,
                ParameterLevel.BASIC,
                "capacity",
            ),
        ),
        outputs,
        (),
        True,
        True,
        "work_rows",
    )
    descriptor.validate()
    return descriptor


def analyze_capacity(
    request: CapacityRequest,
    *,
    progress: ProgressSink | None = None,
    cancel: CancelToken | None = None,
    fault_hook: Any | None = None,
) -> CapacityResult:
    """Compute per-cycle Capacity from an elaboration result and publish it.

    Args:
        request: `CapacityRequest`.
        progress: Optional progress sink (unit ``work_rows``).
        cancel: Optional cancellation token.
        fault_hook: Testing hook; leave it None.

    Returns:
        `CapacityResult`; read ``result.ref.path / "tables" / "capacity.parquet"``.

    Raises:
        ValueError: invalid parameters, wrong upstream operation, fingerprint mismatch or a
            cycle order different from the elaboration.
        StorageError, OperationCancelled: storage failures or cancellation.

    The analysis streams the elaborated points once, so memory use is bounded.
    """
    request.validate()
    root = request.elaboration_path.resolve(strict=True)
    _validate_upstream(request, root)
    half_path = root / "indexes" / "half_cycles.parquet"
    points_path = root / "points" / "elaborated.parquet"
    half_file = pq.ParquetFile(half_path)
    half_rows = half_file.metadata.num_rows
    half_file.close()
    point_file = pq.ParquetFile(points_path)
    point_rows = point_file.metadata.num_rows
    point_file.close()
    total_work = max(point_rows * 3, 1)
    report_progress(
        progress, ProgressEvent(OperationId.CAPACITY, "starting", 0, total_work, "work_rows")
    )
    store = ResultStore(request.caller_root)
    staging = store.begin_job(request.sample_id, request.job_id, request.generation_id)
    try:
        scratch = staging.path / ".m3-scratch"
        scratch.mkdir()
        base_path = scratch / "base.parquet"
        cc_path = scratch / "cc.parquet"
        base_count, reference, base_scanned = _write_base_rows(
            half_path,
            base_path,
            request.parameters.cycle_order,
            request.batch_size,
            cancel,
            request.parameters.reference_cycle,
        )
        report_progress(
            progress,
            ProgressEvent(
                OperationId.CAPACITY, "aggregate-s6", point_rows, total_work, "work_rows"
            ),
        )
        first_direction = 1
        if half_rows:
            first_batch = next(
                pq.ParquetFile(half_path).iter_batches(batch_size=1, columns=["first_direction"])
            )
            first_direction = int(first_batch.column(0)[0].as_py())
        cc_count, cc_scanned = _write_cc_rows(
            points_path,
            cc_path,
            request.batch_size,
            cancel,
            progress,
            point_rows,
            first_direction,
        )
        tables = staging.path / "tables"
        tables.mkdir()
        capacity_path = tables / "capacity.parquet"
        diagnostics_path = tables / "capacity_diagnostics.parquet"
        cycle_count = _finalize_rows(
            base_path,
            cc_path,
            capacity_path,
            diagnostics_path,
            request.parameters,
            reference,
            request.batch_size,
            cancel,
            progress,
            point_rows,
        )
        if base_count != cc_count or base_count != cycle_count:
            raise RuntimeError("Capacity streams produced inconsistent cycle counts")
        scratch_bytes = base_path.stat().st_size + cc_path.stat().st_size
        report = {
            "schema_version": 1,
            "original_source_scans": 0,
            "elaboration_point_scans": 1,
            "half_index_scans": 1,
            "scratch_scans": 2,
            "half_index_records": base_scanned,
            "point_records": cc_scanned,
            "capacity_records": cycle_count,
            "scratch_bytes": scratch_bytes,
            "reference_cycle": request.parameters.reference_cycle,
            "reference_capacity_ah": reference if reference is not None and reference > 0 else None,
        }
        (staging.path / "reports").mkdir()
        (staging.path / "reports" / "capacity.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        shutil.rmtree(scratch)
        dependency = DependencyRef("elaboration", request.elaboration_fingerprint)
        base_parameters = {"cycle_order": request.parameters.cycle_order.value}
        base_fingerprint = _fingerprint(
            b"gregorithm-capacity-s6-v1", request.elaboration_fingerprint, base_parameters
        )
        c_rate_fingerprint = _fingerprint(
            b"gregorithm-capacity-s8-v1",
            request.elaboration_fingerprint,
            {**base_parameters, "reference_cycle": request.parameters.reference_cycle},
        )
        mass_fingerprint = (
            None
            if request.parameters.active_material_mass_mg is None
            else _fingerprint(
                b"gregorithm-capacity-s7-mass-v1",
                base_fingerprint,
                {"active_material_mass_mg": request.parameters.active_material_mass_mg},
            )
        )
        area_fingerprint = (
            None
            if request.parameters.electrode_area_cm2 is None
            else _fingerprint(
                b"gregorithm-capacity-s7-area-v1",
                base_fingerprint,
                {"electrode_area_cm2": request.parameters.electrode_area_cm2},
            )
        )
        overall = _fingerprint(
            b"gregorithm-capacity-result-v1",
            request.elaboration_fingerprint,
            request.parameters.as_dict(),
        )
        manifest = staging.build_manifest(
            operation=OperationId.CAPACITY,
            parameter_hash=hash_parameters(request.parameters.as_dict()),
            dependency_hash=hash_dependencies((dependency,)),
            dependencies=(dependency,),
            content_hash=overall,
            diagnostics=(),
            row_counts={"capacity_records": cycle_count},
            logical_schemas={
                "tables/capacity.parquet": "gregorithm.capacity.v1",
                "tables/capacity_diagnostics.parquet": "gregorithm.capacity_diagnostics.v1",
                "reports/capacity.json": "gregorithm.capacity_report.v1",
            },
        )
        report_progress(
            progress,
            ProgressEvent(
                OperationId.CAPACITY, "capacity-complete", total_work, total_work, "work_rows"
            ),
        )
        ref = store.publish(staging, manifest, cancel=cancel, fault_hook=fault_hook)
        return CapacityResult(
            ref,
            cycle_count,
            overall,
            base_fingerprint,
            mass_fingerprint,
            area_fingerprint,
            c_rate_fingerprint,
        )
    except Exception:
        staging.close()
        raise
