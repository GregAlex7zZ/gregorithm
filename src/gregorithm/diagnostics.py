"""Structured diagnostics used by every gregorithm operation.

A `Diagnostic` has a stable machine-readable `code`, a `kind` and an English message, plus
optional context such as cycle, source row or parameter name. Diagnostics explain warnings,
failures and why a result value is unavailable (null); an unavailable value is never zero.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class DiagnosticKind(StrEnum):
    """Category of a diagnostic.

    `ERROR` stops an operation, `WARNING` reports a noteworthy but valid situation, `UNAVAILABLE`
    explains a null result value and `EMPTY` reports that an analysis produced no records.
    """

    ERROR = "error"
    WARNING = "warning"
    UNAVAILABLE = "unavailable"
    EMPTY = "empty"


class DiagnosticCode(StrEnum):
    """Stable diagnostic codes.

    Codes are part of the public contract and may be used in programs and translations.
    Scientific codes cover import, integration, Capacity, normalization, C-rate, DCIR and dQ/dV.
    Storage, path, export and cancellation codes describe infrastructure failures.
    """

    IMPORT_MAPPING_REQUIRED = "IMPORT_MAPPING_REQUIRED"
    IMPORT_INVALID_TIME = "IMPORT_INVALID_TIME"
    IMPORT_TIME_DECREASE = "IMPORT_TIME_DECREASE"
    IMPORT_INSUFFICIENT_ROWS = "IMPORT_INSUFFICIENT_ROWS"
    IMPORT_REMOVED_INVALID_MEASUREMENT = "IMPORT_REMOVED_INVALID_MEASUREMENT"
    IMPORT_REMOVED_DUPLICATE_TIME = "IMPORT_REMOVED_DUPLICATE_TIME"
    IMPORT_UNSUPPORTED_FORMAT = "IMPORT_UNSUPPORTED_FORMAT"
    IMPORT_PARSE_ERROR = "IMPORT_PARSE_ERROR"
    IMPORT_IO_ERROR = "IMPORT_IO_ERROR"
    INSUFFICIENT_DISK_SPACE = "INSUFFICIENT_DISK_SPACE"
    INPUT_INVALID_PARAMETER = "INPUT_INVALID_PARAMETER"
    INTEGRATION_ENERGY_UNAVAILABLE = "INTEGRATION_ENERGY_UNAVAILABLE"
    CAPACITY_MISSING_PHASE = "CAPACITY_MISSING_PHASE"
    CAPACITY_ZERO_DENOMINATOR = "CAPACITY_ZERO_DENOMINATOR"
    CAPACITY_NO_ACTIVE_PAIR_CYCLE_ZERO = "CAPACITY_NO_ACTIVE_PAIR_CYCLE_ZERO"
    NORMALIZATION_BASIS_MISSING = "NORMALIZATION_BASIS_MISSING"
    NUMERIC_RESULT_UNREPRESENTABLE = "NUMERIC_RESULT_UNREPRESENTABLE"
    C_RATE_REFERENCE_UNAVAILABLE = "C_RATE_REFERENCE_UNAVAILABLE"
    C_RATE_CC_UNAVAILABLE = "C_RATE_CC_UNAVAILABLE"
    DCIR_NO_CANDIDATES = "DCIR_NO_CANDIDATES"
    DCIR_MISSING_PRE_PULSE_REFERENCE = "DCIR_MISSING_PRE_PULSE_REFERENCE"
    DCIR_ZERO_CURRENT_DIFFERENCE = "DCIR_ZERO_CURRENT_DIFFERENCE"
    DCIR_TIME_AFTER_LAST_SAMPLE = "DCIR_TIME_AFTER_LAST_SAMPLE"
    DCIR_SOC_UNAVAILABLE = "DCIR_SOC_UNAVAILABLE"
    DCIR_MANUAL_CORRECTION_BOUNDARY_INVALID = "DCIR_MANUAL_CORRECTION_BOUNDARY_INVALID"
    DQDV_NO_VALID_SPAN = "DQDV_NO_VALID_SPAN"
    DQDV_RECONSTRUCTION_UNAVAILABLE = "DQDV_RECONSTRUCTION_UNAVAILABLE"
    GITT_ICI_INCOMPLETE_EVENT = "GITT_ICI_INCOMPLETE_EVENT"
    GITT_ICI_MISSING_REFERENCE = "GITT_ICI_MISSING_REFERENCE"
    GITT_ICI_FIT_UNAVAILABLE = "GITT_ICI_FIT_UNAVAILABLE"
    GITT_ICI_EXTRAPOLATION_WARNING = "GITT_ICI_EXTRAPOLATION_WARNING"

    OPERATION_CANCELLED = "OPERATION_CANCELLED"
    PATH_INVALID = "PATH_INVALID"
    PATH_OUTSIDE_ROOT = "PATH_OUTSIDE_ROOT"
    RESULT_JOB_EXISTS = "RESULT_JOB_EXISTS"
    RESULT_JOB_ACTIVE = "RESULT_JOB_ACTIVE"
    RESULT_GENERATION_EXISTS = "RESULT_GENERATION_EXISTS"
    RESULT_MANIFEST_INVALID = "RESULT_MANIFEST_INVALID"
    RESULT_FILE_MISSING = "RESULT_FILE_MISSING"
    RESULT_FILE_UNEXPECTED = "RESULT_FILE_UNEXPECTED"
    RESULT_FILE_CHECKSUM_MISMATCH = "RESULT_FILE_CHECKSUM_MISMATCH"
    RESULT_POINTER_INVALID = "RESULT_POINTER_INVALID"
    RESULT_PUBLICATION_CONFLICT = "RESULT_PUBLICATION_CONFLICT"
    STORAGE_IO_ERROR = "STORAGE_IO_ERROR"
    EXPORT_INVALID_SOURCE = "EXPORT_INVALID_SOURCE"
    EXPORT_DESTINATION_EXISTS = "EXPORT_DESTINATION_EXISTS"
    EXPORT_IO_ERROR = "EXPORT_IO_ERROR"
    ESTIMATE_INVALID_REQUEST = "ESTIMATE_INVALID_REQUEST"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """One structured diagnostic record.

    Attributes:
        code: Stable `DiagnosticCode`.
        kind: `DiagnosticKind` category.
        message: Human-readable English explanation (never empty).
        operation, count, column, cycle, half_cycle, event_id, source_row, time_s, parameter:
            Optional context, present only when genuinely known.
    """

    code: DiagnosticCode
    kind: DiagnosticKind
    message: str
    operation: str | None = None
    count: int | None = None
    column: str | None = None
    cycle: int | None = None
    half_cycle: int | None = None
    event_id: str | None = None
    source_row: int | None = None
    time_s: float | None = None
    parameter: str | None = None

    def __post_init__(self) -> None:
        """Reject empty messages and negative counts or source rows."""
        if not self.message:
            raise ValueError("Diagnostic message must not be empty")
        if self.count is not None and self.count < 0:
            raise ValueError("Diagnostic count must not be negative")
        if self.source_row is not None and self.source_row < 0:
            raise ValueError("Diagnostic source row must not be negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary with only the populated fields.

        `time_s` is written as ``time_s_hex`` (`float.hex()`), preserving exact binary64 values.
        """
        data: dict[str, Any] = {
            "code": self.code.value,
            "kind": self.kind.value,
            "message": self.message,
        }
        for name in (
            "operation",
            "count",
            "column",
            "cycle",
            "half_cycle",
            "event_id",
            "source_row",
            "parameter",
        ):
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        if self.time_s is not None:
            data["time_s_hex"] = self.time_s.hex()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Diagnostic:
        """Rebuild a diagnostic from `to_dict` output; unknown keys raise ValueError."""
        allowed = {
            "code",
            "kind",
            "message",
            "operation",
            "count",
            "column",
            "cycle",
            "half_cycle",
            "event_id",
            "source_row",
            "time_s_hex",
            "parameter",
        }
        if set(data) - allowed:
            raise ValueError("Unknown diagnostic fields")
        time_hex = data.get("time_s_hex")
        return cls(
            code=DiagnosticCode(data["code"]),
            kind=DiagnosticKind(data["kind"]),
            message=str(data["message"]),
            operation=data.get("operation"),
            count=data.get("count"),
            column=data.get("column"),
            cycle=data.get("cycle"),
            half_cycle=data.get("half_cycle"),
            event_id=data.get("event_id"),
            source_row=data.get("source_row"),
            time_s=float.fromhex(time_hex) if time_hex is not None else None,
            parameter=data.get("parameter"),
        )


def first_cause(candidates: Iterable[Diagnostic | None]) -> Diagnostic | None:
    """Return the first non-None diagnostic, or None when all candidates are None.

    Callers list checks in their documented evaluation order so that a null value is explained by
    its first causal failure.
    """
    return next((candidate for candidate in candidates if candidate is not None), None)
