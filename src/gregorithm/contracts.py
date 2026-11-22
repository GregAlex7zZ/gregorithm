"""Typed public contracts shared by all gregorithm operations.

These frozen dataclasses and string enums describe operations, analysis descriptors (used by user
interfaces to build forms and tables), resource estimates, result references and export requests.
They contain no scientific computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .diagnostics import Diagnostic

type JsonScalar = bool | int | float | str | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


class OperationId(StrEnum):
    """Stable identifiers of the operations recorded in result manifests."""

    INSPECT = "inspect"
    CANONICAL_IMPORT = "canonical_import"
    INGEST_AND_ELABORATE = "ingest_and_elaborate"
    CAPACITY = "capacity"
    DETECT_DCIR = "detect_dcir"
    MEASURE_DCIR = "measure_dcir"
    RECALCULATE_CYCLES = "recalculate_cycles"
    DQDV = "dqdv"
    GITT_ICI = "gitt_ici"
    READ_WINDOW = "read_window"
    EXPORT_TABLE = "export_table"
    CLEANUP = "cleanup"


class ExportFormat(StrEnum):
    """File formats supported by `export_table`: CSV and Parquet."""

    CSV = "csv"
    PARQUET = "parquet"


class ParameterLevel(StrEnum):
    """Whether a parameter belongs to the basic or the advanced settings group."""

    BASIC = "basic"
    ADVANCED = "advanced"


@dataclass(frozen=True, slots=True)
class LibraryInfo:
    """Library identification; `version` is the installed gregorithm version."""

    version: str


@dataclass(frozen=True, slots=True)
class ParameterGroup:
    """A named, ordered group of parameters in an `AnalysisDescriptor`."""

    id: str
    label: str
    order: int


@dataclass(frozen=True, slots=True)
class ParameterCondition:
    """Rule stating when a parameter is relevant.

    The parameter that carries the condition is used only while the parameter `parameter_id`
    holds one of `values`. Interfaces may hide or disable it otherwise; the library still
    validates every parameter, so a hidden control must keep a valid value.
    """

    parameter_id: str
    values: tuple[JsonScalar, ...]


@dataclass(frozen=True, slots=True)
class LiteratureReference:
    """Bibliographic reference of a published method implemented by the library.

    Interfaces and reports should show `citation()` next to results computed with the method.
    """

    authors: str
    title: str
    journal: str
    volume: str
    article: str
    year: int
    doi: str

    @property
    def url(self) -> str:
        """Return the DOI resolver URL of the reference."""
        return f"https://doi.org/{self.doi}"

    def citation(self) -> str:
        """Return a one-line citation, for example for a status bar or an export header."""
        return (
            f'{self.authors}, "{self.title}", {self.journal} {self.volume}, {self.article} '
            f"({self.year}), doi:{self.doi}"
        )


@dataclass(frozen=True, slots=True)
class ParameterDescriptor:
    """Machine-readable description of one analysis parameter.

    Includes identifier, label, help text, type, unit, default, optional numeric bounds or
    allowed values, settings level and group. `type` is ``"number"``, ``"integer"``,
    ``"boolean"`` or ``"string"``; a string with `values` is a choice. A `None` default marks an
    optional value. `conditions` (all must hold) tell when the parameter is relevant. Interfaces
    should render controls from these descriptors instead of duplicating scientific defaults.
    """

    id: str
    label: str
    help: str
    type: str
    unit: str | None
    default: JsonValue
    minimum: int | float | None
    maximum: int | float | None
    values: tuple[JsonScalar, ...] | None
    level: ParameterLevel
    group: str
    conditions: tuple[ParameterCondition, ...] = ()


@dataclass(frozen=True, slots=True)
class AvailabilityDescriptor:
    """Rule, reason code and explanation for when an output may be null."""

    rule: str
    reason_code: str
    reason: str


@dataclass(frozen=True, slots=True)
class OutputDescriptor:
    """Machine-readable description of one output column.

    Includes identifier, label, logical dtype, unit, whether it is shown by default, group and
    the availability rule explaining when its value can be null.
    """

    id: str
    label: str
    dtype: str
    unit: str | None
    display_default: bool
    group: str
    availability: AvailabilityDescriptor
    logical_values: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class WarningDescriptor:
    """Description of a warning code an analysis may emit, with its template."""

    code: str
    message_template: str
    fields: tuple[str, ...]
    severity: str = "warning"


@dataclass(frozen=True, slots=True)
class AnalysisDescriptor:
    """Self-description of an analysis: parameters, outputs and warnings.

    Returned by `capacity_descriptor`, `dcir_descriptor`, `dqdv_descriptor` and
    `cycle_correction_descriptor`. It lets applications build settings panels and result tables
    without hard-coding scientific defaults. `references` lists the published methods the
    analysis implements; applications should cite them.
    """

    schema_version: int
    id: OperationId
    title: str
    description: str
    prerequisites: tuple[str, ...]
    parameter_groups: tuple[ParameterGroup, ...]
    parameters: tuple[ParameterDescriptor, ...]
    outputs: tuple[OutputDescriptor, ...]
    warnings: tuple[WarningDescriptor, ...]
    supports_progress: bool
    supports_cancellation: bool
    progress_unit: str | None
    references: tuple[LiteratureReference, ...] = ()

    def validate(self) -> None:
        """Check internal consistency and raise `ValueError` on problems.

        Verifies the schema version, unique identifiers, that each parameter references a
        declared group, that conditions name other declared parameters with admissible values
        and that progress support matches the declared progress unit.
        """
        if self.schema_version != 1:
            raise ValueError("Unsupported descriptor schema version")
        group_ids = [group.id for group in self.parameter_groups]
        parameter_ids = [parameter.id for parameter in self.parameters]
        output_ids = [output.id for output in self.outputs]
        warning_codes = [warning.code for warning in self.warnings]
        for label, values in (
            ("parameter group", group_ids),
            ("parameter", parameter_ids),
            ("output", output_ids),
            ("warning", warning_codes),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate {label} identifier")
        if any(parameter.group not in group_ids for parameter in self.parameters):
            raise ValueError("Parameter references an unknown group")
        by_id = {parameter.id: parameter for parameter in self.parameters}
        for parameter in self.parameters:
            for condition in parameter.conditions:
                target = by_id.get(condition.parameter_id)
                if target is None or target.id == parameter.id:
                    raise ValueError("Parameter condition references an unknown parameter")
                if not condition.values:
                    raise ValueError("Parameter condition has no values")
                if target.values is not None and any(
                    value not in target.values for value in condition.values
                ):
                    raise ValueError("Parameter condition uses a value the target cannot take")
        if self.supports_progress != (self.progress_unit is not None):
            raise ValueError("Progress support and progress unit are inconsistent")


@dataclass(frozen=True, slots=True)
class DependencyRef:
    """Identifier and fingerprint of one input a result depends on."""

    id: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class OperationRequest:
    """Generic operation description used only by the generic `estimate` call.

    Typed request classes such as `ImportRequest` or `CapacityRequest` are used to run
    operations.
    """

    operation: OperationId
    sample_root: Path
    parameters: dict[str, JsonValue]
    dependencies: tuple[DependencyRef, ...] = ()


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    """Byte estimate for an operation that writes files.

    Attributes:
        expected_bytes: Expected size of the new output.
        pessimistic_bytes: Upper estimate of the output size (never below `expected_bytes`).
        required_free_bytes: Free space required before the operation is admitted.
    """

    expected_bytes: int
    pessimistic_bytes: int
    required_free_bytes: int

    def __post_init__(self) -> None:
        """Reject negative values and pessimistic < expected."""
        if min(self.expected_bytes, self.pessimistic_bytes, self.required_free_bytes) < 0:
            raise ValueError("Resource estimates must not be negative")
        if self.pessimistic_bytes < self.expected_bytes:
            raise ValueError("Pessimistic bytes must cover expected bytes")


@dataclass(frozen=True, slots=True)
class ExportRequest:
    """Request to export one table of a published result to an external file.

    Attributes:
        result_path: Generation directory, for example `result.ref.path`.
        table_path: Relative path of a Parquet table inside the generation, such as
            ``"tables/capacity.parquet"`` or ``"points/elaborated.parquet"``.
        destination: New output file; its extension must match `format`.
        format: `ExportFormat.CSV` or `ExportFormat.PARQUET`.
        columns: Optional unique subset of columns, in output order.
        batch_size: Rows copied per streaming batch.

    The constructor raises `ValueError` for unsafe table paths, a mismatched extension, an empty
    or duplicated column list, or a non-positive batch size.
    """

    result_path: Path
    table_path: str
    destination: Path
    format: ExportFormat
    columns: tuple[str, ...] | None = None
    batch_size: int = 65_536

    def __post_init__(self) -> None:
        """Validate the table path, extension, columns and batch size."""
        table = Path(self.table_path)
        if (
            not self.table_path
            or table.is_absolute()
            or ".." in table.parts
            or "." in table.parts
            or table.suffix.lower() != ".parquet"
        ):
            raise ValueError("table_path must name a safe relative Parquet result table")
        if self.destination.suffix.lower() != f".{self.format.value}":
            raise ValueError("destination extension does not match export format")
        if self.columns is not None and (
            not self.columns or len(self.columns) != len(set(self.columns))
        ):
            raise ValueError("columns must be nonempty and unique when supplied")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")


@dataclass(frozen=True, slots=True)
class ExportReceipt:
    """Description of a completed export.

    Records the written path, format, row count, byte size, SHA-256 of the file and the exported
    columns. Keep the checksum together with result fingerprints for reproducibility.
    """

    path: Path
    format: ExportFormat
    rows: int
    bytes: int
    sha256: str
    columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResultRef:
    """Reference to one immutable published result generation.

    Attributes:
        path: Generation directory containing `manifest.json` and the result files.
        generation_id: Identifier chosen by the caller when the result was produced.
        manifest_sha256: SHA-256 of the manifest bytes, used to detect substitution.
    """

    path: Path
    generation_id: str
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class ResultInfo:
    """Validated information about a reopened result generation.

    Returned by `open_result` and `open_current` after every file checksum has been verified.
    """

    ref: ResultRef
    operation: OperationId
    library_version: str
    diagnostics: tuple[Diagnostic, ...]
    files: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class CleanupReport:
    """Paths removed and paths deliberately skipped by a cleanup operation."""

    removed: tuple[Path, ...]
    skipped: tuple[Path, ...]
