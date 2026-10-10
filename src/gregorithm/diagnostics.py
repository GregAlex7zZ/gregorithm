"""Typed failures and serializable diagnostics; no UI dependencies."""

from dataclasses import asdict, dataclass


class BatteryError(Exception):
    """Base exception for explicit library contract and data failures."""

    pass


class SchemaError(BatteryError):
    """Required standardized columns, types, names, or mappings are invalid."""

    pass


class MaterializationLimitError(BatteryError):
    """A pandas/input materialization exceeds an explicit row or conservative byte limit; no implicit truncation occurs."""

    pass


class BudgetExceeded(BatteryError):
    """A managed reader/computation buffer cannot fit its declared budget."""

    pass


class PlotBudgetExceeded(BudgetExceeded):
    """The requested whole-figure/update display exceeds point or serialized payload limits."""

    pass


class SourceChangedError(BatteryError):
    """The captured source fingerprint/content no longer matches the requested revision."""

    pass


class RevisionMismatchError(BatteryError):
    """Selections or results from incompatible source/assignment revisions were combined."""

    pass


class AmbiguousTimeSelectionError(BatteryError):
    """A numeric time query spans multiple acquisition sections without section IDs."""

    pass


class Cancelled(BatteryError):
    """Cooperative cancellation interrupted unfinished work; completed results remain valid."""

    pass


@dataclass(frozen=True)
class Diagnostic:
    """Serializable scientific quality issue with source support, consequence, severity, and suggested inspection action."""

    code: str
    severity: str = "warning"
    trace_id: str | None = None
    sample_start: int | None = None
    sample_end: int | None = None
    analysis: str | None = None
    consequence: str = ""
    action: str = "Inspect original samples and configuration"

    def to_dict(self):
        """Return serializable configuration/provenance fields without materializing signal data."""
        return asdict(self)
