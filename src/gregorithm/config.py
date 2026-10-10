"""Operational proposals, not experimentally validated thresholds."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field


def canonical(value):
    """Serialize a scientific configuration or manifest deterministically as sorted compact JSON; nonfinite JSON numbers are rejected.

    Parameters
    ----------
    value
        Bounded operation-specific value described by the function contract and API reference."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    """Return the SHA-256 digest of canonical JSON for configuration, selection, or revision identity.

    Parameters
    ----------
    value
        Bounded operation-specific value described by the function contract and API reference."""
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class DQDVConfig:
    """Immutable derivative configuration for uniform-grid Q(V) differentiation.

    ``legacy_corrected`` uses 10% of usable rows (3..5000 automatic nodes),
    a strict 0.005 V endpoint trim, mean capacity at repeated voltage, and no
    smoothing. ``voltage_grid`` explicitly requests a fixed voltage step.
    """

    profile: str = "legacy_corrected"
    n_points: int | None = None
    fraction: float | None = None
    voltage_step_V: float | None = None
    # Automatic scientific-grid cap; None disables it, not the execution budget.
    max_auto_points: int | None = 5000
    trim_extremes_V: float = 0.005
    e_lo: float | None = None
    e_hi: float | None = None
    duplicates: str = "mean"
    smoothing: bool = False
    window_points: int = 5
    polyorder: int = 2
    signed_capacity: bool = False

    def __post_init__(self):
        """Validate immutable constructor settings and reject unsupported or scientifically contradictory values."""
        for name in ("e_lo", "e_hi"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(name)
        if self.e_lo is not None and self.e_hi is not None and self.e_hi <= self.e_lo:
            raise ValueError("Positive voltage window required")
        if self.profile not in {"legacy_corrected", "voltage_grid"}:
            raise ValueError("dQ/dV profile")
        if sum(v is not None for v in (self.n_points, self.fraction, self.voltage_step_V)) > 1:
            raise ValueError("Choose points, fraction OR voltage step")
        if self.n_points is not None and (not isinstance(self.n_points, int) or self.n_points < 3):
            raise ValueError("n_points >= 3")
        if self.fraction is not None and not 0 < self.fraction <= 1:
            raise ValueError("fraction in (0,1]")
        if self.voltage_step_V is not None and (
            not math.isfinite(self.voltage_step_V) or self.voltage_step_V <= 0
        ):
            raise ValueError("positive voltage step")
        if self.profile == "legacy_corrected" and self.voltage_step_V is not None:
            raise ValueError("Select voltage_grid explicitly")
        if self.profile == "voltage_grid" and (
            self.fraction is not None or self.n_points is not None
        ):
            raise ValueError(
                "Point count/fraction require legacy_corrected; voltage_grid uses voltage_step_V"
            )
        if self.duplicates not in {"first", "last", "mean"}:
            raise ValueError("duplicates")
        if not math.isfinite(self.trim_extremes_V) or self.trim_extremes_V < 0:
            raise ValueError("trim")
        if self.max_auto_points is not None and (
            isinstance(self.max_auto_points, bool)
            or not isinstance(self.max_auto_points, int)
            or self.max_auto_points < 3
        ):
            raise ValueError("max_auto_points must be an integer >= 3 or None")
        if self.smoothing and (
            self.window_points < 3
            or self.window_points % 2 != 1
            or not 0 <= self.polyorder < self.window_points
        ):
            raise ValueError("Savitzky-Golay window/order")


@dataclass(frozen=True)
class AnalysisConfig:
    """Immutable scientific and execution defaults. Current is mA, voltage V, time seconds; rest threshold is inclusive 0.0001 mA, confirmation requires 2 samples and 0.1 s, and charge_then_discharge is the default order. Linear point signals govern integration. Managed memory defaults to 2 GiB shared across workers; this is not a hard RSS limit. See docs/API.md for references and all configurable fields."""

    rest_threshold_mA: float = 0.0001
    min_samples: int = 2
    min_duration_s: float = 0.1
    cycle_order: str = "charge_then_discharge"
    rest_assignment: str = "previous"
    integration: str = "linear_signals"
    max_gap_s: float | None = None
    batch_rows: int = 262144
    managed_memory_bytes: int = 2 * 1024**3
    workers: int = 1
    backend: str = "auto"
    compression: str = "zstd"
    compression_level: int = 1
    mass_g: float | None = None
    area_cm2: float | None = None
    reference_capacity_mAh: float | None = None
    reference_cycle_id: int | None = None
    reference_direction: str | None = None
    reference_half_cycle_id: int | None = None
    cc_window_s: float = 5.0
    cc_absolute_mA: float = 0.0001
    cc_relative: float = 0.02
    cv_band_V: float = 0.002
    dqdv: DQDVConfig = field(default_factory=DQDVConfig)

    def __post_init__(self):
        """Validate immutable constructor settings and reject unsupported or scientifically contradictory values."""
        for name in ("min_samples", "batch_rows", "workers"):
            if not isinstance(getattr(self, name), int) or isinstance(getattr(self, name), bool):
                raise ValueError(name + " must be an integer")
        if not math.isfinite(self.cc_window_s) or self.cc_window_s <= 0:
            raise ValueError("cc_window_s")
        for name in ("reference_cycle_id", "reference_half_cycle_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value <= 0):
                raise ValueError(name)
        for n in (
            "rest_threshold_mA",
            "min_duration_s",
            "cc_absolute_mA",
            "cc_relative",
            "cv_band_V",
        ):
            v = getattr(self, n)
            if not math.isfinite(v) or v < 0:
                raise ValueError(n)
        for n in ("mass_g", "area_cm2", "reference_capacity_mAh", "max_gap_s"):
            v = getattr(self, n)
            if v is not None and (not math.isfinite(v) or v <= 0):
                raise ValueError(n)
        if (
            self.min_samples < 1
            or self.batch_rows < 1
            or self.workers < 1
            or self.managed_memory_bytes < 65536
        ):
            raise ValueError("Invalid resource/persistence limits")
        if self.cycle_order not in {"charge_then_discharge", "discharge_then_charge"}:
            raise ValueError("cycle_order")
        if self.rest_assignment not in {"previous", "unassigned"}:
            raise ValueError("rest_assignment")
        if self.integration not in {"linear_signals", "left", "right"}:
            raise ValueError("integration")
        if self.backend not in {"auto", "numpy", "numba"}:
            raise ValueError("backend")
        if (
            sum(
                x is not None
                for x in (
                    self.reference_capacity_mAh,
                    self.reference_cycle_id,
                    self.reference_half_cycle_id,
                )
            )
            > 1
        ):
            raise ValueError("One C-rate reference selector")
        if self.reference_direction not in {None, "charge", "discharge"}:
            raise ValueError("reference_direction")

    def to_dict(self):
        """Return serializable configuration/provenance fields without materializing signal data."""
        return asdict(self)

    @property
    def config_hash(self):
        """Return a deterministic digest of the complete configuration, including operational resource settings."""
        return digest(self.to_dict())

    @classmethod
    def from_dict(cls, d):
        """Reconstruct this configuration or decision object from its serialized dictionary; constructor validation still applies."""
        d = dict(d)
        d["dqdv"] = DQDVConfig(**d.get("dqdv", {}))
        return cls(**d)

    @classmethod
    def legacy_persistence(cls, **kwargs):
        """Construct the explicit historical five-sample confirmation preset with zero minimum duration; other keyword overrides remain configurable."""
        return cls(min_samples=5, min_duration_s=0, **kwargs)
