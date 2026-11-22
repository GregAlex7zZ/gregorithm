"""Explicit conversion of operator-entered physical quantities to internal SI units."""

from __future__ import annotations

import math

from .contracts import AreaUnit, LengthUnit, PhysicalInputs, VolumeUnit

_VOLUME = {VolumeUnit.M3: 1.0, VolumeUnit.CM3: 1e-6, VolumeUnit.MM3: 1e-9}
_AREA = {AreaUnit.M2: 1.0, AreaUnit.CM2: 1e-4}
_LENGTH = {LengthUnit.M: 1.0, LengthUnit.MM: 1e-3, LengthUnit.UM: 1e-6}


def _convert(value: float, factor: float, label: str) -> float:
    """Validate a finite positive value and multiply it by its SI factor."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return value * factor


def physical_inputs_si(
    *,
    active_volume: float | None = None,
    active_volume_unit: VolumeUnit = VolumeUnit.M3,
    effective_area: float | None = None,
    effective_area_unit: AreaUnit = AreaUnit.M2,
    particle_radius: float | None = None,
    particle_radius_unit: LengthUnit = LengthUnit.M,
) -> PhysicalInputs:
    """Convert direct physical inputs to SI without deriving or varying geometry.

    Each input is optional and independent; none is computed from the others.
    """

    def convert(value: float | None, factor: float, label: str) -> float | None:
        """Convert one optional value, keeping None as missing."""
        return None if value is None else _convert(value, factor, label)

    return PhysicalInputs(
        convert(active_volume, _VOLUME[active_volume_unit], "active volume"),
        convert(effective_area, _AREA[effective_area_unit], "effective area"),
        convert(particle_radius, _LENGTH[particle_radius_unit], "particle radius"),
    )
