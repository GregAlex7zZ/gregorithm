"""Differential capacity (dQ/dV) curves from elaborated data.

For each selected cycle and phase, only constant-current (CC) points are used, with the
cumulative charge capacity for charge phases and discharge capacity for discharge phases. Points
inside excluded spans (for example Accepted DCIR pulses) are omitted. An optional voltage window
(`voltage_min_v`, `voltage_max_v`) restricts every method. Three methods are available
(`DqdvMethod`):

- ``interpolated_gradient`` (default): the classic Gregorithm workflow. Points within
  `edge_trim_mv` of the half-cycle voltage extremes are dropped, the rest are sorted by voltage,
  Q(V) is linearly interpolated onto an evenly spaced grid with a number of points given as a
  count or as a percentage of the measured points, and differentiated with central differences
  (`numpy.gradient`); the value is placed at the grid nodes.
- ``histogram``: the derivative-free voltage histogram of E. Flores and S. Clark,
  J. Electrochem. Soc. 173 (2026) 120520, doi:10.1149/1945-7111/ae7e5c (Algorithm 1). The
  number of points per voltage bin, normalised to a probability density and scaled by the total
  capacity, approximates dQ/dV; the value is placed at the bin centres. It assumes uniform
  capacity increments between measurements.
- ``histogram_weighted``: the same histogram where each measurement interval is weighted by
  the capacity it carried, as proposed in the same paper for non-uniform sampling or C-rate.

Values are in Ah/V with their mathematical sign (negative when capacity grows while voltage
falls). Optional smoothing is stored next to the raw values: Savitzky-Golay for the derivative
methods, a Gaussian filter (in bins) for the histogram methods.

Optionally (default on) each curve is integrated back into a voltage curve (voltage
reconstruction, Eq. 5 of Flores and Clark) and compared with the measured one through the
absolute percentage error in total capacity (APE, Eq. 6) and the voltage RMSE over the
overlapping capacity interval (Eq. 7). These metrics make dQ/dV curves computed with different
methods or settings objectively comparable.

`analyze_dqdv` publishes ``tables/dqdv.parquet`` (columns `cycle`, `half_cycle`, `phase`,
`anchor_source_row`, `segment_index` (always 0), `voltage_mid_v`, `dqdv_raw_ah_per_v`,
`dqdv_smoothed_ah_per_v`), ``tables/dqdv_reconstruction.parquet`` when reconstruction is
enabled (the ``measured`` and ``reconstructed`` voltage curves, column `curve`) and
``reports/dqdv.json`` with unavailable curves and reconstruction metrics.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.dataset as ds  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from . import __version__
from .contracts import (
    AnalysisDescriptor,
    AvailabilityDescriptor,
    DependencyRef,
    JsonScalar,
    JsonValue,
    LiteratureReference,
    OperationId,
    OutputDescriptor,
    ParameterCondition,
    ParameterDescriptor,
    ParameterGroup,
    ParameterLevel,
    ResultRef,
)
from .diagnostics import DiagnosticCode
from .elaboration import Step
from .identity import canonical_parameter_bytes, hash_dependencies, hash_parameters
from .progress import CancelToken, ProgressEvent, ProgressSink, check_cancelled, report_progress
from .storage import GenerationManifest, ResultStore, open_result

DQDV_HISTOGRAM_REFERENCE = LiteratureReference(
    authors="E. Flores and S. Clark",
    title="Simple and Comparable Computation of Incremental Capacity Curves",
    journal="J. Electrochem. Soc.",
    volume="173",
    article="120520",
    year=2026,
    doi="10.1149/1945-7111/ae7e5c",
)
"""Source of the histogram methods and of the voltage reconstruction and its metrics."""


class DqdvPhase(StrEnum):
    """Phase of a dQ/dV curve: charge or discharge."""

    CHARGE = "charge"
    DISCHARGE = "discharge"


class DqdvMethod(StrEnum):
    """How dQ/dV is computed; see the module documentation for each method."""

    INTERPOLATED_GRADIENT = "interpolated_gradient"
    HISTOGRAM = "histogram"
    HISTOGRAM_WEIGHTED = "histogram_weighted"


class InterpolationPoints(StrEnum):
    """Unit of `DqdvParameters.interpolation_value`: percent of measured points or a count."""

    PERCENT = "percent"
    COUNT = "count"


_HISTOGRAM_METHODS = frozenset({DqdvMethod.HISTOGRAM, DqdvMethod.HISTOGRAM_WEIGHTED})


def _finite(value: object) -> bool:
    """Return True for a finite real number that is not a boolean."""
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True, slots=True)
class DqdvParameters:
    """Settings of the dQ/dV analysis.

    Every field is validated whatever the method, so unused fields must keep valid values.

    Attributes:
        smoothing_enabled: Also compute smoothed values (Savitzky-Golay for derivative methods,
            Gaussian for histogram methods). Default False.
        sg_window_points: Savitzky-Golay window in curve points (odd, >= 1). Default 5.
        sg_poly_order: Savitzky-Golay polynomial order (0 <= order < window). Default 2.
        method: `DqdvMethod`. Default ``interpolated_gradient``.
        voltage_min_v, voltage_max_v: Optional voltage window in V, applied to every method;
            None means the measured range.
        interpolation_mode: ``interpolated_gradient`` only: whether `interpolation_value` is a
            percentage of the measured points of the half-cycle or a point count.
        interpolation_value: Percentage in ``(0, 100]`` (default 10) or integer count >= 3.
        interpolation_max_points: Upper bound of the interpolation grid size. Default 5000.
            The grid always has at least 3 points.
        edge_trim_mv: ``interpolated_gradient`` only: drop points within this distance of the
            maximum and minimum voltage of the half-cycle (0 disables). Default 5.
        histogram_bin_mv: Histogram bin size in mV (positive). Default 1, as recommended by
            Flores and Clark for low-rate curves; it should not be much smaller than the typical
            voltage spacing between measurements.
        gaussian_sigma_bins: Standard deviation of the histogram Gaussian filter, in bins
            (positive). Default 1, the kernel reported by Flores and Clark.
        reconstruction_enabled: Integrate each curve back into a voltage curve and compute APE
            and RMSE. Default True.
    """

    smoothing_enabled: bool = False
    sg_window_points: int = 5
    sg_poly_order: int = 2
    method: DqdvMethod = DqdvMethod.INTERPOLATED_GRADIENT
    voltage_min_v: float | None = None
    voltage_max_v: float | None = None
    interpolation_mode: InterpolationPoints = InterpolationPoints.PERCENT
    interpolation_value: float = 10.0
    interpolation_max_points: int = 5000
    edge_trim_mv: float = 5.0
    histogram_bin_mv: float = 1.0
    gaussian_sigma_bins: float = 1.0
    reconstruction_enabled: bool = True

    def validate(self) -> None:
        """Raise ValueError for any invalid setting."""
        if not isinstance(self.smoothing_enabled, bool):
            raise ValueError("smoothing_enabled must be a boolean")
        if not isinstance(self.reconstruction_enabled, bool):
            raise ValueError("reconstruction_enabled must be a boolean")
        if not isinstance(self.sg_window_points, int) or isinstance(self.sg_window_points, bool):
            raise ValueError("sg_window_points must be an integer")
        if self.sg_window_points < 1:
            raise ValueError("sg_window_points must be a positive odd integer")
        if self.sg_window_points % 2 == 0:
            raise ValueError("sg_window_points must be odd")
        if not isinstance(self.sg_poly_order, int) or isinstance(self.sg_poly_order, bool):
            raise ValueError("sg_poly_order must be an integer")
        if not 0 <= self.sg_poly_order < self.sg_window_points:
            raise ValueError("sg_poly_order must be nonnegative and below sg_window_points")
        if self.method not in tuple(DqdvMethod):
            raise ValueError("method must be a DqdvMethod value")
        for name in ("voltage_min_v", "voltage_max_v"):
            value = getattr(self, name)
            if value is not None and not _finite(value):
                raise ValueError(f"{name} must be finite or None")
        if (
            self.voltage_min_v is not None
            and self.voltage_max_v is not None
            and self.voltage_min_v >= self.voltage_max_v
        ):
            raise ValueError("voltage_min_v must be below voltage_max_v")
        if self.interpolation_mode not in tuple(InterpolationPoints):
            raise ValueError("interpolation_mode must be 'percent' or 'count'")
        if not _finite(self.interpolation_value):
            raise ValueError("interpolation_value must be finite")
        if self.interpolation_mode == InterpolationPoints.PERCENT:
            if not 0 < self.interpolation_value <= 100:
                raise ValueError("interpolation_value must be a percentage in (0, 100]")
        elif self.interpolation_value != int(self.interpolation_value) or (
            self.interpolation_value < 3
        ):
            raise ValueError("interpolation_value must be an integer count of at least 3")
        if (
            not isinstance(self.interpolation_max_points, int)
            or isinstance(self.interpolation_max_points, bool)
            or self.interpolation_max_points < 3
        ):
            raise ValueError("interpolation_max_points must be an integer of at least 3")
        if not _finite(self.edge_trim_mv) or self.edge_trim_mv < 0:
            raise ValueError("edge_trim_mv must be nonnegative and finite")
        if not _finite(self.histogram_bin_mv) or self.histogram_bin_mv <= 0:
            raise ValueError("histogram_bin_mv must be positive and finite")
        if not _finite(self.gaussian_sigma_bins) or self.gaussian_sigma_bins <= 0:
            raise ValueError("gaussian_sigma_bins must be positive and finite")

    def as_dict(self) -> dict[str, Any]:
        """Return the parameters as a JSON-compatible dictionary."""
        return {
            "smoothing_enabled": self.smoothing_enabled,
            "sg_window_points": self.sg_window_points,
            "sg_poly_order": self.sg_poly_order,
            "method": str(self.method),
            "voltage_min_v": self.voltage_min_v,
            "voltage_max_v": self.voltage_max_v,
            "interpolation_mode": str(self.interpolation_mode),
            "interpolation_value": self.interpolation_value,
            "interpolation_max_points": self.interpolation_max_points,
            "edge_trim_mv": self.edge_trim_mv,
            "histogram_bin_mv": self.histogram_bin_mv,
            "gaussian_sigma_bins": self.gaussian_sigma_bins,
            "reconstruction_enabled": self.reconstruction_enabled,
        }


@dataclass(frozen=True, slots=True)
class DqdvSpan:
    """The computed values of a dQ/dV curve.

    Every method produces one piece per curve; the structure (and the `segment_index` column) is
    kept so that tables stay compatible.

    Attributes:
        segment_index: Position of the piece within its curve; always 0.
        voltage_mid_v: Voltage coordinate of each value, ascending, in V: grid nodes
            (``interpolated_gradient``) or bin centres (histogram methods).
        raw_ah_per_v: Raw dQ/dV values in Ah/V.
        smoothed_ah_per_v: Smoothed values, or None when smoothing is disabled.
    """

    segment_index: int
    voltage_mid_v: tuple[float, ...]
    raw_ah_per_v: tuple[float, ...]
    smoothed_ah_per_v: tuple[float, ...] | None


@dataclass(frozen=True, slots=True)
class DqdvReconstruction:
    """Voltage curve rebuilt by integrating a dQ/dV curve, with its quality metrics.

    Attributes:
        capacity_ah: Reconstructed cumulative capacity in Ah, in measurement order, starting at
            the first measured capacity of the curve.
        voltage_v: Voltage of each reconstructed point in V.
        ape_percent: Absolute percentage error between the measured and reconstructed total
            capacity, or None when the measured capacity range is zero.
        rmse_v: Voltage RMSE over the overlapping capacity interval, or None without overlap.
        unavailable_code: `DQDV_RECONSTRUCTION_UNAVAILABLE` when a metric is None.
        measured_capacity_ah, measured_voltage_v: The measured points the curve was computed
            from and compared with (CC, outside excluded spans, inside the window and, for
            ``interpolated_gradient``, after the edge trim), in measurement order.
    """

    capacity_ah: tuple[float, ...]
    voltage_v: tuple[float, ...]
    ape_percent: float | None
    rmse_v: float | None
    unavailable_code: DiagnosticCode | None = None
    measured_capacity_ah: tuple[float, ...] = ()
    measured_voltage_v: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class DqdvCurve:
    """dQ/dV curve of one half-cycle.

    Attributes:
        cycle, half_cycle, phase: Identification of the half-cycle.
        anchor_source_row: First source row of the half-cycle, used to re-map curves after a
            cycle correction.
        spans: Pieces of the curve; empty when no valid piece exists.
        unavailable_code: `DQDV_NO_VALID_SPAN` when `spans` is empty, otherwise None.
        reconstruction: Voltage reconstruction, or None when disabled or `spans` is empty.
    """

    cycle: int
    half_cycle: int
    phase: DqdvPhase
    anchor_source_row: int
    spans: tuple[DqdvSpan, ...]
    unavailable_code: DiagnosticCode | None = None
    reconstruction: DqdvReconstruction | None = None


@dataclass(frozen=True, slots=True)
class DqdvRequest:
    """Parameters of `analyze_dqdv`.

    Attributes:
        elaboration_path: Elaboration or cycle-correction generation directory.
        elaboration_fingerprint: Fingerprint of that generation.
        caller_root, sample_id, job_id, generation_id: Destination in the result store.
        cycles: Cycles to analyse; empty means every positive cycle.
        accepted_spans: Inclusive ``(first_source_row, last_source_row)`` ranges to exclude,
            typically from `accepted_correction_spans`.
        parameters: `DqdvParameters`.
        batch_size: Rows per streaming batch.
    """

    elaboration_path: Path
    elaboration_fingerprint: str
    caller_root: Path
    sample_id: str
    job_id: str
    generation_id: str
    cycles: tuple[int, ...] = ()
    accepted_spans: tuple[tuple[int, int], ...] = ()
    parameters: DqdvParameters = DqdvParameters()
    batch_size: int = 65_536

    def validate(self) -> None:
        """Validate parameters, fingerprint, cycles and excluded spans."""
        self.parameters.validate()
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len(self.elaboration_fingerprint) != 64 or any(
            value not in "0123456789abcdef" for value in self.elaboration_fingerprint
        ):
            raise ValueError("elaboration_fingerprint must be lowercase SHA-256 hex")
        if any(cycle < 0 for cycle in self.cycles):
            raise ValueError("cycles must be nonnegative")
        if len(self.cycles) != len(set(self.cycles)):
            raise ValueError("cycles must be unique")
        if any(first < 0 or last < first for first, last in self.accepted_spans):
            raise ValueError("accepted_spans must be ordered nonnegative inclusive ranges")


@dataclass(frozen=True, slots=True)
class DqdvResult:
    """Published dQ/dV result: reference, number of curves and points, fingerprint."""

    ref: ResultRef
    curves: int
    points: int
    fingerprint: str


_DEFAULT_DQDV_PARAMETERS = DqdvParameters()


def dqdv_method_references(parameters: DqdvParameters) -> tuple[LiteratureReference, ...]:
    """Return the publications to cite for results computed with `parameters`.

    The histogram methods and the voltage reconstruction come from Flores and Clark (2026).
    """
    if DqdvMethod(parameters.method) in _HISTOGRAM_METHODS or parameters.reconstruction_enabled:
        return (DQDV_HISTOGRAM_REFERENCE,)
    return ()


def savgol_interp(values: np.ndarray, window: int, order: int) -> np.ndarray:
    """Savitzky-Golay smoothing implemented with NumPy only.

    Fits a polynomial of `order` over a sliding window of `window` points using a Legendre basis
    for numerical stability. At the edges the nearest complete window is used. Arrays shorter
    than the window are returned unchanged (copied).
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) < window:
        return values.copy()
    half = window // 2
    result = np.empty_like(values)
    base = np.linspace(-1.0, 1.0, window)
    design = np.polynomial.legendre.legvander(base, order)
    for index in range(len(values)):
        start = min(max(index - half, 0), len(values) - window)
        position = index - start
        coefficients = np.linalg.lstsq(design, values[start : start + window], rcond=None)[0]
        result[index] = np.polynomial.legendre.legval(base[position], coefficients)
    return result


def gaussian_smooth(values: np.ndarray, sigma: float, truncate: float = 4.0) -> np.ndarray:
    """Gaussian filter implemented with NumPy only.

    Convolves `values` with a normalised Gaussian kernel of standard deviation `sigma` (in
    array points) truncated at ``truncate * sigma``. Edges are extended by mirror reflection
    including the edge value, which matches ``scipy.ndimage.gaussian_filter1d`` with its default
    ``mode="reflect"``. Used on histogram dQ/dV curves, where Flores and Clark report good
    results with ``sigma = 1`` bin.
    """
    values = np.asarray(values, dtype=np.float64)
    if not _finite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite")
    radius = int(truncate * sigma + 0.5)
    if len(values) == 0 or radius == 0:
        return values.copy()
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, radius, mode="symmetric")
    return np.convolve(padded, kernel, mode="valid")


def _voltage_window(voltage: np.ndarray, parameters: DqdvParameters) -> np.ndarray:
    """Return the mask of points inside the optional voltage window."""
    mask = np.ones(len(voltage), dtype=bool)
    if parameters.voltage_min_v is not None:
        mask &= voltage >= parameters.voltage_min_v
    if parameters.voltage_max_v is not None:
        mask &= voltage <= parameters.voltage_max_v
    return mask


def _edge_trim_mask(voltage: np.ndarray, trim_mv: float) -> np.ndarray:
    """Mask points farther than `trim_mv` from both voltage extremes (all points if 0)."""
    if trim_mv == 0 or len(voltage) == 0:
        return np.ones(len(voltage), dtype=bool)
    tolerance = trim_mv / 1000.0
    upper = float(voltage.max()) - tolerance
    lower = float(voltage.min()) + tolerance
    return np.asarray((voltage < upper) & (voltage > lower), dtype=bool)


def interpolation_point_count(
    measured_points: int,
    mode: InterpolationPoints,
    value: float,
    max_points: int = 5000,
) -> int:
    """Return the grid size of the ``interpolated_gradient`` method.

    In ``percent`` mode the size is ``round(value / 100 * measured_points)``; in ``count`` mode
    it is `value`. The result is clamped to ``[3, max_points]``.
    """
    if mode == InterpolationPoints.PERCENT:
        points = round(value / 100.0 * measured_points)
    else:
        points = int(value)
    return max(3, min(max_points, points))


def _interpolated_gradient(
    voltage: np.ndarray,
    capacity: np.ndarray,
    points: int,
    voltage_min_v: float | None,
    voltage_max_v: float | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Sort Q(V) by voltage, interpolate onto an even grid and apply `numpy.gradient`."""
    finite = np.isfinite(voltage) & np.isfinite(capacity)
    voltage, capacity = voltage[finite], capacity[finite]
    order = np.argsort(voltage, kind="stable")
    voltage, capacity = voltage[order], capacity[order]
    keep = np.r_[True, np.diff(voltage) != 0] if len(voltage) else np.empty(0, dtype=bool)
    voltage, capacity = voltage[keep], capacity[keep]
    if len(voltage) < 2:
        return None
    low = float(voltage[0]) if voltage_min_v is None else max(voltage_min_v, float(voltage[0]))
    high = float(voltage[-1]) if voltage_max_v is None else min(voltage_max_v, float(voltage[-1]))
    if high <= low or points < 2:
        return None
    grid = np.linspace(low, high, points)
    return grid, np.gradient(np.interp(grid, voltage, capacity), grid)


def interpolated_gradient(
    voltage_v: np.ndarray,
    capacity_ah: np.ndarray,
    points: int,
    *,
    voltage_min_v: float | None = None,
    voltage_max_v: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute dQ/dV with the classic Gregorithm interpolate-then-gradient workflow (pure).

    The points are sorted by voltage (the first of equal voltages is kept), Q(V) is linearly
    interpolated onto `points` evenly spaced voltages between the measured extremes (clipped to
    the optional window) and differentiated with second-order central differences
    (`numpy.gradient`). Non-monotonic voltage segments are therefore folded onto one voltage
    axis, which is meaningful only for essentially monotonic half-cycles.

    Returns:
        Grid voltages (ascending, V) and dQ/dV values (Ah/V).

    Raises:
        ValueError: fewer than two distinct finite voltages, an empty window or ``points < 2``.
    """
    result = _interpolated_gradient(
        np.asarray(voltage_v, dtype=np.float64),
        np.asarray(capacity_ah, dtype=np.float64),
        int(points),
        voltage_min_v,
        voltage_max_v,
    )
    if result is None:
        raise ValueError("interpolated_gradient needs two distinct voltages inside the window")
    return result


def _histogram(
    voltage: np.ndarray,
    capacity: np.ndarray,
    bin_v: float,
    weighted: bool,
    contiguous: np.ndarray | None,
    increasing: bool | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Histogram dQ/dV (Flores and Clark, Algorithm 1); None when it is undefined."""
    if len(voltage) < 2:
        return None
    low, high = float(voltage.min()), float(voltage.max())
    bins = math.floor((high - low) / bin_v * (1.0 + 1e-12))
    if bins < 1:
        return None
    edges = np.linspace(low, high, bins + 1)
    width = (high - low) / bins
    if weighted:
        delta_q = np.diff(capacity)
        voltage_mid = (voltage[:-1] + voltage[1:]) / 2.0
        if contiguous is not None:
            delta_q = np.where(contiguous, delta_q, 0.0)
        mass, _ = np.histogram(voltage_mid, bins=edges, weights=delta_q)
        if not np.any(mass):
            return None
        values = mass / width
    else:
        counts, _ = np.histogram(voltage, bins=edges)
        total = float(capacity.max() - capacity.min())
        if total <= 0:
            return None
        values = counts / (len(voltage) * width) * total
    if increasing is None:
        increasing = bool(voltage[-1] > voltage[0])
    centres = (edges[:-1] + edges[1:]) / 2.0
    return centres, values if increasing else -values


def dqdv_histogram(
    voltage_v: np.ndarray,
    capacity_ah: np.ndarray,
    bin_mv: float = 1.0,
    *,
    weighted: bool = False,
    contiguous: np.ndarray | None = None,
    increasing: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a derivative-free dQ/dV curve as a voltage histogram (pure function).

    Implements Algorithm 1 of E. Flores and S. Clark, J. Electrochem. Soc. 173 (2026) 120520,
    doi:10.1149/1945-7111/ae7e5c. With ``N_b = floor((max V - min V) / bin)`` bins spanning the
    measured voltages, the counts ``c_j`` become densities ``rho_j = c_j / (n * width)`` and
    ``I_j = rho_j * (max q - min q)``. The result assumes uniform capacity increments between
    measurements.

    With `weighted`, each measurement interval ``i`` contributes its capacity ``q_i - q_(i-1)``
    to the bin containing its midpoint voltage, so ``I_j`` is the capacity passed in the bin
    divided by the bin width; this handles non-uniform sampling or C-rate (the weighting
    proposed in the same paper). `contiguous` (length ``n - 1``) marks intervals between truly
    consecutive measurements; other intervals contribute nothing.

    Args:
        voltage_v, capacity_ah: Measurements in time order; capacity is cumulative.
        bin_mv: Bin size in mV.
        weighted: Use capacity-weighted counts.
        contiguous: Optional mask of valid intervals for `weighted`.
        increasing: Whether capacity grows with voltage; None infers it from the first and last
            voltage. Values are negative when capacity grows while voltage falls.

    Returns:
        Bin centres (ascending, V) and dQ/dV values (Ah/V). Empty bins hold 0.

    Raises:
        ValueError: mismatched arrays, a non-positive bin, fewer than two points, a voltage range
            narrower than one bin or no capacity to distribute.
    """
    voltage = np.asarray(voltage_v, dtype=np.float64)
    capacity = np.asarray(capacity_ah, dtype=np.float64)
    if voltage.shape != capacity.shape or voltage.ndim != 1:
        raise ValueError("voltage_v and capacity_ah must be aligned one-dimensional arrays")
    if not _finite(bin_mv) or bin_mv <= 0:
        raise ValueError("bin_mv must be positive and finite")
    mask = None if contiguous is None else np.asarray(contiguous, dtype=bool)
    if mask is not None and mask.shape != (max(len(voltage) - 1, 0),):
        raise ValueError("contiguous must have one value per measurement interval")
    if not (np.all(np.isfinite(voltage)) and np.all(np.isfinite(capacity))):
        raise ValueError("voltage_v and capacity_ah must be finite")
    result = _histogram(voltage, capacity, bin_mv / 1000.0, weighted, mask, increasing)
    if result is None:
        raise ValueError("histogram dQ/dV is undefined for these points and bin size")
    return result


def reconstruct_voltage_curve(
    voltage_v: np.ndarray,
    dqdv_ah_per_v: np.ndarray,
    capacity_start_ah: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a dQ/dV curve back into a voltage curve (pure function).

    Applies the cumulative trapezoidal rule (Eq. 5 of Flores and Clark, 2026) along the
    strictly increasing voltages. Cumulative capacity only grows during a measurement, so the
    curve starts at the low-voltage end when the integral is positive (charge-like) and at the
    high-voltage end otherwise.

    Returns:
        Reconstructed capacity (Ah, from `capacity_start_ah`) and voltage (V), in measurement
        order.

    Raises:
        ValueError: mismatched arrays, fewer than two points, non-finite values or voltages
            that are not strictly increasing.
    """
    voltage = np.asarray(voltage_v, dtype=np.float64)
    values = np.asarray(dqdv_ah_per_v, dtype=np.float64)
    if voltage.shape != values.shape or voltage.ndim != 1 or len(voltage) < 2:
        raise ValueError("voltage_v and dqdv_ah_per_v must be aligned with at least two points")
    if not (np.all(np.isfinite(voltage)) and np.all(np.isfinite(values))):
        raise ValueError("voltage_v and dqdv_ah_per_v must be finite")
    if np.any(np.diff(voltage) <= 0):
        raise ValueError("voltage_v must be strictly increasing")
    integral = np.r_[0.0, np.cumsum(np.diff(voltage) * (values[:-1] + values[1:]) / 2.0)]
    if integral[-1] >= 0:
        return capacity_start_ah + integral, voltage.copy()
    return (capacity_start_ah + integral - integral[-1])[::-1], voltage[::-1].copy()


def reconstruction_metrics(
    capacity_ah: np.ndarray,
    voltage_v: np.ndarray,
    reconstructed_capacity_ah: np.ndarray,
    reconstructed_voltage_v: np.ndarray,
) -> tuple[float | None, float | None]:
    """Compare a measured and a reconstructed voltage curve (pure function).

    Returns ``(ape_percent, rmse_v)`` following Eqs. 6 and 7 of Flores and Clark (2026):

    - APE: ``|Q - Q_rec| / Q * 100`` with ``Q`` the measured and ``Q_rec`` the reconstructed
      capacity range; None when ``Q`` is not positive.
    - RMSE: root mean square of the voltage differences at the reconstructed capacities inside
      the overlapping capacity interval, with the measured curve linearly interpolated there;
      None when the curves do not overlap.

    Perfect reconstruction gives APE = 0 % and RMSE = 0 V.
    """
    capacity = np.asarray(capacity_ah, dtype=np.float64)
    voltage = np.asarray(voltage_v, dtype=np.float64)
    rebuilt_q = np.asarray(reconstructed_capacity_ah, dtype=np.float64)
    rebuilt_v = np.asarray(reconstructed_voltage_v, dtype=np.float64)
    if len(capacity) == 0 or len(rebuilt_q) == 0:
        return None, None
    total = float(np.ptp(capacity))
    ape = (
        abs(total - float(np.ptp(rebuilt_q))) / total * 100.0
        if total > 0 and math.isfinite(total)
        else None
    )
    order = np.argsort(capacity, kind="stable")
    xp, fp = capacity[order], voltage[order]
    low = max(float(xp[0]), float(rebuilt_q.min()))
    high = min(float(xp[-1]), float(rebuilt_q.max()))
    overlap = (rebuilt_q >= low) & (rebuilt_q <= high)
    if high < low or not np.any(overlap):
        return ape, None
    difference = np.interp(rebuilt_q[overlap], xp, fp) - rebuilt_v[overlap]
    rmse = float(np.sqrt(np.mean(difference**2)))
    return ape, rmse if math.isfinite(rmse) else None


def _reconstruct_curve(
    spans: list[DqdvSpan],
    reference_voltage: np.ndarray,
    reference_capacity: np.ndarray,
) -> DqdvReconstruction:
    """Reconstruct the voltage curve of a dQ/dV curve and compare it with the measured points."""
    start = float(reference_capacity.min()) if len(reference_capacity) else 0.0
    capacities: list[np.ndarray] = []
    voltages: list[np.ndarray] = []
    for span in spans:
        values = span.smoothed_ah_per_v if span.smoothed_ah_per_v is not None else span.raw_ah_per_v
        if len(values) < 2:
            continue
        capacity, voltage = reconstruct_voltage_curve(
            np.asarray(span.voltage_mid_v), np.asarray(values), start
        )
        capacities.append(capacity)
        voltages.append(voltage)
        start = float(capacity[-1])
    unavailable = DiagnosticCode.DQDV_RECONSTRUCTION_UNAVAILABLE
    measured = (tuple(map(float, reference_capacity)), tuple(map(float, reference_voltage)))
    if not capacities:
        return DqdvReconstruction((), (), None, None, unavailable, *measured)
    capacity = np.concatenate(capacities)
    voltage = np.concatenate(voltages)
    ape, rmse = reconstruction_metrics(reference_capacity, reference_voltage, capacity, voltage)
    return DqdvReconstruction(
        tuple(map(float, capacity)),
        tuple(map(float, voltage)),
        ape,
        rmse,
        None if ape is not None and rmse is not None else unavailable,
        *measured,
    )


def _curve_spans(
    voltage: np.ndarray,
    capacity: np.ndarray,
    positions: np.ndarray,
    phase: DqdvPhase,
    parameters: DqdvParameters,
) -> tuple[list[DqdvSpan], np.ndarray]:
    """Compute the pieces of one curve and the measured points they represent.

    `positions` are the row indexes of the eligible points; consecutive indexes identify
    consecutive measurements for the weighted histogram.
    """
    method = DqdvMethod(parameters.method)
    spans: list[DqdvSpan] = []
    if method is DqdvMethod.INTERPOLATED_GRADIENT:
        trimmed = _edge_trim_mask(voltage, parameters.edge_trim_mv)
        points = interpolation_point_count(
            int(trimmed.sum()),
            InterpolationPoints(parameters.interpolation_mode),
            parameters.interpolation_value,
            parameters.interpolation_max_points,
        )
        gradient = _interpolated_gradient(
            voltage[trimmed],
            capacity[trimmed],
            points,
            parameters.voltage_min_v,
            parameters.voltage_max_v,
        )
        if gradient is not None:
            grid, raw = gradient
            smoothed = (
                savgol_interp(raw, parameters.sg_window_points, parameters.sg_poly_order)
                if parameters.smoothing_enabled
                else None
            )
            spans.append(_make_span(grid, raw, smoothed))
        return spans, trimmed & _voltage_window(voltage, parameters)
    window = _voltage_window(voltage, parameters)
    selected_v, selected_q = voltage[window], capacity[window]
    increasing = None
    if len(selected_v) and selected_v[-1] == selected_v[0]:
        increasing = phase is DqdvPhase.CHARGE
    histogram = _histogram(
        selected_v,
        selected_q,
        parameters.histogram_bin_mv / 1000.0,
        method is DqdvMethod.HISTOGRAM_WEIGHTED,
        np.diff(positions[window]) == 1,
        increasing,
    )
    if histogram is not None:
        centres, raw = histogram
        smoothed = (
            gaussian_smooth(raw, parameters.gaussian_sigma_bins)
            if parameters.smoothing_enabled
            else None
        )
        spans.append(_make_span(centres, raw, smoothed))
    return spans, window


def _make_span(voltage: np.ndarray, raw: np.ndarray, smoothed: np.ndarray | None) -> DqdvSpan:
    """Build the single piece of a non-segmented curve."""
    return DqdvSpan(
        0,
        tuple(map(float, voltage)),
        tuple(map(float, raw)),
        None if smoothed is None else tuple(map(float, smoothed)),
    )


def compute_dqdv(
    cycle: np.ndarray,
    half_cycle: np.ndarray,
    phase: np.ndarray,
    step_label: np.ndarray,
    source_row: np.ndarray,
    voltage_v: np.ndarray,
    charge_capacity_ah: np.ndarray,
    discharge_capacity_ah: np.ndarray,
    *,
    accepted_spans: tuple[tuple[int, int], ...] = (),
    parameters: DqdvParameters = _DEFAULT_DQDV_PARAMETERS,
) -> tuple[DqdvCurve, ...]:
    """Compute dQ/dV curves from elaborated point arrays (pure function).

    All arrays must have equal length and be in measurement order. `phase` holds
    ``"charge"``/``"discharge"`` (other values are ignored), `step_label` holds `Step` codes and
    `source_row` the canonical row numbers used by `accepted_spans`. Returns one `DqdvCurve`
    per positive cycle, half-cycle and phase, computed with ``parameters.method``.
    """
    parameters.validate()
    arrays = tuple(
        np.asarray(value)
        for value in (
            cycle,
            half_cycle,
            phase,
            step_label,
            source_row,
            voltage_v,
            charge_capacity_ah,
            discharge_capacity_ah,
        )
    )
    if len({len(value) for value in arrays}) != 1:
        raise ValueError("dQ/dV input arrays must align")
    cycle_a, half_a, phase_a, step_a, row_a, voltage_a, cq_a, dq_a = arrays
    excluded = np.zeros(len(cycle_a), dtype=bool)
    for first, last in accepted_spans:
        excluded |= (row_a >= first) & (row_a <= last)
    curves: list[DqdvCurve] = []
    keys = sorted(
        {
            (int(c), int(h), str(p))
            for c, h, p in zip(cycle_a.tolist(), half_a.tolist(), phase_a.tolist(), strict=True)
            if int(c) > 0 and str(p) in (DqdvPhase.CHARGE.value, DqdvPhase.DISCHARGE.value)
        }
    )
    for cycle_id, half_id, phase_name in keys:
        group = (cycle_a == cycle_id) & (half_a == half_id) & (phase_a == phase_name)
        positions = np.flatnonzero(group & (step_a == int(Step.CC)) & ~excluded)
        voltage = np.asarray(voltage_a[positions], dtype=np.float64)
        capacity_source = cq_a if phase_name == DqdvPhase.CHARGE.value else dq_a
        capacity = np.asarray(capacity_source[positions], dtype=np.float64)
        phase_id = DqdvPhase(phase_name)
        spans, reference = _curve_spans(voltage, capacity, positions, phase_id, parameters)
        reconstruction = (
            _reconstruct_curve(spans, voltage[reference], capacity[reference])
            if parameters.reconstruction_enabled and spans
            else None
        )
        curves.append(
            DqdvCurve(
                cycle_id,
                half_id,
                phase_id,
                int(row_a[group][0]),
                tuple(spans),
                None if spans else DiagnosticCode.DQDV_NO_VALID_SPAN,
                reconstruction,
            )
        )
    return tuple(curves)


def _validate_elaboration(path: Path, fingerprint: str) -> Path:
    """Check the input operation and fingerprint; return the resolved path."""
    root = path.resolve(strict=True)
    info = open_result(root)
    if info.operation not in (OperationId.INGEST_AND_ELABORATE, OperationId.RECALCULATE_CYCLES):
        raise ValueError("dQ/dV requires an elaboration or corrected-cycle generation")
    manifest = GenerationManifest.from_dict(
        json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    )
    if manifest.content_hash != fingerprint:
        raise ValueError("elaboration_fingerprint does not match the input manifest")
    return root


def _phase_from_half(half: np.ndarray, first_direction: int) -> np.ndarray:
    """Return charge/discharge phase labels for half-cycle numbers."""
    result = np.full(len(half), "", dtype="U9")
    active = half > 0
    directions = first_direction * np.where(half[active] % 2 == 1, 1, -1)
    result[active] = np.where(directions > 0, DqdvPhase.CHARGE.value, DqdvPhase.DISCHARGE.value)
    return result


_DQDV_SCHEMA = pa.schema(
    [
        ("cycle", pa.int32()),
        ("half_cycle", pa.int32()),
        ("phase", pa.string()),
        ("anchor_source_row", pa.uint64()),
        ("segment_index", pa.int32()),
        ("voltage_mid_v", pa.float64()),
        ("dqdv_raw_ah_per_v", pa.float64()),
        ("dqdv_smoothed_ah_per_v", pa.float64()),
    ]
)

_RECONSTRUCTION_SCHEMA = pa.schema(
    [
        ("cycle", pa.int32()),
        ("half_cycle", pa.int32()),
        ("phase", pa.string()),
        ("anchor_source_row", pa.uint64()),
        ("curve", pa.string()),
        ("point_index", pa.int32()),
        ("capacity_ah", pa.float64()),
        ("voltage_v", pa.float64()),
    ]
)


def _reconstruction_record(curve: DqdvCurve) -> dict[str, Any]:
    """Return the report entry of a curve's reconstruction metrics."""
    reconstruction = curve.reconstruction
    code = (
        DiagnosticCode.DQDV_RECONSTRUCTION_UNAVAILABLE
        if reconstruction is None
        else reconstruction.unavailable_code
    )
    return {
        "cycle": curve.cycle,
        "half_cycle": curve.half_cycle,
        "phase": curve.phase.value,
        "anchor_source_row": curve.anchor_source_row,
        "ape_percent": None if reconstruction is None else reconstruction.ape_percent,
        "rmse_v": None if reconstruction is None else reconstruction.rmse_v,
        "points": 0 if reconstruction is None else len(reconstruction.capacity_ah),
        "code": None if code is None else code.value,
    }


def analyze_dqdv(
    request: DqdvRequest,
    *,
    progress: ProgressSink | None = None,
    cancel: CancelToken | None = None,
    fault_hook: Any | None = None,
) -> DqdvResult:
    """Compute dQ/dV curves for selected cycles and publish them.

    Args:
        request: `DqdvRequest`.
        progress: Optional progress sink (unit ``cycles``).
        cancel: Optional cancellation token.
        fault_hook: Testing hook; leave it None.

    Returns:
        `DqdvResult`; read ``result.ref.path / "tables" / "dqdv.parquet"`` and, when
        reconstruction is enabled, ``tables/dqdv_reconstruction.parquet``. Reconstruction
        metrics are listed in ``reports/dqdv.json`` under ``reconstruction``.

    Raises:
        ValueError: invalid parameters, an unsupported input operation or a fingerprint mismatch.
        StorageError, OperationCancelled: storage failures or cancellation.

    Each selected cycle is read separately, so memory use depends on the largest cycle.
    """
    request.validate()
    root = _validate_elaboration(request.elaboration_path, request.elaboration_fingerprint)
    points_path = root / "points" / "elaborated.parquet"
    half_path = root / "indexes" / "half_cycles.parquet"
    available_cycles: list[int] = []
    first_direction = 1
    last_cycle: int | None = None
    half_file = pq.ParquetFile(half_path)
    try:
        for batch in half_file.iter_batches(
            batch_size=request.batch_size, columns=["cycle", "first_direction"]
        ):
            table = pa.Table.from_batches((batch,))
            if last_cycle is None and batch.num_rows:
                first_direction = int(table["first_direction"][0].as_py())
            cycles = table["cycle"].to_numpy(zero_copy_only=False)
            starts = np.r_[0, np.flatnonzero(cycles[1:] != cycles[:-1]) + 1]
            for cycle_id in map(int, cycles[starts]):
                if cycle_id != last_cycle:
                    available_cycles.append(cycle_id)
                    last_cycle = cycle_id
    finally:
        half_file.close()
    selected = (
        list(request.cycles)
        if request.cycles
        else [value for value in available_cycles if value > 0]
    )
    reconstruct = request.parameters.reconstruction_enabled
    store = ResultStore(request.caller_root)
    staging = store.begin_job(request.sample_id, request.job_id, request.generation_id)
    try:
        check_cancelled(cancel, OperationId.DQDV.value)
        tables = staging.path / "tables"
        reports = staging.path / "reports"
        tables.mkdir()
        reports.mkdir()
        output_path = tables / "dqdv.parquet"
        reconstruction_path = tables / "dqdv_reconstruction.parquet"
        row_group_size = min(request.batch_size, 262_144)
        writer = pq.ParquetWriter(output_path, _DQDV_SCHEMA, compression="zstd")
        rebuilt_writer = (
            pq.ParquetWriter(reconstruction_path, _RECONSTRUCTION_SCHEMA, compression="zstd")
            if reconstruct
            else None
        )
        curve_count = point_count = rebuilt_count = scanned = 0
        unavailable: list[dict[str, Any]] = []
        metrics: list[dict[str, Any]] = []
        try:
            dataset = ds.dataset(points_path, format="parquet")
            for position, cycle_id in enumerate(selected):
                check_cancelled(cancel, OperationId.DQDV.value)
                table = dataset.to_table(
                    filter=ds.field("cycle") == cycle_id,
                    columns=[
                        "source_row",
                        "cycle",
                        "half_cycle",
                        "step",
                        "voltage_v",
                        "charge_capacity_ah",
                        "discharge_capacity_ah",
                    ],
                )
                scanned += len(table)
                half = table["half_cycle"].combine_chunks().to_numpy(zero_copy_only=False)
                curves = compute_dqdv(
                    table["cycle"].combine_chunks().to_numpy(zero_copy_only=False),
                    half,
                    _phase_from_half(half, first_direction),
                    table["step"].combine_chunks().to_numpy(zero_copy_only=False),
                    table["source_row"].combine_chunks().to_numpy(zero_copy_only=False),
                    table["voltage_v"].combine_chunks().to_numpy(zero_copy_only=False),
                    table["charge_capacity_ah"].combine_chunks().to_numpy(zero_copy_only=False),
                    table["discharge_capacity_ah"].combine_chunks().to_numpy(zero_copy_only=False),
                    accepted_spans=request.accepted_spans,
                    parameters=request.parameters,
                )
                for curve in curves:
                    curve_count += 1
                    if curve.unavailable_code is not None:
                        unavailable.append(
                            {
                                "cycle": curve.cycle,
                                "phase": curve.phase.value,
                                "code": curve.unavailable_code.value,
                            }
                        )
                    for span in curve.spans:
                        count = len(span.raw_ah_per_v)
                        point_count += count
                        writer.write_table(
                            pa.table(
                                {
                                    "cycle": [curve.cycle] * count,
                                    "half_cycle": [curve.half_cycle] * count,
                                    "phase": [curve.phase.value] * count,
                                    "anchor_source_row": [curve.anchor_source_row] * count,
                                    "segment_index": [span.segment_index] * count,
                                    "voltage_mid_v": span.voltage_mid_v,
                                    "dqdv_raw_ah_per_v": span.raw_ah_per_v,
                                    "dqdv_smoothed_ah_per_v": (
                                        span.smoothed_ah_per_v
                                        if span.smoothed_ah_per_v is not None
                                        else [None] * count
                                    ),
                                },
                                schema=_DQDV_SCHEMA,
                            ),
                            row_group_size=row_group_size,
                        )
                    if rebuilt_writer is None:
                        continue
                    metrics.append(_reconstruction_record(curve))
                    rebuilt = curve.reconstruction
                    if rebuilt is None:
                        continue
                    for kind, capacity_values, voltage_values in (
                        ("measured", rebuilt.measured_capacity_ah, rebuilt.measured_voltage_v),
                        ("reconstructed", rebuilt.capacity_ah, rebuilt.voltage_v),
                    ):
                        count = len(capacity_values)
                        if not count:
                            continue
                        rebuilt_count += count
                        rebuilt_writer.write_table(
                            pa.table(
                                {
                                    "cycle": [curve.cycle] * count,
                                    "half_cycle": [curve.half_cycle] * count,
                                    "phase": [curve.phase.value] * count,
                                    "anchor_source_row": [curve.anchor_source_row] * count,
                                    "curve": [kind] * count,
                                    "point_index": list(range(count)),
                                    "capacity_ah": capacity_values,
                                    "voltage_v": voltage_values,
                                },
                                schema=_RECONSTRUCTION_SCHEMA,
                            ),
                            row_group_size=row_group_size,
                        )
                report_progress(
                    progress,
                    ProgressEvent(
                        OperationId.DQDV, "cycles", position + 1, len(selected), "cycles"
                    ),
                )
        finally:
            writer.close()
            if rebuilt_writer is not None:
                rebuilt_writer.close()
        parameters = {
            **request.parameters.as_dict(),
            "cycles": selected,
            "accepted_spans": [list(value) for value in request.accepted_spans],
        }
        digest = hashlib.sha256(
            b"gregorithm-dqdv-v2\0"
            + __version__.encode()
            + b"\0"
            + request.elaboration_fingerprint.encode()
            + b"\0"
            + canonical_parameter_bytes(parameters)
            + b"\0"
        )
        for path in (output_path, reconstruction_path) if reconstruct else (output_path,):
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            digest.update(b"\0")
        fingerprint = digest.hexdigest()
        references = dqdv_method_references(request.parameters)
        (reports / "dqdv.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "method": str(request.parameters.method),
                    "references": [
                        {"citation": value.citation(), "doi": value.doi} for value in references
                    ],
                    "selected_cycles": selected,
                    "curve_records": curve_count,
                    "point_records": point_count,
                    "unavailable": unavailable,
                    "reconstruction_enabled": reconstruct,
                    "reconstruction_point_records": rebuilt_count,
                    "reconstruction": metrics,
                    "original_source_scans": 0,
                    "localized_cycle_queries": len(selected),
                    "elaboration_point_records_read": scanned,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        dependency = DependencyRef("elaboration", request.elaboration_fingerprint)
        row_counts = {"dqdv": point_count}
        logical_schemas = {
            "tables/dqdv.parquet": "gregorithm.dqdv.v1",
            "reports/dqdv.json": "gregorithm.dqdv_report.v2",
        }
        if reconstruct:
            row_counts["dqdv_reconstruction"] = rebuilt_count
            logical_schemas["tables/dqdv_reconstruction.parquet"] = (
                "gregorithm.dqdv_reconstruction.v1"
            )
        manifest = staging.build_manifest(
            operation=OperationId.DQDV,
            parameter_hash=hash_parameters(parameters),
            dependency_hash=hash_dependencies((dependency,)),
            dependencies=(dependency,),
            content_hash=fingerprint,
            row_counts=row_counts,
            logical_schemas=logical_schemas,
        )
        ref = store.publish(staging, manifest, cancel=cancel, fault_hook=fault_hook)
        return DqdvResult(ref, curve_count, point_count, fingerprint)
    except Exception:
        staging.close()
        raise


def _parameter(
    identifier: str,
    label: str,
    help_text: str,
    kind: str,
    unit: str | None,
    default: JsonValue,
    *,
    minimum: float | None = None,
    values: tuple[JsonScalar, ...] | None = None,
    level: ParameterLevel = ParameterLevel.BASIC,
    group: str = "dqdv",
    conditions: tuple[ParameterCondition, ...] = (),
) -> ParameterDescriptor:
    """Build one dQ/dV `ParameterDescriptor` with keyword defaults."""
    return ParameterDescriptor(
        identifier,
        label,
        help_text,
        kind,
        unit,
        default,
        minimum,
        None,
        values,
        level,
        group,
        conditions,
    )


def _method_is(*methods: DqdvMethod) -> ParameterCondition:
    """Return the condition 'method is one of `methods`'."""
    return ParameterCondition("method", tuple(method.value for method in methods))


def dqdv_descriptor() -> AnalysisDescriptor:
    """Describe the dQ/dV analysis for user interfaces.

    The ``method`` choice drives which parameters are relevant (see each parameter's
    `conditions`); `references` holds the publication to cite for the histogram methods and
    the voltage reconstruction.
    """
    defaults = _DEFAULT_DQDV_PARAMETERS
    derivative = (DqdvMethod.INTERPOLATED_GRADIENT,)
    histogram = (DqdvMethod.HISTOGRAM, DqdvMethod.HISTOGRAM_WEIGHTED)
    smoothing_on = ParameterCondition("smoothing_enabled", (True,))
    descriptor = AnalysisDescriptor(
        1,
        OperationId.DQDV,
        "dQ/dV",
        "Signed dQ/dV from confirmed constant-current points and cumulative capacity, by finite "
        "differences, interpolated gradient or derivative-free voltage histograms, with "
        "optional voltage reconstruction.",
        ("elaboration",),
        (
            ParameterGroup("dqdv", "dQ/dV", 0),
            ParameterGroup("method", "Method settings", 1),
            ParameterGroup("smoothing", "Smoothing", 2),
            ParameterGroup("reconstruction", "Voltage reconstruction", 3),
        ),
        (
            _parameter(
                "method",
                "Method",
                "interpolated_gradient: classic interpolation onto N points and numpy.gradient; "
                "histogram: derivative-free voltage histogram (Flores and Clark, 2026); "
                "histogram_weighted: histogram weighted by the capacity of each interval.",
                "string",
                None,
                defaults.method.value,
                values=tuple(method.value for method in DqdvMethod),
            ),
            _parameter(
                "voltage_min_v",
                "E min",
                "Lower bound of the voltage window; empty for the measured minimum.",
                "number",
                "V",
                None,
            ),
            _parameter(
                "voltage_max_v",
                "E max",
                "Upper bound of the voltage window; empty for the measured maximum.",
                "number",
                "V",
                None,
            ),
            _parameter(
                "interpolation_mode",
                "Interpolation points unit",
                "Interpret the interpolation value as a percentage of the measured points of "
                "each half-cycle or as a point count.",
                "string",
                None,
                defaults.interpolation_mode.value,
                values=tuple(mode.value for mode in InterpolationPoints),
                group="method",
                conditions=(_method_is(DqdvMethod.INTERPOLATED_GRADIENT),),
            ),
            _parameter(
                "interpolation_value",
                "Interpolation points",
                "Percentage in (0, 100] or integer count >= 3, e.g. 10 % or 200 points.",
                "number",
                None,
                defaults.interpolation_value,
                minimum=0.0,
                group="method",
                conditions=(_method_is(DqdvMethod.INTERPOLATED_GRADIENT),),
            ),
            _parameter(
                "interpolation_max_points",
                "Maximum interpolation points",
                "Upper bound of the interpolation grid size.",
                "integer",
                None,
                defaults.interpolation_max_points,
                minimum=3,
                level=ParameterLevel.ADVANCED,
                group="method",
                conditions=(_method_is(DqdvMethod.INTERPOLATED_GRADIENT),),
            ),
            _parameter(
                "edge_trim_mv",
                "Edge trim",
                "Drop points within this distance of each half-cycle's voltage extremes "
                "(0 disables).",
                "number",
                "mV",
                defaults.edge_trim_mv,
                minimum=0.0,
                level=ParameterLevel.ADVANCED,
                group="method",
                conditions=(_method_is(DqdvMethod.INTERPOLATED_GRADIENT),),
            ),
            _parameter(
                "histogram_bin_mv",
                "Bin size",
                "Voltage bin width; sets the resolution of the curve. Keep it close to the "
                "typical voltage spacing of the measurements.",
                "number",
                "mV",
                defaults.histogram_bin_mv,
                minimum=0.0,
                group="method",
                conditions=(_method_is(*histogram),),
            ),
            _parameter(
                "smoothing_enabled",
                "Smoothing",
                "Also store smoothed values: Savitzky-Golay after differentiation, or a "
                "Gaussian filter for histograms.",
                "boolean",
                None,
                defaults.smoothing_enabled,
                group="smoothing",
            ),
            _parameter(
                "sg_window_points",
                "SG window",
                "Positive odd derivative-point window.",
                "integer",
                None,
                defaults.sg_window_points,
                minimum=1,
                level=ParameterLevel.ADVANCED,
                group="smoothing",
                conditions=(_method_is(*derivative), smoothing_on),
            ),
            _parameter(
                "sg_poly_order",
                "SG order",
                "Polynomial order below the window size.",
                "integer",
                None,
                defaults.sg_poly_order,
                minimum=0,
                level=ParameterLevel.ADVANCED,
                group="smoothing",
                conditions=(_method_is(*derivative), smoothing_on),
            ),
            _parameter(
                "gaussian_sigma_bins",
                "Gaussian sigma",
                "Standard deviation of the Gaussian filter in bins.",
                "number",
                "bins",
                defaults.gaussian_sigma_bins,
                minimum=0.0,
                level=ParameterLevel.ADVANCED,
                group="smoothing",
                conditions=(_method_is(*histogram), smoothing_on),
            ),
            _parameter(
                "reconstruction_enabled",
                "Reconstruct voltage",
                "Integrate each curve back into a voltage curve and report APE and RMSE.",
                "boolean",
                None,
                defaults.reconstruction_enabled,
                group="reconstruction",
            ),
        ),
        (
            OutputDescriptor(
                "voltage_mid_v",
                "Voltage",
                "float64",
                "V",
                True,
                "dqdv",
                AvailabilityDescriptor(
                    "valid_span",
                    DiagnosticCode.DQDV_NO_VALID_SPAN.value,
                    "Grid midpoints, grid nodes or bin centres, depending on the method.",
                ),
            ),
            OutputDescriptor(
                "dqdv_raw_ah_per_v",
                "Raw dQ/dV",
                "float64",
                "Ah/V",
                True,
                "dqdv",
                AvailabilityDescriptor(
                    "valid_span",
                    DiagnosticCode.DQDV_NO_VALID_SPAN.value,
                    "Available for half-cycles with enough CC data for the method.",
                ),
            ),
            OutputDescriptor(
                "dqdv_smoothed_ah_per_v",
                "Smoothed dQ/dV",
                "float64",
                "Ah/V",
                False,
                "dqdv",
                AvailabilityDescriptor(
                    "smoothing_enabled",
                    DiagnosticCode.DQDV_NO_VALID_SPAN.value,
                    "Null when smoothing is disabled.",
                ),
            ),
            OutputDescriptor(
                "capacity_ah",
                "Reconstructed capacity",
                "float64",
                "Ah",
                True,
                "reconstruction",
                AvailabilityDescriptor(
                    "reconstruction_enabled",
                    DiagnosticCode.DQDV_RECONSTRUCTION_UNAVAILABLE.value,
                    "Written to dqdv_reconstruction.parquet when reconstruction is enabled.",
                ),
            ),
            OutputDescriptor(
                "voltage_v",
                "Reconstructed voltage",
                "float64",
                "V",
                True,
                "reconstruction",
                AvailabilityDescriptor(
                    "reconstruction_enabled",
                    DiagnosticCode.DQDV_RECONSTRUCTION_UNAVAILABLE.value,
                    "Written to dqdv_reconstruction.parquet when reconstruction is enabled.",
                ),
            ),
        ),
        (),
        True,
        True,
        "cycles",
        (DQDV_HISTOGRAM_REFERENCE,),
    )
    descriptor.validate()
    return descriptor
