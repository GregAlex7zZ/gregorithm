from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest
from scipy.optimize import least_squares

from gregorithm.gitt_ici import (
    CanonicalSample,
    DelacourtOffsetMode,
    DelacourtParameters,
    delacourt_voltage,
    fit_delacourt,
    spherical_eigenvalues,
    spherical_f,
)


def sample(row: int, time_s: float, voltage_v: float) -> CanonicalSample:
    return CanonicalSample(row, time_s, 1.0, voltage_v, 1, 1, 0.0, 0.0)


def synthetic(
    *,
    log_theta: float,
    offset: float,
    tau: float = 100.0,
    u0: float = 3.0,
    u1: float = 3.2,
    noise: np.ndarray | None = None,
) -> tuple[CanonicalSample, ...]:
    times = np.linspace(2.0, tau, 30)
    voltage = np.asarray(
        delacourt_voltage(
            times,
            log10_theta_tau=log_theta,
            pulse_duration_s=tau,
            initial_ocv_v=u0,
            final_ocv_v=u1,
            offset_v=offset,
        )
    )
    if noise is not None:
        voltage = voltage + noise
    return tuple(
        sample(index, float(time), float(value))
        for index, (time, value) in enumerate(zip(times, voltage, strict=True), 1)
    )


def test_eigenvalues_are_positive_roots() -> None:
    roots = spherical_eigenvalues(20)
    assert len(roots) == 20
    assert all(left < right for left, right in pairwise(roots))
    assert roots[0] == pytest.approx(4.4934094579, rel=1e-10, abs=0)
    assert max(abs(math.tan(root) - root) for root in roots) < 1e-8


def test_spherical_response_zero_short_and_long_limits() -> None:
    assert spherical_f(0.0) == 0.0
    theta = 1e-8
    assert spherical_f(theta) / (2 * math.sqrt(theta / math.pi)) == pytest.approx(
        1.0, rel=1e-4, abs=0
    )
    large = 10.0
    assert spherical_f(large) == pytest.approx(3 * large + 0.2, rel=2e-4, abs=0)


def test_voltage_starts_at_ocv_plus_offset() -> None:
    assert delacourt_voltage(
        0.0,
        log10_theta_tau=-1,
        pulse_duration_s=100,
        initial_ocv_v=3.0,
        final_ocv_v=3.2,
        offset_v=0.1,
    ) == pytest.approx(3.1, rel=1e-12, abs=0)


def test_short_time_voltage_recovers_approved_diffusion_limit() -> None:
    radius, tau, diffusion, delta_u = 5e-6, 100.0, 2e-14, 0.2
    theta_tau = diffusion * tau / radius**2
    expected_slope = 2 * delta_u * radius / (3 * tau * math.sqrt(math.pi * diffusion))
    times = np.array([1e-8, 4e-8]) * tau / theta_tau
    voltage = np.asarray(
        delacourt_voltage(
            times,
            log10_theta_tau=math.log10(theta_tau),
            pulse_duration_s=tau,
            initial_ocv_v=3.0,
            final_ocv_v=3.0 + delta_u,
            offset_v=0.0,
        )
    )
    numerical_slope = (voltage[1] - voltage[0]) / (math.sqrt(times[1]) - math.sqrt(times[0]))
    assert numerical_slope == pytest.approx(expected_slope, rel=3e-4, abs=0)


def test_fixed_offset_fit_recovers_synthetic_diffusion() -> None:
    radius, tau, log_theta = 5e-6, 100.0, -0.7
    result = fit_delacourt(
        synthetic(log_theta=log_theta, offset=0.04, tau=tau),
        reference_time_s=0,
        parameters=DelacourtParameters(radius, tau, 3.0, 3.2, DelacourtOffsetMode.FIXED, 0.04),
    )
    expected = 10**log_theta * radius**2 / tau
    assert result.diffusion_m2_s == pytest.approx(expected, rel=1e-7, abs=0)
    assert result.offset_v == pytest.approx(0.04, rel=1e-12, abs=0)
    assert result.rmse_v < 1e-10
    assert result.identifiable and not result.at_search_boundary


def test_free_offset_variable_projection_recovers_both_parameters() -> None:
    radius, tau, log_theta, offset = 8e-6, 60.0, 0.3, -0.025
    noise = 2e-5 * np.sin(np.arange(30))
    result = fit_delacourt(
        synthetic(log_theta=log_theta, offset=offset, tau=tau, noise=noise),
        reference_time_s=0,
        parameters=DelacourtParameters(radius, tau, 3.0, 3.2, DelacourtOffsetMode.FREE),
    )
    assert result.log10_theta_tau == pytest.approx(log_theta, abs=2e-3)
    assert result.offset_v == pytest.approx(offset, abs=3e-5)
    assert result.standard_uncertainty_m2_s is not None
    assert result.identifiable


def test_variable_projection_matches_joint_least_squares() -> None:
    tau, log_theta, offset = 90.0, -0.2, 0.03
    rows = synthetic(log_theta=log_theta, offset=offset, tau=tau)
    fit = fit_delacourt(
        rows,
        reference_time_s=0,
        parameters=DelacourtParameters(6e-6, tau, 3.0, 3.2, DelacourtOffsetMode.FREE),
    )
    times = np.array([row.time_s for row in rows])
    voltage = np.array([row.voltage_v for row in rows])

    def residual(parameters: np.ndarray) -> np.ndarray:
        return voltage - np.asarray(
            delacourt_voltage(
                times,
                log10_theta_tau=float(parameters[0]),
                pulse_duration_s=tau,
                initial_ocv_v=3.0,
                final_ocv_v=3.2,
                offset_v=float(parameters[1]),
            )
        )

    joint = least_squares(residual, x0=np.array([0.0, 0.0]), bounds=([-6, -1], [3, 1]))
    assert fit.log10_theta_tau == pytest.approx(joint.x[0], abs=1e-7)
    assert fit.offset_v == pytest.approx(joint.x[1], abs=1e-9)


def test_minimum_at_boundary_is_flagged_and_uncertainty_suppressed() -> None:
    result = fit_delacourt(
        synthetic(log_theta=-7.0, offset=0.0),
        reference_time_s=0,
        parameters=DelacourtParameters(5e-6, 100, 3.0, 3.2),
    )
    assert result.at_search_boundary
    assert not result.identifiable
    assert result.standard_uncertainty_m2_s is None


def test_fit_uses_complete_observed_pulse_but_not_reference_or_after_pulse() -> None:
    rows = (
        sample(0, 0.0, 99.0),
        *synthetic(log_theta=-0.5, offset=0.01, tau=100),
        sample(99, 101.0, -99.0),
    )
    result = fit_delacourt(
        rows,
        reference_time_s=0,
        parameters=DelacourtParameters(5e-6, 100, 3.0, 3.2, fixed_offset_v=0.01),
    )
    assert 0 not in result.source_rows and 99 not in result.source_rows
    assert result.relative_times_s[-1] == 100.0
