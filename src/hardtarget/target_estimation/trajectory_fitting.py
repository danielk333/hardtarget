"""Fit a smooth radial trajectory to range and range-rate observations [^1].

[^1]: Markkanen et al. (2013), "High-precision measurement of satellite range and velocity using the EISCAT radar".
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class TrajectoryFit:
    """Result of a joint cubic range and range-rate trajectory fit.

    The four parameters describe range, range-rate, acceleration, and jerk at
    `reference_time`. Also stores the fit residuals and computes the covariance
    """

    reference_time: float
    parameters: npt.NDArray[np.float64]
    parameter_covariance: npt.NDArray[np.float64]
    fitted_ranges: npt.NDArray[np.float64]
    fitted_range_rates: npt.NDArray[np.float64]
    range_residuals: npt.NDArray[np.float64]
    range_rate_residuals: npt.NDArray[np.float64]

    def range(self, times: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the fitted range"""
        return range_theory_matrix(times - self.reference_time) @ self.parameters

    def range_rate(self, times: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the fitted range-rate"""
        return range_rate_theory_matrix(times - self.reference_time) @ self.parameters

    def acceleration(self, times: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the fitted acceleration"""
        return self.parameters[2] + self.parameters[3] * (times - self.reference_time)


def range_theory_matrix(dt: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.stack((np.ones_like(dt), dt, 0.5 * dt**2, dt**3 / 6.0), axis=-1)


def range_rate_theory_matrix(dt: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.stack((np.zeros_like(dt), np.ones_like(dt), dt, 0.5 * dt**2), axis=-1)


def fit_trajectory(
    range_times: npt.NDArray[np.float64],
    ranges: npt.NDArray[np.float64],
    range_rate_times: npt.NDArray[np.float64],
    range_rates: npt.NDArray[np.float64],
    range_variances: npt.NDArray[np.float64] | None = None,
    range_rate_variances: npt.NDArray[np.float64] | None = None,
    reference_time: float | None = None,
) -> TrajectoryFit:
    """Jointly fit range and range-rate observations to a cubic range model.

    The fitted model is

    `r(t) = r0 + v0*dt + acceleration0*dt**2/2 + jerk*dt**3/6`

    Its analytic derivative is used for the range-rate observations. Range and
    range-rate may be observed at different times. `range_variances` and
    `range_rate_variances` are the observation variances used in the
    generalized least-squares inversion and default to one.
    `reference_time` defaults to the midpoint of the complete observation interval.
    """
    if range_variances is None:
        range_variances = np.ones_like(ranges)
    if range_rate_variances is None:
        range_rate_variances = np.ones_like(range_rates)

    if reference_time is None:
        reference_time = 0.5 * (
            min(range_times.min(), range_rate_times.min()) + max(range_times.max(), range_rate_times.max())
        )

    range_theory = range_theory_matrix(range_times - reference_time)
    range_rate_theory = range_rate_theory_matrix(range_rate_times - reference_time)
    theory_matrix = np.vstack((range_theory, range_rate_theory))
    observations = np.concatenate((ranges, range_rates))
    variances = np.concatenate((range_variances, range_rate_variances))

    if variances.shape != observations.shape:
        raise ValueError("Observation variances must match the observations")
    if np.any(variances <= 0) or np.any(np.isnan(variances)):
        raise ValueError("Observation variances must be positive")

    # Equation (30) of Markkanen et al. (2013), evaluated as a whitened
    # least-squares problem rather than by explicitly forming both inverses:
    # x_hat = (B.T @ Sigma^-1 @ B)^-1 @ B.T @ Sigma^-1 @ m.
    inverse_standard_deviations = 1.0 / np.sqrt(variances)
    weighted_theory = inverse_standard_deviations[:, None] * theory_matrix
    weighted_observations = inverse_standard_deviations * observations
    parameters = np.linalg.lstsq(weighted_theory, weighted_observations, rcond=None)[0]

    fitted_ranges = range_theory @ parameters
    fitted_rates = range_rate_theory @ parameters
    range_residuals = ranges - fitted_ranges
    rate_residuals = range_rates - fitted_rates
    # Equation (31): covariance(x_hat) = (B.T @ Sigma^-1 @ B)^-1.
    parameter_covariance = np.linalg.inv(weighted_theory.T @ weighted_theory)

    return TrajectoryFit(
        reference_time=reference_time,
        parameters=parameters.astype(np.float64),
        parameter_covariance=parameter_covariance.astype(np.float64),
        fitted_ranges=fitted_ranges,
        fitted_range_rates=fitted_rates,
        range_residuals=range_residuals,
        range_rate_residuals=rate_residuals,
    )
