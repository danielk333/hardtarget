# # Fit a trajectory and estimate internal observation variances
# ---
# Range and range-rate observations can be combined using the cubic range
# model from Markkanen et al. (2013). The variance estimates below are derived
# from residuals from the fitted trajectory.

import matplotlib.pyplot as plt
import numpy as np

from hardtarget.target_estimation import fit_trajectory

# Random numbers and constants

rng = np.random.default_rng(87346)
reference_time = 0.0
parameters = np.array([850e3, -4.2e3, 140.0, -0.6])

# We define a arbitrary trajectory function.


def trajectory(times):
    dt = times - reference_time
    r0, v0, acceleration0, jerk0 = parameters
    ranges = r0 + v0 * dt + 0.5 * acceleration0 * dt**2 + jerk0 * dt**3 / 6.0
    range_rates = v0 + acceleration0 * dt + 0.5 * jerk0 * dt**2
    return ranges, range_rates


# Extract range and range rate at the given trajectory

times = np.linspace(-4.0, 4.0, 120)
true_ranges, true_range_rates = trajectory(times)

# Example of a possible observed range and range rate from the trajectory

t_unit = (times - times.min()) / (times.max() - times.min())
range_sigma = 10 + 20 * (2 * t_unit - 1) ** 2
range_rate_sigma = 0.3 + 1.0 * (2 * t_unit - 1) ** 2
observed_ranges = true_ranges + range_sigma * rng.normal(size=times.size)
observed_range_rates = true_range_rates + range_rate_sigma * rng.normal(size=times.size)

# Fit the observed range and range rate.

fit = fit_trajectory(
    times,
    observed_ranges,
    times,
    observed_range_rates,
    range_variances=range_sigma**2,
    range_rate_variances=range_rate_sigma**2,
    reference_time=reference_time,
)

# From this we can se the fitted parameters:

print(f"Fitted [range, range-rate, acceleration, jerk]: {fit.parameters}")
print(f"[range, range-rate, acceleration, jerk] error : {fit.parameters - parameters}")

# Visualization

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
#
axes[0, 0].plot(times, observed_ranges * 1e-3, ".", label="Observations")
axes[0, 0].plot(times, fit.range(times) * 1e-3, label="Fit")
axes[0, 0].set_ylabel("Range [km]")
axes[0, 0].legend()
axes[0, 0].set_title("Range observation vs fit")
#
axes[0, 1].plot(times, observed_range_rates * 1e-3, ".", label="Observations")
axes[0, 1].plot(times, fit.range_rate(times) * 1e-3, label="Fit")
axes[0, 1].set_ylabel("Range-rate [km/s]")
axes[0, 1].legend()
axes[0, 1].set_title("Range-rate observation vs fit")
#
axes[1, 0].plot(times, fit.range_residuals, ".")
axes[1, 0].plot(times, range_sigma, c="r", ls="--", label="1 sigma")
axes[1, 0].plot(times, -range_sigma, c="r", ls="--")
axes[1, 0].set(xlabel="Time [s]", ylabel="Range residual [m]")
axes[1, 0].legend()
axes[1, 0].set_title("Range residuals")
#
axes[1, 1].plot(times, fit.range_rate_residuals, ".")
axes[1, 1].plot(times, range_rate_sigma, c="r", ls="--", label="1 sigma")
axes[1, 1].plot(times, -range_rate_sigma, c="r", ls="--")
axes[1, 1].set(xlabel="Time [s]", ylabel="Range-rate residual [m/s]")
axes[1, 1].legend()
axes[1, 1].set_title("Range-rate residuals")
#
fig.tight_layout()
plt.show()
