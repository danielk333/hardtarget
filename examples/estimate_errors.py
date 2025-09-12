#!/usr/bin/env python
"""
Estimate errors with simulation
================================

"""

import numpy as np
import matplotlib.pyplot as plt
import argparse

from hardtarget.simulation.errors import monte_carlo_sample_errors

parser = argparse.ArgumentParser()
parser.add_argument("sim_output_path")
parser.add_argument("-c", "--clobber", action="store_true")
args = parser.parse_args()

errors = monte_carlo_sample_errors(
    snr_db=20,
    range0=2000e3,
    vel0=0.3e3,
    acel0=-0.1e3,
    samples=50,
    clobber=args.clobber,
    output_path=args.sim_output_path,
)

cov = errors["cov"]
print(f"Covariance matrix: {cov}")
print(f"Standard deviations: {np.sqrt(np.diag(cov))}")

fig, axes = plt.subplots(2, 2)
axes = axes.flatten()
axes[0].hist(errors["delta_r"])
axes[0].set_xlabel("Range [m]")
axes[1].hist(errors["delta_v"])
axes[1].set_xlabel("Velocity [m/s]")
axes[2].hist(errors["delta_a"])
axes[2].set_xlabel("Acceleration [m/s^2]")
axes[3].hist(errors["delta_snr"])
axes[3].set_xlabel("SNR [1]")

fig, axes = plt.subplots(2, 2)
axes = axes.flatten()
axes[0].plot(errors["delta_r"], ".")
axes[0].set_ylabel("Range [m]")
axes[1].plot(errors["delta_v"], ".")
axes[1].set_ylabel("Velocity [m/s]")
axes[2].plot(errors["delta_a"], ".")
axes[2].set_ylabel("acceleration [m/s^2]")
axes[3].plot(errors["delta_snr"], ".")
axes[3].set_ylabel("SNR [1]")


plt.show()
