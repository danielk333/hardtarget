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

results = monte_carlo_sample_errors(
    snr_db=np.linspace(10, 40, num=10),
    # snr_db=10,
    range0=2000e3,
    vel0=0.3e3,
    acel0=-0.1e3,
    samples=100,
    clobber=args.clobber,
    output_path=args.sim_output_path,
)

cov = results["cov"]
print(f"Covariance matrix: {cov}")
print(f"Standard deviations: {np.sqrt(np.diag(cov))}")

fig, axes = plt.subplots(2, 2)
axes = axes.flatten()
axes[0].plot(results["range"], ".")
axes[0].set_xlabel("Sample")
axes[0].set_ylabel("Range [m]")
axes[1].plot(results["range_rate"], ".")
axes[1].set_xlabel("Sample")
axes[1].set_ylabel("Velocity [m/s]")
axes[2].plot(results["acceleration"], ".")
axes[2].set_xlabel("Sample")
axes[2].set_ylabel("Acceleration [m/s^2]")
axes[3].plot(np.log10(results["snr"])*10, ".")
axes[3].set_xlabel("Sample")
axes[3].set_ylabel("SNR [dB]")

fig, axes = plt.subplots(2, 2)
axes = axes.flatten()
axes[0].hist(results["delta_r"])
axes[0].set_xlabel("Range [m]")
axes[1].hist(results["delta_v"])
axes[1].set_xlabel("Velocity [m/s]")
axes[2].hist(results["delta_a"])
axes[2].set_xlabel("Acceleration [m/s^2]")
axes[3].hist(results["delta_snr"])
axes[3].set_xlabel("SNR [1]")

fig, axes = plt.subplots(2, 2)
axes = axes.flatten()
axes[0].plot(results["delta_r"], ".")
axes[0].set_ylabel("Range [m]")
axes[1].plot(results["delta_v"], ".")
axes[1].set_ylabel("Velocity [m/s]")
axes[2].plot(results["delta_a"], ".")
axes[2].set_ylabel("acceleration [m/s^2]")
axes[3].plot(results["delta_snr"], ".")
axes[3].set_ylabel("SNR [1]")


plt.show()
