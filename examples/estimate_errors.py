#!/usr/bin/env python
"""
Estimate errors with simulation
================================

"""

import matplotlib.pyplot as plt
import argparse

from hardtarget.simulation.errors import monte_carlo_sample_errors

parser = argparse.ArgumentParser()
parser.add_argument("sim_output_path")
args = parser.parse_args()

errors = monte_carlo_sample_errors(
    snr_db=10,
    range0=2000e3,
    vel0=0.3e3,
    acel0=-0.1e3,
    samples=200,
    clobber=False,
    output_path=args.sim_output_path,
)


fig, axes = plt.subplots(3, 1)
axes[0].hist(errors["delta_r"])
axes[0].set_xlabel("Range [m]")
axes[1].hist(errors["delta_v"])
axes[1].set_xlabel("Velocity [m/s]")
axes[2].hist(errors["delta_a"])
axes[2].set_xlabel("Acceleration [m/s^2]")


fig, axes = plt.subplots(3, 1)
axes[0].plot(errors["delta_r"], ".")
axes[0].set_xlabel("Range [m]")
axes[1].plot(errors["delta_v"], ".")
axes[1].set_xlabel("Velocity [m/s]")
axes[2].plot(errors["delta_a"], ".")
axes[2].set_xlabel("Acceleration [m/s^2]")


plt.show()
