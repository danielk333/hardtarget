# Estimate errors with simulation
# ---

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hardtarget.data_simulation.errors import monte_carlo_sample_errors

tmp_path = tempfile.TemporaryDirectory()
results = monte_carlo_sample_errors(
    snr_db=np.linspace(10, 40, num=10),
    range0=2000e3,
    vel0=0.3e3,
    acel0=-0.1e3,
    samples=100,
    clobber=True,
    output_path=Path(tmp_path.name),
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
axes[3].plot(np.log10(results["snr"]) * 10, ".")
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
tmp_path.cleanup()

"""
sample_rates = [500_000, 1_000_000, 2_000_000]
num = len(sample_rates)

results = []
for ind in range(num):
    result = monte_carlo_sample_errors(
        snr_db=40,
        range0=2000e3,
        vel0=0.3e3,
        acel0=-0.1e3,
        samples=50,
        clobber=args.clobber,
        output_path=pathlib.Path(args.sim_output_path) / f"sim{ind}",
        sample_rate=sample_rates[ind],
    )
    results.append(result) """
