# # Music - Error per resolution level

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from radardef.radar_stations import Mu
from radardef.radar_stations.mu.experiments import mu_exp
from tqdm import tqdm

from hardtarget.analyse import direction_of_arrival
from hardtarget.data_simulation.simulate_h5 import simulate_h5
from hardtarget.interferometry.types import DOACfgParams

# Define station and experiment
station = Mu()
exp_def = mu_exp

# Define data to test, we simulate a object stuck at a specific location, for just one ipp
target_location = np.array([0.323, 0.577, 1.0])


def trajectory_func(t: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    return np.ones((len(t),)) * 210e3, np.repeat(np.atleast_2d(target_location), len(t), axis=0).T


res_steps = np.arange(10, 150, 2)
distributed_peaks = np.arange(1, 15, 1)

k_diff_x = np.zeros((len(res_steps), len(distributed_peaks)))
k_diff_y = np.zeros((len(res_steps), len(distributed_peaks)))
k_diff_z = np.zeros((len(res_steps), len(distributed_peaks)))


x, y = np.meshgrid(distributed_peaks, res_steps)

with tempfile.TemporaryDirectory() as tmp_dir:
    simulation_path = Path(tmp_dir) / "sim_data"
    simulate_h5(
        output_dir=simulation_path,
        exp_params=exp_def,
        start_time=0,
        end_time=exp_def.t_ipp_usec,
        target_start_time=0,
        target_end_time=exp_def.t_ipp_usec,
        trajectory_function=trajectory_func,
        beam=station.beam,
        beam_params=station.beam_parameters,
        noise_sigma=0.1,
    )
    pbar = tqdm(total=len(res_steps) * len(distributed_peaks))
    for res_i, resolution in enumerate(res_steps):
        for peak_i, n_peaks in enumerate(distributed_peaks):
            result = direction_of_arrival(
                data=simulation_path,
                config=DOACfgParams(
                    n_ipp=1,
                    min_range_gate=81,
                    max_range_gate=166,
                    elevation_limit=0,
                    resolution=resolution,
                    distributed_peaks=n_peaks,
                ),
                array_beam=station.beam,
                beam_params=station.beam_parameters,
            )["data"]

            out_data, _, _, _ = result[0]
            k_diff = np.abs(target_location - out_data.k_vec[0])
            k_diff_x[res_i, peak_i] = k_diff[0]
            k_diff_y[res_i, peak_i] = k_diff[1]
            k_diff_z[res_i, peak_i] = k_diff[2]
            pbar.update(1)

ax = plt.figure().add_subplot(projection="3d")
ax.plot_surface(x, y, k_diff_x, label="x", edgecolor="royalblue", lw=0.5, rstride=8, cstride=8, alpha=0.3)
ax.plot_surface(x, y, k_diff_y, label="y", edgecolor="orange", lw=0.5, rstride=8, cstride=8, alpha=0.3)
ax.plot_surface(x, y, k_diff_z, label="z", edgecolor="green", lw=0.5, rstride=8, cstride=8, alpha=0.3)

ax.set_ylabel("Resolution")
ax.set_xlabel("N distributed peaks")
ax.set_zlabel("Error")
ax.legend()
plt.show()
