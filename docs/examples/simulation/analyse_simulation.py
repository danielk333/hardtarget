# # Analyse simulation data
# ---
# Defining and simulating an object moving across the beam at a specified range.
# Thereafter running each analysis method on the data.


import datetime as dt
import tempfile
from pathlib import Path

import numpy as np
import numpy.typing as npt
from matplotlib import gridspec
from matplotlib import pyplot as plt
from radardef import Mu
from radardef.radar_stations.mu.experiments import mu_exp
from radardef.types.types import ExpDef
from spacecoords import interpolation

from hardtarget import (
    direction_of_arrival,
    echo_search,
    load_analysed_data,
    plotting,
    target_estimation,
)
from hardtarget.data_simulation import simulate_h5
from hardtarget.interferometry.types import DOACfgParams, DOAProParams, DOAVars

# ##Prerequisites
# ---

# Station and experiment specs
station = Mu()
experiment = mu_exp
# Temporary directory
tmp_dir = tempfile.TemporaryDirectory()
# Data directory
data_path = Path(tmp_dir.name) / "data"

# ## Trajectory function
# ---
# Object moving across the beam at a range of 110 km


def trajectory_func(t: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:

    r0 = 100e3
    # simple trajectory moving the object from one end of the beam to the next
    path = np.array([[-1, 0.5, 1], [0, 0, 0.95], [1, -0.5, 0.90]]) * r0
    t_samps = np.linspace(0, 1.0, path.shape[0])
    fun = interpolation.Linear(states=path.T, t=t_samps)

    trajectory = fun.get_state(t)

    r = np.linalg.norm(trajectory, axis=0)
    two_way_range = 2 * r
    rx_k_vecs = trajectory / r

    # trajectory
    return two_way_range, rx_k_vecs


# ## Simulate data
# ---
# The mu radar will be used as the source of the simulation.

# Start and end time of measurement and when the object is visible
measurement_start = dt.datetime.now()
measurement_end = measurement_start + dt.timedelta(seconds=1)
target_start_time_us = 250000
target_end_time_us = 750000
# Simulate data and write to file
simulate_h5(
    output_dir=data_path,
    exp_params=experiment,
    start_time=measurement_start,
    end_time=measurement_end,
    target_start_time=target_start_time_us,
    target_end_time=target_end_time_us,
    target_relative_time=True,
    trajectory_function=trajectory_func,
    noise_sigma=0.1,
    beam=station.beam,
    beam_params=station.beam_parameters,
)
# Load the raw data so it's available if needed.
data = station.load_data(data_path, experiment=mu_exp)
assert data is not None, "Not possible to load data"

# ## Analysis configuration
# ---
# As we will analyse the data using several different method we configure it using a dict
# as we then can add all parameters at one time and do not need to write to any file.

cfg = {
    "n_ipp": 1,
    "ipp_offset": 0,
    "min_range_gate": 81,
    "max_range_gate": 140,
    "range_gate_step": 1,
    "num_cohints_per_file": 500,
    "tx_amp_limit": 0.2,
    "min_acceleration": 0,
    "max_acceleration": 0,
    "range_gate_sub_resolution": 10,
    "frequency_decimation": 1,
    "acceleration_steps": 1,
    "doppler_freq_min": -30000,
    "doppler_freq_max": 5000,
    "doppler_freq_step": 1000,
    "elevation_limit": 0,
    "resolution": 150,
}

# ## Echo search
# ---
# Run an echo search and see that we have a 100% match of an event.

output_path = Path(tmp_dir.name) / "echo_search"
echo_search(
    data=data,
    config=cfg,
    output=output_path,
)

# ### Plot echo search results
# As we can see in the plots below there is an echo of an object.

# load data
data_generator = load_analysed_data(output_path)
out_data, exp_params, cfg_params, pro_params = list(data_generator)[0]
#
fig, ax = plt.subplots(2, 2)
plotting.plot_echo_search(ax, exp_params, out_data, limit=0.6)
_, handles = plotting.rti(
    ax[1, 1],
    data,
    axis_units=True,
    log=True,
    relative_time=True,
    colorbar=False,
)
fig.set_size_inches(10, 10)

# ## Target estimation
# ---
# Estimate the range-rate and range of the object.

output_path = Path(tmp_dir.name) / "target_estimation"
target_estimation(
    data=data,
    config=cfg,
    output=output_path,
    exp_params=experiment,
)

# ### Plot target estimation results
# Target estimation we can visualise with several different plots.

# load data,
data_generator = load_analysed_data(output_path)
out_data, exp_params, cfg_params, pro_params = list(data_generator)[0]

# Plot peaks, this visualizes the peaks  of the hardtargets range, velocity
# and acceleration over time (red is marking the detections). Furthermore the the signal to noise ratio is
# shown.

fig, axes = plt.subplots(2, 2)
plotting.target_estimation_plots.plot_peaks(
    axes,
    out_data,
    exp_params,
    cfg_params,
    pro_params,
    snr_dB_limit=15.0,
)
fig.set_size_inches(10, 10)

# Plot detections, this visualizes only the hardtarget detections from the measurement
# (red dots from previous plot).
# Additionaly it also adds the acceleration and range gate relative to the range for each detection.

fig, axes = plt.subplots(2, 3)
plotting.target_estimation_plots.plot_detections(
    axes,
    out_data,
    exp_params,
    cfg_params,
    pro_params,
)
fig.set_size_inches(12, 10)

# Plot map, this visualizes the hardtarget at the specific range gates,
# aswell the estimated noise relative to range.

fig = plt.figure()
gs = gridspec.GridSpec(2, 2, figure=fig)
axes = [
    fig.add_subplot(gs[0, :]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
]
plotting.target_estimation_plots.plot_map(
    axes,
    out_data,
    exp_params,
    cfg_params,
    pro_params,
)
fig.set_size_inches(10, 10)
# Store ranges as it will be needed to visualise the true trajectory of the object
ranges = out_data.r_vec

# ## Direction of arrival
# ---
# Determine where the object traveled from, at first we run the analysis.

output_path = Path(tmp_dir.name) / "direction_of_arrival"
direction_of_arrival(
    data=data,
    config=cfg,
    array_beam=station.beam,
    beam_params=station.beam_parameters,
    output=output_path,
    relative_time=True,
)

# ### Plot direction of arrival results
# Load analysed data, here we also

# load data
data_generator = load_analysed_data(output_path)
output: tuple[DOAVars, ExpDef, DOACfgParams, DOAProParams] = list(data_generator)[0]
out_data, exp_params, cfg_params, pro_params = output
# plot
fig, ax = plt.subplots(2, 2)
ax = plotting.plot_direction_of_arrival(ax, out_data, pro_params, 10.0)
fig.set_size_inches(10, 10)


# ## Determine the true trajectory of the object
# ---
# Using the determined range and the direction of arrival we can estimate the true trajectory of the object,
# Here we visualize it next to the simulated trajectory and the beam pointing direction.

fig = plt.figure()
ax = plt.axes(projection="3d")
ax = plotting.plot_true_vs_estimated_trajectory(
    ax=ax,
    exp_params=exp_params,
    trajectory_func=trajectory_func,
    doa_peaks=out_data.peak,
    doa_azimuth=out_data.azimuth,
    doa_elevation=out_data.elevation,
    ranges=ranges,
    pointing=np.array([data.pointing(0)[0], data.pointing(0)[1]]),
    measurement_start=measurement_start,
    measurement_end=measurement_end,
    target_start_us=target_start_time_us,
    target_end_us=target_end_time_us,
    detection_limit=10.0,
)
fig.set_size_inches(10, 10)

plt.show()
tmp_dir.cleanup()
