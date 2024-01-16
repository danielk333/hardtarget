#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import hardtarget

range0 = 2000e3
vel0 = 4.4e3
accel_ph = 1.0/2.0
acel0 = 0.04e3


gmflib = "cuda"

base_path = pathlib.Path("/home/danielk/data/spade")
drf_path = base_path / "beamparks_raw" / "leo_bpark_2.1u_NO@uhf_drf_sim"
rx_channel = "sign"
config_path = pathlib.Path("./examples/cfg/sim_test.ini").resolve()
output_path = base_path / "beamparks_analyzed" / f"leo_bpark_2.1u_NO@uhf_{gmflib}_sim"


simulation_params = {
    "epoch": "2021-04-12T12:15:40",
    "start_time": 0,
    "end_time": 15.0,
    "noise_sigma": 0,
}

t = np.arange(5, 10, 0.1)


simulation_data = {
    "ranges": range0 + vel0*t - np.sin(t*accel_ph)*acel0/accel_ph**2,
    "velocities": vel0 + np.cos(t*accel_ph)*acel0/accel_ph,
    "accelerations": np.sin(t*accel_ph)*acel0,
    "times": t,
}

exp_conf = hardtarget.load_expconfig("leo_bpark")
experiment_params = {}
for key in hardtarget.drf_utils.BOOL_PROPS:
    if key not in exp_conf["2.1u"]:
        continue
    experiment_params[key] = exp_conf.getboolean("2.1u", key)
for key in hardtarget.drf_utils.FLOAT_PROPS:
    if key not in exp_conf["2.1u"]:
        continue
    experiment_params[key] = exp_conf.getfloat("2.1u", key)
for key in hardtarget.drf_utils.INT_PROPS:
    if key not in exp_conf["2.1u"]:
        continue
    experiment_params[key] = exp_conf.getint("2.1u", key)

experiment_params["frequency"] = 929.6
experiment_params["baud_length"] = 30.0
experiment_params["code"] = hardtarget.load_radar_code("leo_bpark")

for key, val in experiment_params.items():
    print(f"{key}: {val}")

hardtarget.simulation.drf(drf_path, simulation_data, simulation_params, experiment_params, clobber=True)

reader, params = hardtarget.drf_utils.load_hardtarget_drf(drf_path)

# process
results = hardtarget.compute_gmf(
    rx=(drf_path, rx_channel),
    tx=(drf_path, rx_channel),
    config=config_path,
    job={"idx": 0, "N": 1},
    gmflib=gmflib,
    clobber=True,
    output=output_path,
    progress=True,
)


fig, axes = plt.subplots(3, 1)
axes[0].plot(t, 1e-3*simulation_data["ranges"])
axes[0].set_xlabel("Time [s]")
axes[0].set_ylabel("Range [km]")
axes[1].plot(t, 1e-3*simulation_data["velocities"])
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Velocity [km/s]")
axes[2].plot(t, simulation_data["accelerations"])
axes[2].set_xlabel("Time [s]")
axes[2].set_ylabel("Acceleration [m/s^2]")

fig, ax = plt.subplots()
ax, handles = hardtarget.plotting.rti(
    ax,
    reader,
    params,
    keep_tx=True,
    axis_units=True,
)

paths = hardtarget.plotting.gmf.collect_paths(
    output_path,
)

data_generator = hardtarget.plotting.gmf.yield_chunked_data(paths)
for data in data_generator:
    fig, axes = plt.subplots(2, 2)
    hardtarget.plotting.gmf.plot_peaks(axes, data)
    fig, axes = plt.subplots(2, 3)
    hardtarget.plotting.gmf.plot_detections(axes, data)
    fig, axes = plt.subplots(2, 2)
    hardtarget.plotting.gmf.plot_map(axes, data)
plt.show()
