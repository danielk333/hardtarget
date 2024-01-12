#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import hardtarget

range0 = 2000e3
vel0 = 0.1e3
accel_ph = 1.0/2.0
acel0 = 0.04e3

target = "/home/danielk/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf_sim"

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
    "snrs": np.ones_like(t)*20,
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

hardtarget.simulation.drf(target, simulation_data, simulation_params, experiment_params, clobber=True)

reader, params = hardtarget.drf_utils.load_hardtarget_drf(target)


fig, axes = plt.subplots(3, 1)
axes[0].plot(t, simulation_data["ranges"])
axes[1].plot(t, simulation_data["velocities"])
axes[2].plot(t, simulation_data["accelerations"])

fig, ax = plt.subplots()
ax, handles = hardtarget.plotting.rti(
    ax,
    reader,
    params,
)

plt.show()
