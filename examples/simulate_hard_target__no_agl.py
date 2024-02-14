#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import hardtarget
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--action", required=False,
    choices=("all", "simulate", "analyse", "plot"), default="all",
)
args = parser.parse_args()


range0 = 2000e3
vel0 = 0.4e3
acel0 = -0.20e3

# gmflib = "cuda"
# gmflib = "c"
gmflib = "numpy"
# gmflib = "numpy_daf"

base_path = pathlib.Path("/home/danielk/data/spade")
drf_path = base_path / "beamparks_raw" / "leo_bpark_2.1u_NO@uhf_drf_sim"
rx_channel = "sim"
config_path = pathlib.Path("./examples/cfg/sim_test.ini").resolve()
output_path = base_path / "beamparks_analyzed" / f"leo_bpark_2.1u_NO@uhf_{gmflib}_sim"

t_start = 0
echo_len = 5.0
t_abs = np.arange(t_start, t_start + echo_len, 0.1)
t = t_abs - t_start
peak_SNR = 10**(15.0/10.0)

simulation_params = {
    "epoch": "2021-04-12T12:15:40",
    "start_time": 0,
    "end_time": 15.0,
    "noise_sigma": 0,
    # "noise_sigma": 1.0,
    "tx_amp": np.sqrt(peak_SNR),
}

simulation_data = {
    "ranges": range0 + vel0*t + acel0*0.5*t**2,
    "velocities": vel0 + t*acel0,
    "accelerations": np.ones_like(t)*acel0,
    # "snr": peak_SNR*np.exp(-(t - echo_len/2)**2/2.0**2),
    "snr": peak_SNR*np.ones_like(t),
    "times": t_abs,
}

experiment_params = {
    "sample_rate": 1000000,
    "ipp": 20000,
    "tx_pulse_length": 1920.0,
    "tx_start": 82.0,
    "tx_end": 2002.0,
    "rx_start": 0,
    "rx_end": 20000,
    "cal_on": 19900.0,
    "cal_off": 19997.0,
    "frequency": 929.6,
    "baud_length": 30.0,
    "code": hardtarget.load_radar_code("leo_bpark"),
}

for key, val in experiment_params.items():
    print(f"{key}: {val}")

if args.action in ("all", "simulate"):
    hardtarget.simulation.drf(drf_path, simulation_data, simulation_params, experiment_params, clobber=True)

reader, params = hardtarget.drf_utils.load_hardtarget_drf(drf_path)

all_params = hardtarget.load_gmf_params(drf_path, config_path)
look_time_ind = np.argmin(np.abs(t - echo_len/2))
look_time = t_abs[look_time_ind]
range_true = simulation_data["ranges"][look_time_ind]
range_ind = np.argmin(np.abs(all_params["DER"]["ranges"] - range_true))
acc_ind = np.argmin(np.abs(all_params["DER"]["accelerations"] - acel0))

print(f"Look at time: {look_time} s")
print("Best range gate: ", all_params["DER"]["rel_rgs"][range_ind])
print("Range gate: ", all_params["DER"]["ranges"][range_ind]*1e-3, " km vs ", range_true*1e-3, " km")
print("Best range gate index: ", all_params["DER"]["rgs"][range_ind])

if args.action in ("all", "analyse"):
    # process
    results = hardtarget.compute_gmf(
        rx=(drf_path, rx_channel),
        tx=(drf_path, rx_channel),
        config=config_path,
        job={"idx": 0, "N": 1},
        gmflib=gmflib,
        gmf_optimize_lib="no",
        clobber=True,
        output=output_path,
        progress=True,
        subprogress=True,
    )

if args.action in ("all", "plot"):

    fig, axes = plt.subplots(3, 1)
    axes[0].plot(t_abs, 1e-3*simulation_data["ranges"], c="red")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Range [km]")
    axes[1].plot(t_abs, 1e-3*simulation_data["velocities"], c="red")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Velocity [km/s]")
    axes[2].plot(t_abs, simulation_data["accelerations"], c="red")
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

    data_generator = hardtarget.load_gmf_out(output_path)
    for data, meta in data_generator:
        fig, axes = plt.subplots(2, 2)
        hardtarget.plotting.gmf.plot_peaks(axes, data, meta)
        fig, axes = plt.subplots(2, 3)
        hardtarget.plotting.gmf.plot_detections(axes, data, meta)
        fig, axes = plt.subplots(3, 1)
        hardtarget.plotting.gmf.plot_map(axes, data, meta)

        fig, axes = plt.subplots(2, 2)
        snr = hardtarget.noise.snr(data["gmf"], data["nf_range"])
        r_inds = np.argmax(data["gmf"], axis=1)
        coh_inds = np.arange(data["gmf"].shape[0])
        snr = snr[coh_inds, r_inds]

        h00 = axes[0, 0].plot(data["t"], data["range_peak"]*1e-3*0.5, c="blue")
        axes[0, 0].plot(t_abs, simulation_data["ranges"]*1e-3*0.5, c="red")
        axes[0, 0].set_xlabel("Time [s]")
        axes[0, 0].set_ylabel("range [km]")

        h01 = axes[0, 1].plot(data["t"], data["range_rate_peak"]*1e-3*0.5, c="blue")
        axes[0, 1].plot(t_abs, simulation_data["velocities"]*1e-3*0.5, c="red")
        axes[0, 1].set_xlabel("Time [s]")
        axes[0, 1].set_ylabel("range rate [km/s]")

        h10 = axes[1, 0].plot(data["t"], data["acceleration_peak"]*0.5, c="blue")
        axes[1, 0].plot(t_abs, simulation_data["accelerations"]*0.5, c="red")
        axes[1, 0].set_xlabel("Time [s]")
        axes[1, 0].set_ylabel("acceleration [m/s^2]")

        h11 = axes[1, 1].plot(data["t"], np.sqrt(snr))
        axes[1, 1].set_xlabel("Time [s]")
        axes[1, 1].set_ylabel("sqrt(ENR) [1]")

    plt.show()
