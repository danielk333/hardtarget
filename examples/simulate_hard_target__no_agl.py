#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import hardtarget
import argparse
import logging

logger = logging.getLogger("hardtarget.example_simulation")
hardtarget.profiling.setup_loggers(stdout=True, verbosity=1)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--action", required=False,
    choices=("all", "simulate", "analyse", "plot"), default="all",
)
args = parser.parse_args()

gmfimpl_o, gmfmethod_o = (None, None)

# gmfimpl, gmfmethod = ("numpy", "grid-fast-gmf")
gmfimpl, gmfmethod = ("numpy", "grid-fast-dpt")
# gmfimpl_o, gmfmethod_o = ("numpy", "optimize-grid-gmf")
# gmfimpl_o, gmfmethod_o = ("numpy", "optimize-scipy-gmf")
# gmflib = "c"
# gmflib = "numpy"
# gmflib = "numpy_daf"

base_path = pathlib.Path("/home/danielk/data/spade")
drf_path = base_path / "beamparks_raw" / "leo_bpark_2.1u_NO@uhf_drf_sim"
rx_channel = "sim"
config_path = pathlib.Path("./examples/cfg/sim_test.ini").resolve()
output_path = base_path / "beamparks_analyzed" / f"leo_bpark_2.1u_NO@uhf_{gmfimpl}_{gmfmethod}_sim"

if gmfimpl_o is not None:
    gmfimpl, gmfmethod = gmfimpl_o, gmfmethod_o

t_start = 0
echo_len = 5.0
t_abs = np.arange(t_start, t_start + echo_len + 0.1, 0.1)
t_rel = t_abs - t_start
peak_SNR = 10**(15.0/10.0)


# TODO: add system for multiple targets, each with its own data function
simulation_params = {
    "epoch": "2021-04-12T12:15:40",
    "start_time": 0,
    "end_time": 15.0,
    "target_start_time": t_abs[0],
    "target_end_time": t_abs[-1],
    "noise_sigma": 0,
    # "noise_sigma": 1.0,
    "tx_amp": np.sqrt(peak_SNR),
}

range0 = 2000e3
vel0 = 0.4e3
acel0 = -0.20e3

sim_r = range0 + vel0*t_rel + acel0*0.5*t_rel**2
sim_v = vel0 + acel0*t_rel
sim_a = np.ones_like(t_rel)*acel0


def range_function(t):
    inds = np.logical_and(t >= t_abs[0], t <= t_abs[-1])
    if np.any(inds):
        _t = t - t_start
        # return range0 + vel0*t + acel0*(1/2)*t**2 + dacel0*(1/6)*t**3
        return range0 + vel0*_t[inds] + acel0*0.5*_t[inds]**2
    else:
        return np.nan


def snr_function(t):
    inds = np.logical_and(t >= t_abs[0], t <= t_abs[-1])
    if np.any(inds):
        # _t = t - t_start
        # return peak_SNR*np.exp(-(_t - echo_len/2)**2/2.0**2)
        return peak_SNR
    else:
        return np.nan


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
    logger.info("Simulating")
    hardtarget.simulation.drf(
        drf_path,
        range_function,
        simulation_params,
        experiment_params,
        snr_function=snr_function,
        dtype=np.complex64,
        clobber=True,
    )

reader, params = hardtarget.drf_utils.load_hardtarget_drf(drf_path)

all_params = hardtarget.load_gmf_params(drf_path, config_path)

for key, val in all_params["PRO"].items():
    print(f"{key}: {val}")

look_time_ind = np.argmin(np.abs(t_abs - t_start - echo_len/2))
look_time = t_abs[look_time_ind]
range_true = range_function(look_time)[0]
range_ind = np.argmin(np.abs(all_params["DER"]["ranges"] - range_true))
acc_ind = np.argmin(np.abs(all_params["DER"]["accelerations"] - acel0))

print(f"Look at time: {look_time} s")
print("Best range gate: ", all_params["DER"]["rel_rgs"][range_ind])
print("Range gate: ", all_params["DER"]["ranges"][range_ind]*1e-3, " km vs ", range_true*1e-3, " km")
print("Best range gate index: ", all_params["DER"]["rgs"][range_ind])

if args.action in ("all", "analyse"):
    logger.info("Analysing")
    # process
    results = hardtarget.compute_gmf(
        rx=(drf_path, rx_channel),
        tx=(drf_path, rx_channel),
        config=config_path,
        gmf_method=gmfmethod,
        gmf_implementation=gmfimpl,
        clobber=True,
        output=output_path,
        progress=True,
        subprogress=True,
    )

if args.action in ("all", "plot"):
    logger.info("Plotting")

    fig, axes = plt.subplots(3, 1)
    axes[0].plot(t_abs, 1e-3*sim_r, c="red")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Range [km]")
    axes[1].plot(t_abs, 1e-3*sim_v, c="red")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Velocity [km/s]")
    axes[2].plot(t_abs, sim_a, c="red")
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
        data["t"] -= np.min(data["t"])

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
        axes[0, 0].plot(t_abs, sim_r*1e-3*0.5, c="red")
        axes[0, 0].set_xlabel("Time [s]")
        axes[0, 0].set_ylabel("range [km]")

        h01 = axes[0, 1].plot(data["t"], data["range_rate_peak"]*1e-3*0.5, c="blue")
        axes[0, 1].plot(t_abs, sim_v*1e-3*0.5, c="red")
        axes[0, 1].set_xlabel("Time [s]")
        axes[0, 1].set_ylabel("range rate [km/s]")

        h10 = axes[1, 0].plot(data["t"], data["acceleration_peak"]*0.5, c="blue")
        axes[1, 0].plot(t_abs, sim_a*0.5, c="red")
        axes[1, 0].set_xlabel("Time [s]")
        axes[1, 0].set_ylabel("acceleration [m/s^2]")
        axes[1, 0].set_ylim([-300, 300])

        h11 = axes[1, 1].plot(data["t"], np.sqrt(snr))
        axes[1, 1].set_xlabel("Time [s]")
        axes[1, 1].set_ylabel("sqrt(ENR) [1]")

    plt.show()
