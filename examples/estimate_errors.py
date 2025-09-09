#!/usr/bin/env python
"""
Estimate errors with simulation
================================

"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import hardtarget
import argparse
import logging

logger = logging.getLogger("hardtarget.error_estimation")
hardtarget.profiling.setup_loggers(stdout=True, verbosity=1)
config = Path(__file__).parent.absolute() / "cfg" / "sim_test.ini"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--action",
    required=False,
    choices=("all", "simulate", "analyse", "plot"),
    default="all",
)
parser.add_argument("sim_output_path", type=Path)
parser.add_argument("analysis_output_path", type=Path)
parser.add_argument("--SNRdB", type=float, default=10.0)
parser.add_argument("--samples", type=int, default=200)
parser.add_argument("--config", type=Path, default=config)
parser.add_argument("--lib", default="c")
parser.add_argument("--method", default="fgmf")

# define A*G(k) = 1, Nc = 1
# snr = N/(2*sigmac**2)
# N = samples integrated

args = parser.parse_args()

gmfimpl, gmfmethod = (args.lib, args.method)
args = parser.parse_args()

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
    "radar_frequency": 929.6,
    "baud_length": 30.0,
    "code": hardtarget.load_radar_code("leo_bpark"),
}

params_pro = hardtarget.configuration.load_gmf_processing_params(args.config)
experiment_params, params_pro, params_der = hardtarget.configuration.compute_derived_gmf_params(
    experiment_params, params_pro
)

coh_samples = experiment_params["tx_pulse_samps"] * params_pro["n_ipp"]
SNR = 10 ** (args.SNRdB / 10.0)
noise_sigma = np.sqrt(coh_samples / (2 * SNR))

t_start = 0
coh_int_time = params_pro["n_ipp"] * experiment_params["ipp"] * 1e-6
echo_len = coh_int_time * args.samples
t_abs = np.zeros(args.samples)
t_rel = t_abs - t_start


def time_modder(t):
    return t % coh_int_time


simulation_params = {
    "epoch": "2021-04-12T12:15:40",
    "start_time": t_start,
    "end_time": echo_len,
    "target_start_time": t_start,
    "target_end_time": echo_len,
    "noise_sigma": noise_sigma,
    "tx_amp": 1,
}

# r0 = range_function(np.array([t0], dtype=np.float64))[0]
# sn0 = snr_function(t0 + t_tx) if snr_function is not None else 1.0
# rg0 = np.round((r0 / scipy.constants.c) * sample_rate).astype(np.int64)
# rg_samp0 = rg0 + T_tx_start_samp

rx_channel = "sim"

range0 = 2000e3
vel0 = -0.4e3
acel0 = 0.20e3

sim_r = range0 + vel0 * t_rel + acel0 * 0.5 * t_rel**2
sim_v = vel0 + acel0 * t_rel
sim_a = np.ones_like(t_rel) * acel0


config_str = f"""

[signal-processing]
    n_ipp={n_ipp}
    ipp_offset=0
    min_range_gate=6640
    max_range_gate=6700
    min_acceleration=-300.0
    max_acceleration=300.0
    range_gate_step=1
    frequency_decimation={frequency_decimation}
    num_cohints_per_file=10
    node_gpus=1
    dpt_ipp_delay_parameter={tau_ipp}

"""


def range_function(t):
    _t = time_modder(t - t_start)
    return range0 + vel0 * _t + acel0 * 0.5 * _t ** 2


for key, val in experiment_params.items():
    print(f"{key}: {val}")

if args.action in ("all", "simulate"):
    logger.info("Simulating")
    hardtarget.simulation.drf(
        args.sim_output_path,
        range_function,
        simulation_params,
        experiment_params,
        chnl=rx_channel,
        dtype=np.complex64,
        clobber=True,
    )

reader, params = hardtarget.drf_utils.load_hardtarget_drf(args.sim_output_path)

all_params = hardtarget.load_gmf_params(args.sim_output_path, args.config)

for key, val in all_params["PRO"].items():
    print(f"{key}: {val}")

if args.action in ("all", "analyse"):
    logger.info("Analysing")
    # process
    results = hardtarget.compute_gmf(
        rx=(args.sim_output_path, rx_channel),
        tx=(args.sim_output_path, rx_channel),
        config=args.config,
        gmf_method=gmfmethod,
        gmf_implementation=gmfimpl,
        clobber=True,
        output=args.analysis_output_path,
        progress=True,
        subprogress=True,
    )

if args.action in ("all", "plot"):
    logger.info("Plotting")

    fig, axes = plt.subplots(3, 1)
    axes[0].plot(t_abs, 1e-3 * sim_r, c="red")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Range [km]")
    axes[1].plot(t_abs, 1e-3 * sim_v, c="red")
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

    data_generator = hardtarget.load_gmf_out(args.analysis_output_path)
    for data, meta in data_generator:
        data["t"] -= np.min(data["t"])

        fig, axes = plt.subplots(2, 2)
        hardtarget.plotting.gmf.plot_peaks(axes, data, meta)
        fig, axes = plt.subplots(2, 3)
        hardtarget.plotting.gmf.plot_detections(axes, data, meta)
        fig, axes = plt.subplots(3, 1)
        hardtarget.plotting.gmf.plot_map(axes, data, meta)

        fig, axes = plt.subplots(2, 2)
        meas_snr = hardtarget.noise.snr(data["gmf"], data["nf_range"])
        r_inds = np.argmax(data["gmf"], axis=1)
        coh_inds = np.arange(data["gmf"].shape[0])
        meas_snr = meas_snr[coh_inds, r_inds]

        h00 = axes[0, 0].plot(data["t"], data["range_peak"] * 1e-3 * 0.5, c="blue")
        axes[0, 0].plot(t_abs, sim_r * 1e-3 * 0.5, c="red")
        axes[0, 0].set_xlabel("Time [s]")
        axes[0, 0].set_ylabel("range [km]")

        h01 = axes[0, 1].plot(data["t"], data["range_rate_peak"] * 1e-3 * 0.5, c="blue")
        axes[0, 1].plot(t_abs, sim_v * 1e-3 * 0.5, c="red")
        axes[0, 1].set_xlabel("Time [s]")
        axes[0, 1].set_ylabel("range rate [km/s]")

        h10 = axes[1, 0].plot(data["t"], data["acceleration_peak"] * 0.5, c="blue")
        axes[1, 0].plot(t_abs, sim_a * 0.5, c="red")
        axes[1, 0].set_xlabel("Time [s]")
        axes[1, 0].set_ylabel("acceleration [m/s^2]")
        axes[1, 0].set_ylim([-300, 300])

        h11 = axes[1, 1].plot(data["t"], 10*np.log10(meas_snr))
        axes[1, 1].axhline(10*np.log10(SNR), c="r")
        axes[1, 1].set_xlabel("Time [s]")
        axes[1, 1].set_ylabel("SNR [dB]")

    plt.show()
