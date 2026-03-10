import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import radardef.radar_stations.eiscat.utils as radardef_utils
from radardef import RadarDef
from radardef.types import BoundParams, ExpParams
from scipy import constants

import hardtarget
from hardtarget.data_simulation import DRFSimParams, simulate_drf
from hardtarget.event_detection.types import XCorrCfgParams


def wip_event_detection():

    frequency_decimation = 10
    n_ipp = 1

    cfg = XCorrCfgParams(
        n_ipp=n_ipp,
        ipp_offset=0,
        min_range_gate=6640,
        max_range_gate=6750,
        min_acceleration=-300.0,
        max_acceleration=300.0,
        range_gate_step=1,
        frequency_decimation=frequency_decimation,
        num_cohints_per_file=10,
        node_gpus=1,
        doppler_freq_min=-30000,
        doppler_freq_max=5000,
        doppler_freq_step=1000,
    )

    exp_params = ExpParams(
        name="simulation",
        radar_frequency=929.6,
        t_ipp_usec=20000,
        ipp_samps=20000,
        sample_rate=1000000.0,
        t_samp_usec=1,
        rx_channels=["sim"],
        t_tx_start_usec=82.0,
        t_tx_end_usec=2001.0,
        t_rx_start_usec=0,
        t_rx_end_usec=20000,
        tx_channel="sim",
        tx_pulse_length=1919,
        t_cal_on_usec=19900.0,
        t_cal_off_usec=19997.0,
        wavelength=constants.c / (929.6 * 1e6),
        code=radardef_utils.load_radar_code("leo_bpark"),
    )

    bounds_params = BoundParams(
        ts_start_usec=1445511612800000,
        ts_end_usec=1445551228800000,
    )

    t_start = 0
    t_end = exp_params.t_ipp_usec * 750
    target_start = t_start + (exp_params.t_ipp_usec * 200)
    target_end = t_start + (exp_params.t_ipp_usec * 300)
    coh_int_len = exp_params.t_ipp_usec * n_ipp
    t_abs_us = np.arange(0, t_end + coh_int_len, coh_int_len)
    t_abs = t_abs_us * 1e-6

    simulation_params = DRFSimParams(
        epoch="2021-04-12T12:15:40",
        start_time_us=t_start,
        end_time_us=t_end,
        target_start_time_us=target_start,
        target_end_time_us=target_end,
        noise_sigma=0.2,
    )

    range0 = 2000e3
    vel0 = 0.4e3
    acel0 = 0.10e3

    def range_function(t):
        inds = np.logical_and(t >= t_abs[0], t <= t_abs[-1])
        if np.any(inds):
            return range0 + vel0 * t[inds] + acel0 * 0.5 * t[inds] ** 2
        else:
            return np.nan

    with (
        tempfile.TemporaryDirectory(suffix="_drf") as tmp_sim_path,
        tempfile.TemporaryDirectory() as tmp_analysis_path,
    ):
        simulate_drf(
            Path(tmp_sim_path),
            range_function,
            simulation_params,
            exp_params,
            bounds_params,
            snr_function=None,
            dtype=np.complex64,
            clobber=True,
            include_tx_signal=True,
        )

        # process
        _ = hardtarget.event_detection(
            path=Path(tmp_sim_path).resolve(),
            rx_channel="sim",
            config=cfg,
            start_time=simulation_params.start_time_us,
            end_time=simulation_params.end_time_us,
            relative_time=True,
            clobber=False,
            output=tmp_analysis_path,
            progress=False,
        )

        reader = RadarDef().load_data(Path(tmp_sim_path))
        assert reader is not None

        fig, ax = plt.subplots()
        # plot raw data power
        _, handles = hardtarget.plotting.rti(
            ax,
            reader,
            start_time=t_start,
            end_time=t_end,
            axis_units=True,
            log=True,
            relative_time=True,
            colorbar=False,
        )
        out, exp, _cfg, pro = list(hardtarget.load_analysed_data(tmp_analysis_path))[0]
        ax_twin = ax.twinx()
        ax_twin.plot(
            np.arange(t_start * 1e-6, t_end * 1e-6, step=coh_int_len * 1e-6),
            np.abs(out.max_peak),
        )

        plt.show()
