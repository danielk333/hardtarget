import datetime as dt
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest
import scipy
from radardef.radar_stations import Mu
from radardef.types import ExpDef

import hardtarget
from hardtarget.constants import Impl
from hardtarget.data_simulation import simulate_h5, tx_signal_model
from hardtarget.echo_search import EchoSearchOutArgs, EchoSearchProParams
from hardtarget.echo_search.types import EchoSearchCfgParams
from hardtarget.libs import load_c_lib
from hardtarget.plotting import echo_search_plot, rti


@pytest.mark.parametrize("impl", [Impl.c])  # Impl.numpy
def test_echo(plot, impl: Impl):
    exp_def = ExpDef(
        name="Mu",
        radar_frequency=46.5,
        t_ipp_usec=3120,
        t_samp_usec=6,
        t_rx_start_usec=486,
        t_rx_end_usec=486 + 6 * 85,
        t_tx_start_usec=0,
        t_tx_end_usec=13 * 2 * 6,
        baud_length_usec=12,
        code=np.array(
            [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
            dtype=np.float64,
        ),
        rx_channels=np.arange(1, 26).tolist(),
        samples_per_file=10000000,
    )

    # ## Define a objects trajectory function,
    # ---
    r0: float = 110e3
    v0: float = 60e3
    a0: float = -0.05e3
    # Initial position over radar
    k0 = np.array([0, 0, 1])
    v_vec = np.array([0, 0, -1])

    def trajectory_func(t: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        # 3d position
        x_start = k0 * r0
        # Velocity vector
        distance_traveled = v0 * t + a0 * 0.5 * t**2

        trajectory = x_start[:, None] + v_vec[:, None] * distance_traveled[None, :]
        r = np.linalg.norm(trajectory, axis=0)
        two_way_range = 2 * r
        rx_k_vecs = trajectory / r

        # trajectory
        return two_way_range, rx_k_vecs

    def model_trajectory(t: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        velocity_mag = v0 + a0 * t
        distance_traveled = v0 * t + a0 * 0.5 * t**2
        x_start = k0 * r0
        # trajectory
        return x_start[:, None] + v_vec[:, None] * distance_traveled[None, :], v_vec[:, None] * velocity_mag[
            None, :
        ]

    # ## Simulate the experiment with the MU radars beam.
    # ---

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "sim"
        station = Mu()
        measurement_length_us = exp_def.t_ipp_usec * 50
        target_start_time_us = exp_def.t_ipp_usec * 20
        target_end_time_us = exp_def.t_ipp_usec * 40
        simulate_h5(
            output_dir=output_path,
            exp_def=exp_def,
            start_time=dt.datetime.now(),
            end_time=dt.datetime.now() + dt.timedelta(microseconds=measurement_length_us),
            target_start_time=target_start_time_us,
            target_end_time=target_end_time_us,
            target_relative_time=True,
            trajectory_function=trajectory_func,
            noise_sigma=0,
            beam=station.beam,
            beam_params=station.beam_parameters,
        )

        # Analyse data

        cfg = EchoSearchCfgParams(
            n_ipp=1,
            ipp_offset=0,
            min_range_gate=81,
            max_range_gate=140,
            num_cohints_per_file=10,
            cache=False,
            doppler_freq_min=-30000,
            doppler_freq_max=0,
            doppler_freq_step=10,
            range_gate_step=1,
        )

        output_analysis = Path(temp_dir) / "echo"
        _ = hardtarget.echo_search(
            data=Path(output_path).resolve(),
            config=cfg,
            relative_time=True,
            clobber=False,
            output=output_analysis,
            progress=False,
            exp_def=exp_def,
            implementation=impl,
        )
        output: tuple[EchoSearchOutArgs, ExpDef, EchoSearchCfgParams, EchoSearchProParams] = list(
            hardtarget.load_analysed_data(output_analysis)
        )[0]

        out, exp, _cfg, pro = output

        cohint_timepoints = np.arange(0, measurement_length_us, exp.t_ipp_usec * _cfg.n_ipp)
        target_filter = np.logical_and(
            cohint_timepoints >= target_start_time_us, cohint_timepoints <= target_end_time_us
        )
        no_target_filter = np.invert(target_filter)

        obs_range_rate = (out.best_doppler / (exp.radar_frequency * 1e6)) * scipy.constants.c
        t_offset_usec = exp.t_rx_start_usec - exp.t_tx_start_usec
        obs_range = (out.max_corr_delay * exp.t_samp_usec + t_offset_usec) * 1e-6 * scipy.constants.c

        t_v = np.arange(out.max_corr.shape[0]) * exp.t_ipp_usec * cfg.n_ipp * 1e-6
        pos, vel = model_trajectory(t_v)

        range_tx = np.linalg.norm(pos, axis=0)
        range_rx = range_tx
        target_range = range_tx + range_rx
        v_tx = np.sum(pos * vel, axis=0) / range_tx
        v_rx = np.sum(pos * vel, axis=0) / range_rx
        target_range_rate = v_tx + v_rx

        tx_rx_samps = target_range / (exp.t_samp_usec * 1e-6 * scipy.constants.c)
        tx_rx_start_delta = int((exp.t_rx_start_usec - exp.t_tx_start_usec) / exp.t_samp_usec)

        if plot:
            fig, ax = plt.subplots(3, 2)
            echo_search_plot.plot_echo_search(ax, exp, out)
            ax[0, 1].plot(np.arange(out.max_corr.shape[0]), target_range * 1e-3, "-r")
            ax[1, 1].plot(np.arange(out.max_corr.shape[0]), target_range_rate * 1e-3, "-r")
            rti(ax=ax[2, 0], data_loader=station.load_data(output_path, exp_def=exp))
            ax[2, 1].plot(
                np.arange(out.max_corr.shape[0]),
                tx_rx_samps - tx_rx_start_delta,
                "-r",
            )
            plt.show()

        assert out.max_corr[target_filter] == pytest.approx(1.0, abs=0.02)
        assert out.max_corr[no_target_filter] == pytest.approx(0.0, abs=0.02)
        assert obs_range_rate[target_filter] == pytest.approx(target_range_rate[target_filter], abs=15), (
            f"Mean range rate diff: {np.mean(np.abs(target_range_rate[target_filter] - obs_range_rate[target_filter]))} m/s"
        )
        assert obs_range[target_filter] == pytest.approx(target_range[target_filter], rel=1000), (
            f"Mean range diff: {np.mean(np.abs(target_range[target_filter] - obs_range[target_filter]))} m"
        )


def test_crosscorrelate_tx_model(plot):
    exp_def = ExpDef(
        name="Mu",
        radar_frequency=46.5,
        t_ipp_usec=3120,
        t_samp_usec=6,
        t_rx_start_usec=486,
        t_rx_end_usec=486 + 6 * 85,
        t_tx_start_usec=0,
        t_tx_end_usec=13 * 2 * 6,
        baud_length_usec=12,
        code=np.array(
            [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
            dtype=np.float64,
        ),
        rx_channels=np.arange(1, 26).tolist(),
        samples_per_file=10000000,
    )

    c_lib = load_c_lib()

    tx = (
        tx_signal_model(
            code=exp_def.code,
            baud_length_usec=exp_def.baud_length_usec,
            t_samp_usec=exp_def.t_samp_usec,
            tx_start_samp=0,
            ipp_samps=len(exp_def.code) * 2,
            read_length=len(exp_def.code) * 2,
        )
        .flatten()
        .astype(np.complex64)
    )
    rx = np.zeros(100, dtype=np.complex64)
    start_reciving = 34
    rx[start_reciving : start_reciving + len(tx)] = tx / 10

    res = np.zeros((len(tx) + len(rx)), dtype=np.complex64)
    c_lib.crosscorrelate(
        tx,
        len(tx),
        rx,
        len(rx),
        -len(tx),
        len(rx),
        res,
    )

    if plot:
        fig, ax = plt.subplots()
        delay = np.arange(-len(tx), len(rx))
        ax.plot(delay, res)
        ax.axvline(start_reciving, color="r")
        plt.show()

    assert (np.argmax(res) - len(tx)) == start_reciving


def test_crosscorrelate(plot):
    c_lib = load_c_lib()
    encode = np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1],
        dtype=np.complex64,
    )
    result = np.array(
        [
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            1 + 0 * 1j,
            2 + 0 * 1j,
            1 + 0 * 1j,
            0 + 0 * 1j,
            1 + 0 * 1j,
            2 + 0 * 1j,
            1 + 0 * 1j,
            0 + 0 * 1j,
            1 + 0 * 1j,
            2 + 0 * 1j,
            1 + 0 * 1j,
            0 + 0 * 1j,
            1 + 0 * 1j,
            2 + 0 * 1j,
            1 + 0 * 1j,
            0 + 0 * 1j,
            1 + 0 * 1j,
            2 + 0 * 1j,
            1 + 0 * 1j,
            0 + 0 * 1j,
            1 + 0 * 1j,
            2 + 0 * 1j,
            1 + 0 * 1j,
            0 + 0 * 1j,
            13 + 0 * 1j,
            26 + 0 * 1j,
            13 + 0 * 1j,
            0 + 0 * 1j,
            1 + 0 * 1j,
            2 + 0 * 1j,
            1 + 0 * 1j,
            0 + 0 * 1j,
            1 + 0 * 1j,
            2 + 0 * 1j,
            1 + 0 * 1j,
            0 + 0 * 1j,
            1 + 0 * 1j,
            2 + 0 * 1j,
            1 + 0 * 1j,
            0 + 0 * 1j,
            1 + 0 * 1j,
            2 + 0 * 1j,
            1 + 0 * 1j,
            0 + 0 * 1j,
            1 + 0 * 1j,
            2 + 0 * 1j,
            1 + 0 * 1j,
            0 + 0 * 1j,
            1 + 0 * 1j,
            2 + 0 * 1j,
            1 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
            0 + 0 * 1j,
        ]
    )

    sample_signal = np.zeros(85, dtype=np.complex64)
    sample_signal[23 : 23 + len(encode)] = encode
    rv = np.zeros((len(encode) + len(sample_signal)), dtype=np.complex64)
    c_lib.crosscorrelate(
        encode,
        len(encode),
        sample_signal,
        len(sample_signal),
        -len(encode),
        len(sample_signal),
        rv,
    )

    if plot:
        fig, ax = plt.subplots()
        delay = np.arange(-len(encode), len(sample_signal))
        ax.plot(delay, rv)
        ax.axvline(23, color="r")
        plt.show()

    assert np.array_equal(rv, result)
