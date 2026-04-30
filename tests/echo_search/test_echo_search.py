import datetime as dt
import tempfile
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from radardef.radar_stations import Mu
from radardef.types import ExpDef

import hardtarget
from hardtarget.data_simulation import simulate_h5
from hardtarget.echo_search import EchoSearchOutArgs, EchoSearchProParams
from hardtarget.echo_search.types import EchoSearchCfgParams


def test_echo():
    exp_params = ExpDef(
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

    def trajectory_func(t: npt.NDArray) -> npt.NDArray:
        # Initial values
        r0: float = 220e3
        v0: float = -0.4e3
        a0: float = -0.20e3
        # Initial position over radar
        k0 = np.array([0, 0, 1])
        # 3d position
        x_pos = k0 * r0
        # Velocity vector
        v_vec = np.array([1, 0, 0])
        distance_traveled = v0 * t + a0 * 0.5 * t**2
        # trajectory
        return x_pos[:, None] + v_vec[:, None] * distance_traveled[None, :]

    # ## Simulate the experiment with the MU radars beam.
    # ---

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "sim"
        station = Mu()
        measurement_length_us = exp_params.t_ipp_usec * 10
        target_start_time_us = exp_params.t_ipp_usec * 4
        target_end_time_us = exp_params.t_ipp_usec * 6
        simulate_h5(
            output_dir=output_path,
            exp_params=exp_params,
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
            doppler_freq_min=-30000,
            doppler_freq_max=5000,
            doppler_freq_step=1000,
        )

        output_analysis = Path(temp_dir) / "echo"
        _ = hardtarget.echo_search(
            data=Path(output_path).resolve(),
            config=cfg,
            relative_time=True,
            clobber=False,
            output=output_analysis,
            progress=False,
            exp_params=exp_params,
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

        assert out.max_corr[target_filter] == pytest.approx(1.0)
        assert out.max_corr[no_target_filter] == pytest.approx(0.0)
