import tempfile
from pathlib import Path

import numpy as np
import pytest
import radardef.radar_stations.eiscat.utils as radardef_utils
from radardef.types import BoundParams, ExpParams
from scipy import constants

import hardtarget
from hardtarget.constants import AnalysisMethod, Impl, TargetEstimationMethod
from hardtarget.data_simulation import DRFSimParams, simulate_drf


class TestBlackBoxComputeGMF:
    """
    It is possible that CUDA support is compiled, yet still non-functional.
    """

    @pytest.mark.parametrize("mf_impl", [Impl.numpy, Impl.c])
    def test_dpt(self, mf_impl):
        self.run_test(TargetEstimationMethod.fdpt, mf_impl)

    @pytest.mark.parametrize("mf_impl", [Impl.numpy, Impl.c])
    def test_gmf(self, mf_impl):
        self.run_test(TargetEstimationMethod.fgmf, mf_impl)

    @pytest.mark.cuda
    def test_gmf_cuda(self):
        self.run_test(TargetEstimationMethod.fgmf, Impl.cuda)

    def run_test(self, mf_method, mf_impl):
        """Simulate echoes without noise and analyse the echoes to verify parameters are recovered.

        There is a minimum acceleration thats theoretically detectable depending
        on the total time integrated and the numerical derivative length and the
        doppler of the target

        so min_acc(t_total, tau, dop)

        anything below this acceleration will be aliased into higher acceleration
        bins, even if there is no noise
        """

        mf_impl = mf_impl.name

        debug_test = False
        frequency_decimation = 10
        n_ipp = 10
        tau_ipp = 5

        config_str = f"""
        [processing]
            n_ipp={n_ipp}
            ipp_offset=0
            num_cohints_per_file=10
            min_range_gate=6640
            max_range_gate=6750
            range_gate_step=1
            node_gpus=1
        [target_estimation]
            min_acceleration=-300.0
            max_acceleration=300.0
            frequency_decimation={frequency_decimation}
        [dpt]
            ipp_delay_parameter={tau_ipp}
        [gmf]
            acceleration_steps = 25
        """

        t_start = 0
        t_end = 20000 * 750  # ipp_us = 20000
        coh_int_len = 20000 * n_ipp
        t_abs_us = np.arange(0, t_end + coh_int_len, coh_int_len)
        t_abs = t_abs_us * 1e-6

        simulation_params = DRFSimParams(
            epoch="2021-04-12T12:15:40",
            start_time_us=t_start,
            end_time_us=t_end,
            target_start_time_us=t_start,
            target_end_time_us=t_end,
            noise_sigma=0,
            tx_amp=1,
        )

        range0 = 2000e3
        vel0 = 0.4e3
        acel0 = 0.10e3

        sim_r = range0 + vel0 * t_abs + acel0 * 0.5 * t_abs**2
        sim_v = vel0 + acel0 * t_abs
        sim_a = np.ones_like(t_abs) * acel0

        def range_function(t):
            inds = np.logical_and(t >= t_abs[0], t <= t_abs[-1])
            if np.any(inds):
                return range0 + vel0 * t[inds] + acel0 * 0.5 * t[inds] ** 2
            else:
                return np.nan

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

        sample_rate = 1 / (exp_params.t_samp_usec * 1e-6)
        dec_samp = (exp_params.t_ipp_usec * 1e-6 * sample_rate) / frequency_decimation

        range_gate = constants.c / sample_rate
        doppler_gate = (
            2 * exp_params.wavelength * frequency_decimation / ((exp_params.t_ipp_usec * 1e-6) * n_ipp)
        )

        step = 2 * dec_samp * tau_ipp * frequency_decimation / sample_rate
        max_accels_len = (n_ipp - tau_ipp) * dec_samp
        accel_gate = exp_params.wavelength * 2 * sample_rate / (max_accels_len * frequency_decimation * step)
        print(f"{range_gate=} meters/sample, {doppler_gate=}, {accel_gate=}")

        with (
            tempfile.TemporaryDirectory(suffix="_drf") as tmp_sim_path,
            tempfile.TemporaryDirectory() as tmp_analysis_path,
            tempfile.NamedTemporaryFile(mode="w+") as tmp_config,
        ):
            # hacky way to create a temp config
            tmp_config.write(config_str)
            tmp_config.seek(0)
            tmp_config_path = tmp_config.name

            print(f"{tmp_config_path=}")

            simulate_drf(
                Path(tmp_sim_path),
                range_function,
                simulation_params,
                exp_params,
                bounds_params,
                snr_function=None,
                dtype=np.complex64,
                clobber=True,
            )

            # process
            _ = hardtarget.target_estimation(
                path=Path(tmp_sim_path).resolve(),
                rx_channel="sim",
                config=tmp_config_path,
                start_time=simulation_params.start_time_us,
                end_time=simulation_params.end_time_us,
                relative_time=True,
                method_lib=mf_method,
                implementation=mf_impl,
                clobber=False,
                output=tmp_analysis_path,
                progress=False,
            )

            data_generator = hardtarget.load_analysed_data(tmp_analysis_path)
            for out_args, exp_params, cfg_params, pro_params in data_generator:
                dr = out_args.r_vec - sim_r[:-1]
                dv = out_args.v_vec - sim_v[:-1]
                da = out_args.a_vec - sim_a[:-1]

                def assert_simulated_vs_estimated(delta, limit, param_str):
                    mean_error = np.abs(np.mean(delta))
                    print(f"{param_str} = {mean_error} (std = {np.std(delta)}) < {limit}")
                    assert mean_error < limit, (
                        f"mean {param_str} is over the limit, x̄({param_str}): {mean_error}, limit: {limit}"
                    )
                    std = np.std(delta)
                    assert std < limit, (
                        f"{param_str} standard deviation is to large, std({param_str}): {std} > {limit} "
                    )

                # Assert expected range is equal to estimated range
                assert_simulated_vs_estimated(dr, range_gate, "delta_r")

                # Assert expected velocity is equal to estimated velocity
                assert_simulated_vs_estimated(dv, doppler_gate, "delta_v")

                # Assert expected acceleration is equal to estimated acceleration
                assert_simulated_vs_estimated(da, accel_gate, "delta_a")

            # # This is test debugging code
            if debug_test:
                import matplotlib.pyplot as plt

                data_generator = hardtarget.load_analysed_data(tmp_analysis_path)
                for _out_args, _exp_params, _cfg_params, _pro_params in data_generator:
                    t = _out_args.t - np.min(_out_args.t)

                    fig, axes = plt.subplots(2, 2)
                    hardtarget.plotting.target_estimation_plots.plot_peaks(
                        axes, _out_args, _exp_params, _cfg_params, _pro_params
                    )
                    fig, axes = plt.subplots(2, 3)
                    hardtarget.plotting.target_estimation_plots.plot_detections(
                        axes, _out_args, _exp_params, _cfg_params, _pro_params
                    )
                    fig, axes = plt.subplots(3, 1)
                    hardtarget.plotting.target_estimation_plots.plot_map(
                        axes, _out_args, _exp_params, _cfg_params, _pro_params
                    )

                    fig, axes = plt.subplots(2, 2)
                    nf_vec = np.nanmedian(_out_args.dc, axis=0)
                    nf_vec = nf_vec.reshape((1, nf_vec.size))
                    nf_range = np.nanmedian(nf_vec, axis=0)
                    snr = hardtarget.noise.snr(_out_args.vals, nf_range)
                    r_inds = np.argmax(_out_args.vals, axis=1)
                    coh_inds = np.arange(_out_args.vals.shape[0])
                    snr = snr[coh_inds, r_inds]

                    axes[0, 0].plot(t, _out_args.r_vec * 1e-3 * 0.5, c="blue", label="r_vec")
                    axes[0, 0].plot(t_abs, sim_r * 1e-3 * 0.5, c="red", label="sim_r")
                    axes[0, 0].set_xlabel("Time [s]")
                    axes[0, 0].set_ylabel("range [km]")
                    axes[0, 0].legend(loc="upper left")

                    axes[0, 1].plot(t, _out_args.v_vec * 1e-3 * 0.5, c="blue", label="v_vec")
                    axes[0, 1].plot(t_abs, sim_v * 1e-3 * 0.5, c="red", label="sim_v")
                    axes[0, 1].set_xlabel("Time [s]")
                    axes[0, 1].set_ylabel("range rate [km/s]")
                    axes[0, 1].legend(loc="upper left")

                    axes[1, 0].plot(t, _out_args.a_vec * 0.5, c="blue", label="a_vec")
                    axes[1, 0].plot(t_abs, sim_a * 0.5, c="red", label="sim_a")
                    axes[1, 0].set_xlabel("Time [s]")
                    axes[1, 0].set_ylabel("acceleration [m/s^2]")
                    axes[1, 0].set_ylim([-300, 300])
                    axes[1, 0].legend(loc="lower left")

                    axes[1, 1].plot(t, np.sqrt(snr))
                    axes[1, 1].set_xlabel("Time [s]")
                    axes[1, 1].set_ylabel("sqrt(ENR) [1]")

                plt.show()
