import pytest
import numpy as np
import tempfile
from scipy import constants

from hardtarget.gmf import Impl
import hardtarget


class TestBlackBoxComputeGMF:
    """
    It is possible that CUDA support is compiled, yet still non-functional.
    """

    @pytest.mark.parametrize("gmf_impl", [Impl.numpy, Impl.c])
    def test_dpt(self, gmf_impl):
        self.run_test("fdpt", gmf_impl)

    @pytest.mark.parametrize("gmf_impl", [Impl.numpy, Impl.c])
    def test_gmf(self, gmf_impl):
        self.run_test("fgmf", gmf_impl)

    @pytest.mark.cuda
    def test_gmf_cuda(self):
        self.run_test("fgmf", Impl.cuda)

    def run_test(self, gmf_method, gmf_impl):
        """Simulate echoes without noise and analyse the echoes to verify parameters are recovered.

        There is a minimum acceleration thats theoretically detectable depending
        on the total time integrated and the numerical derivative length and the
        doppler of the target

        so min_acc(t_total, tau, dop)

        anything below this acceleration will be aliased into higher acceleration
        bins, even if there is no noise
        """

        gmf_impl = gmf_impl.name

        debug_test = False
        frequency_decimation = 10
        n_ipp = 10
        tau_ipp = 5

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

        t_end = 4.0
        coh_int_len = 0.2
        t_abs = np.arange(0, t_end + coh_int_len, coh_int_len)

        simulation_params = {
            "epoch": "2021-04-12T12:15:40",
            "start_time": 0,
            "end_time": t_end,
            "target_start_time": 0,
            "target_end_time": t_end,
            "noise_sigma": 0,
            "tx_amp": 1,
        }

        rx_channel = "sim"

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

        wavelength = constants.c / (experiment_params["frequency"] * 1e6)
        print(wavelength)

        sample_rate = experiment_params["sample_rate"]
        dec_samp = (
            (experiment_params["ipp"] * 1e-6 * sample_rate)
            / frequency_decimation
        )

        range_gate = constants.c / sample_rate
        doppler_gate = 2 * wavelength * frequency_decimation / ((experiment_params["ipp"] * 1e-6) * n_ipp)

        step = 2 * dec_samp * tau_ipp * frequency_decimation / sample_rate
        max_accels_len = (n_ipp - tau_ipp) * dec_samp
        accel_gate = wavelength * 2 * sample_rate / (max_accels_len * frequency_decimation * step)
        print(f"{range_gate=}, {doppler_gate=}, {accel_gate=}")

        for key, val in experiment_params.items():
            print(f"{key}: {val}")

        with (
            tempfile.TemporaryDirectory() as tmp_sim_path,
            tempfile.TemporaryDirectory() as tmp_analysis_path,
            tempfile.NamedTemporaryFile(mode="w+") as tmp_config,
        ):
            # hacky way to create a temp config
            tmp_config.write(config_str)
            tmp_config.seek(0)
            tmp_config_path = tmp_config.name

            print(f"{tmp_config_path=}")

            hardtarget.simulation.drf(
                tmp_sim_path,
                range_function,
                simulation_params,
                experiment_params,
                chnl=rx_channel,
                snr_function=None,
                dtype=np.complex64,
                clobber=True,
            )

            reader, params = hardtarget.drf_utils.load_hardtarget_drf(tmp_sim_path)

            all_params = hardtarget.load_gmf_params(tmp_sim_path, tmp_config_path)

            for key, val in all_params["PRO"].items():
                print(f"{key}: {val}")

            # process
            _ = hardtarget.compute_gmf(
                rx=(tmp_sim_path, rx_channel),
                tx=(tmp_sim_path, rx_channel),
                config=tmp_config_path,
                gmf_method=gmf_method,
                gmf_implementation=gmf_impl,
                clobber=False,
                output=tmp_analysis_path,
                progress=False,
                subprogress=False,
            )

            data_generator = hardtarget.load_gmf_out(tmp_analysis_path)
            for data, meta in data_generator:
                dr = data["range_peak"] - sim_r[1:-1]
                dv = data["range_rate_peak"] - sim_v[1:-1]
                da = data["acceleration_peak"] - sim_a[1:-1]

                print(f"dr = {np.abs(np.mean(dr))} (std = {np.std(dr)}) < {range_gate=}")
                assert np.abs(np.mean(dr)) < range_gate
                assert np.std(dr) < range_gate

                print(f"dr = {np.abs(np.mean(dv))} (std = {np.std(dv)}) < {doppler_gate=}")
                assert np.abs(np.mean(dv)) < doppler_gate
                assert np.std(dv) < doppler_gate

                print(f"dr = {np.abs(np.mean(da))} (std = {np.std(da)}) < {accel_gate=}")
                assert np.abs(np.mean(da)) < accel_gate
                assert np.std(da) < accel_gate

            # # This is test debugging code
            if debug_test:
                import matplotlib.pyplot as plt
                data_generator = hardtarget.load_gmf_out(tmp_analysis_path)
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

                    axes[0, 0].plot(data["t"], data["range_peak"]*1e-3*0.5, c="blue")
                    axes[0, 0].plot(t_abs, sim_r*1e-3*0.5, c="red")
                    axes[0, 0].set_xlabel("Time [s]")
                    axes[0, 0].set_ylabel("range [km]")

                    axes[0, 1].plot(data["t"], data["range_rate_peak"]*1e-3*0.5, c="blue")
                    axes[0, 1].plot(t_abs, sim_v*1e-3*0.5, c="red")
                    axes[0, 1].set_xlabel("Time [s]")
                    axes[0, 1].set_ylabel("range rate [km/s]")

                    axes[1, 0].plot(data["t"], data["acceleration_peak"]*0.5, c="blue")
                    axes[1, 0].plot(t_abs, sim_a*0.5, c="red")
                    axes[1, 0].set_xlabel("Time [s]")
                    axes[1, 0].set_ylabel("acceleration [m/s^2]")
                    axes[1, 0].set_ylim([-300, 300])

                    axes[1, 1].plot(data["t"], np.sqrt(snr))
                    axes[1, 1].set_xlabel("Time [s]")
                    axes[1, 1].set_ylabel("sqrt(ENR) [1]")
                plt.show()
