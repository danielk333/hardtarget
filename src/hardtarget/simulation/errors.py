#!/usr/bin/env python

"""Errors

TODO: This feels like it should be in hardtarget? that already has all the models for signals
needed... also it would be nice the have errors as a standard output of processing
"""
import pathlib
import shutil
import numpy as np
import scipy.constants

from .simulate_drf import simulate_drf
from hardtarget.radars.eiscat import load_radar_code
from hardtarget import noise
from hardtarget.drf_utils import load_hardtarget_drf
from hardtarget.analysis import compute_gmf, load_gmf_out


def linearized_mle_covariance(
    snr_db,
    range0,
    vel0,
    acel0,
    n_ipp=10,
    dr=10.0,
    dv=1.0,
    da=1.0,
):
    """ """
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
        "code": load_radar_code("leo_bpark"),
    }
    snr = 10.0 ** (snr_db * 0.1)
    ipp = experiment_params["ipp"] * 1e-6
    simulation_params = {
        "epoch": "2021-04-12T12:15:40",
        "start_time": 0,
        "end_time": n_ipp * ipp,
        "target_start_time": 0,
        "target_end_time": n_ipp * ipp,
        "noise_sigma": 0,
        "tx_amp": 1,
    }

    def range_function(t):
        return range0 + vel0 * t + acel0 * 0.5 * t**2

    def range_function_dr(t):
        return range0 + dr + vel0 * t + acel0 * 0.5 * t**2

    def range_function_dv(t):
        return range0 + (vel0 + dv) * t + acel0 * 0.5 * t**2

    def range_function_da(t):
        return range0 + vel0 * t + (acel0 + da) * 0.5 * t**2

    sim_kw = dict(
        output_path=None,
        sim_params=simulation_params,
        experiment_params=experiment_params,
        snr_function=None,
        include_tx_signal=False,
        dtype=np.complex64,
    )
    z0 = simulate_drf(range_function=range_function, **sim_kw)
    z_dr = simulate_drf(range_function=range_function_dr, **sim_kw)
    z_dv = simulate_drf(range_function=range_function_dv, **sim_kw)
    z_da = simulate_drf(range_function=range_function_da, **sim_kw)

    z_diff_r = (z0 - z_dr) / dr
    z_diff_v = (z0 - z_dv) / dv
    z_diff_a = (z0 - z_da) / da

    A = np.zeros((len(z0), 3), dtype=np.complex64)
    A[:, 0] = z_diff_r
    A[:, 1] = z_diff_v
    A[:, 2] = z_diff_a
    tx_pulse_samps = np.round(
        experiment_params["tx_pulse_length"] * 1e-6 * experiment_params["sample_rate"]
    ).astype(np.int64)
    coh_samples = tx_pulse_samps * n_ipp
    z_sigma_inv = snr / (2 * coh_samples)
    S = np.linalg.inv(np.real(np.transpose(np.conj(A)) @ A * z_sigma_inv))
    return S


def monte_carlo_sample_errors(
    snr_db,
    range0,
    vel0,
    acel0,
    samples,
    output_path,
    clobber=True,
    gmf_method="fgmf",
    gmf_implementation="c",
    n_ipp=10,
):
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
        "code": load_radar_code("leo_bpark"),
    }
    rg0 = np.round((range0 / scipy.constants.c) * experiment_params["sample_rate"]).astype(np.int64)
    rg1 = rg0 + len(experiment_params["code"][0])
    rg_padding = 100

    if not isinstance(output_path, pathlib.Path):
        output_path = pathlib.Path(output_path)

    if output_path.is_file():
        raise FileExistsError("..")
    elif not output_path.is_dir():
        output_path.mkdir()

    config_str = f"""
    [signal-processing]
        n_ipp={n_ipp}
        ipp_offset=0
        min_range_gate={rg0 - rg_padding}
        max_range_gate={rg1 + rg_padding}
        min_acceleration=-300.0
        max_acceleration=300.0
        range_gate_step=1
        frequency_decimation=8
        num_cohints_per_file=10
        node_gpus=1
        dpt_ipp_delay_parameter={int(n_ipp // 2)}
    """
    config_path = output_path / "conf.cfg"
    drf_path = output_path / "drf_data"
    gmf_path = output_path / "gmf_data"
    with open(config_path, "w") as fh:
        fh.write(config_str)

    tx_pulse_samps = np.round(
        experiment_params["tx_pulse_length"] * 1e-6 * experiment_params["sample_rate"]
    ).astype(np.int64)
    coh_samples = tx_pulse_samps * n_ipp
    snr = 10.0 ** (snr_db * 0.1)
    # noise_sigma = np.sqrt(1 / (2 * snr * coh_samples))

    coh_int_time = n_ipp * experiment_params["ipp"] * 1e-6
    sim_len = coh_int_time * samples

    simulation_params = {
        "epoch": "2021-04-12T12:15:40",
        "start_time": 0,
        "end_time": sim_len,
        "target_start_time": 0,
        "target_end_time": sim_len,
        "noise_sigma": 1,
        "tx_amp": 1000,
    }
    rx_channel = "sim"

    def range_function(t):
        _t = t % coh_int_time
        return range0 + vel0 * _t + acel0 * 0.5 * _t**2

    try:
        simulate_drf(
            drf_path,
            range_function,
            simulation_params,
            experiment_params,
            snr_function=lambda t: np.full_like(t, snr / coh_samples),
            chnl=rx_channel,
            dtype=np.complex64,
            clobber=clobber,
        )
    except FileExistsError:
        pass

    reader, params = load_hardtarget_drf(drf_path)

    if clobber and gmf_path.is_dir():
        shutil.rmtree(gmf_path)

    compute_gmf(
        rx=(drf_path, rx_channel),
        tx=(drf_path, rx_channel),
        config=config_path,
        gmf_method=gmf_method,
        gmf_implementation=gmf_implementation,
        clobber=clobber,
        output=gmf_path,
        progress=True,
        subprogress=True,
    )

    errors = {
        "delta_r": np.full((samples,), np.nan, dtype=np.float64),
        "delta_v": np.full((samples,), np.nan, dtype=np.float64),
        "delta_a": np.full((samples,), np.nan, dtype=np.float64),
        "delta_snr": np.full((samples,), np.nan, dtype=np.float64),
    }
    data_generator = load_gmf_out(gmf_path)
    index = 0
    for data, meta in data_generator:
        data_len = len(data["range_peak"])
        print(f"loading {data_len} results")
        dr = data["range_peak"] - range0
        dv = data["range_rate_peak"] - vel0
        da = data["acceleration_peak"] - acel0
        dsnr = data["snr"] - snr
        errors["delta_r"][index : (index + data_len)] = dr
        errors["delta_v"][index : (index + data_len)] = dv
        errors["delta_a"][index : (index + data_len)] = da
        errors["delta_snr"][index : (index + data_len)] = dsnr
    errors["cov"] = np.cov(np.stack([data["range_peak"], data["range_rate_peak"], data["acceleration_peak"]]))
    return errors
