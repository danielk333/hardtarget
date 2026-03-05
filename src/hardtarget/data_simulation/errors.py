"""
Errors

Error estimations
"""

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from radardef.radar_stations.eiscat import load_radar_code
from radardef.types import BoundParams, ExpParams
from scipy import constants

from hardtarget import analyse, load_analysed_data
from hardtarget.types.constants import AnalysisMethod, Impl, TargetEstimationMethod

from .simulate_drf import DRFSimParams, simulate_drf


def linearized_mle_covariance(
    snr_db: float,
    range0: float,
    vel0: float,
    acel0: float,
    n_ipp: int = 10,
    dr: float = 10.0,
    dv: float = 1.0,
    da: float = 1.0,
) -> npt.NDArray[np.floating]:
    """ """
    experiment_params = ExpParams(
        name="simulation",
        radar_frequency=929.6,
        t_ipp_usec=20000,
        ipp_samps=20000,
        sample_rate=1000000,
        t_samp_usec=1,
        rx_channels=["sim"],
        t_tx_start_usec=82.0,
        t_tx_end_usec=2002.0,
        t_rx_start_usec=0,
        t_rx_end_usec=20000,
        tx_channel="sim",
        tx_pulse_length=1920,
        t_cal_on_usec=19900.0,
        t_cal_off_usec=19997.0,
        wavelength=constants.c / (929.6 * 1e6),
        code=load_radar_code("leo_bpark"),
    )
    snr = 10.0 ** (snr_db * 0.1)
    simulation_params = DRFSimParams(
        epoch="2021-04-12T12:15:40",
        start_time_us=0,
        end_time_us=int(n_ipp * experiment_params.t_ipp_usec),
        target_start_time_us=0,
        target_end_time_us=int(n_ipp * experiment_params.t_ipp_usec),
        noise_sigma=0,
        tx_amp=1,
    )

    def range_function(t: float) -> float:
        return range0 + vel0 * t + acel0 * 0.5 * t**2

    def range_function_dr(t: float) -> float:
        return range0 + dr + vel0 * t + acel0 * 0.5 * t**2

    def range_function_dv(t: float) -> float:
        return range0 + (vel0 + dv) * t + acel0 * 0.5 * t**2

    def range_function_da(t: float) -> float:
        return range0 + vel0 * t + (acel0 + da) * 0.5 * t**2

    z0 = simulate_drf(
        output_path=None,
        range_function=range_function,
        sim_params=simulation_params,
        bounds_params=BoundParams(),
        experiment_params=experiment_params,
        snr_function=None,
        include_tx_signal=False,
        dtype=np.complex64,
    )
    z_dr = simulate_drf(
        output_path=None,
        range_function=range_function_dr,
        sim_params=simulation_params,
        bounds_params=BoundParams(),
        experiment_params=experiment_params,
        snr_function=None,
        include_tx_signal=False,
        dtype=np.complex64,
    )
    z_dv = simulate_drf(
        output_path=None,
        range_function=range_function_dv,
        sim_params=simulation_params,
        bounds_params=BoundParams(),
        experiment_params=experiment_params,
        snr_function=None,
        include_tx_signal=False,
        dtype=np.complex64,
    )
    z_da = simulate_drf(
        output_path=None,
        range_function=range_function_da,
        sim_params=simulation_params,
        bounds_params=BoundParams(),
        experiment_params=experiment_params,
        snr_function=None,
        include_tx_signal=False,
        dtype=np.complex64,
    )

    z_diff_r = (z0 - z_dr) / dr
    z_diff_v = (z0 - z_dv) / dv
    z_diff_a = (z0 - z_da) / da

    A = np.zeros((len(z0), 3), dtype=np.complex64)
    A[:, 0] = z_diff_r
    A[:, 1] = z_diff_v
    A[:, 2] = z_diff_a

    assert experiment_params.tx_pulse_length is not None

    tx_pulse_samps = np.round(
        experiment_params.tx_pulse_length * 1e-6 * experiment_params.sample_rate
    ).astype(np.int64)
    coh_samples = tx_pulse_samps * n_ipp
    z_sigma_inv = snr / (2 * coh_samples)
    S = np.linalg.inv(np.real(np.transpose(np.conj(A)) @ A * z_sigma_inv))
    return S


def monte_carlo_sample_errors(
    snr_db: float,
    range0: float,
    vel0: float,
    acel0: float,
    samples: int,
    output_path: str | Path,
    clobber: bool = True,
    mf_method: TargetEstimationMethod = TargetEstimationMethod.fgmf,
    mf_implementation: Impl = Impl.c,
    n_ipp: int = 5,
    sample_rate: float = 1_000_000,
    t_ipp_s: float = 2e-2,
    tx_pulse_length: float = 2e-3,
    radar_frequency: float = 930e6,
) -> dict[str, npt.NDArray]:

    tx_start = 0
    tx_end = tx_pulse_length * 1e6 - 1
    rx_start = 0
    rx_end = t_ipp_s * 1e6
    tx_pulse_length = int(tx_pulse_length * 1e6) - 1  # should be (end-start) - t_samp_usec
    t_samp_usec = (1 / sample_rate) * 1e6
    experiment_params = ExpParams(
        name="simulation",
        radar_frequency=radar_frequency * 1e-6,
        t_ipp_usec=int(t_ipp_s * 1e6),
        ipp_samps=20000,  # needs to corrected
        sample_rate=sample_rate,
        t_samp_usec=t_samp_usec,
        rx_channels=["sim"],
        t_tx_start_usec=tx_start,
        t_tx_end_usec=tx_end,
        t_rx_start_usec=rx_start,
        t_rx_end_usec=rx_end,
        tx_channel="sim",
        tx_pulse_length=tx_pulse_length,
        t_cal_on_usec=0,
        t_cal_off_usec=0,
        wavelength=constants.c / radar_frequency,
        code=load_radar_code("leo_bpark"),
    )

    bounds_params = BoundParams()

    rg0 = np.round((range0 / constants.c) * experiment_params.sample_rate).astype(np.int64)
    rg1 = rg0 + len(experiment_params.code[0])
    rg_padding = 100

    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    if output_path.is_file():
        raise FileExistsError("..")
    elif not output_path.is_dir():
        output_path.mkdir()

    config_str = f"""
    [processing]
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
    [dpt]
        ipp_delay_parameter={int(n_ipp // 2)}
    [gmf]
        acceleration_steps = 20
    """

    config_path = output_path / "conf.cfg"
    drf_path = output_path / "data_drf"
    gmf_path = output_path / "gmf_data"

    with open(config_path, "w") as fh:
        fh.write(config_str)

    tx_pulse_samps = tx_pulse_length / t_samp_usec
    coh_samples = tx_pulse_samps * n_ipp
    snr = 10.0 ** (snr_db * 0.1)
    if not isinstance(snr, np.ndarray):
        snr = np.array([snr])
    snr_len = len(snr)

    coh_int_time = n_ipp * experiment_params.t_ipp_usec
    step_size = coh_int_time * samples
    sim_len = step_size * snr_len

    simulation_params = DRFSimParams(
        epoch="2021-04-12T12:15:40",
        start_time_us=0,
        end_time_us=int(sim_len),
        target_start_time_us=0,
        target_end_time_us=int(sim_len),
        noise_sigma=1,
        tx_amp=10000,
    )

    def range_function(t: float) -> float:
        _t = t % coh_int_time
        return range0 + vel0 * _t + acel0 * 0.5 * _t**2

    try:
        simulate_drf(
            output_path=drf_path,
            range_function=range_function,
            sim_params=simulation_params,
            experiment_params=experiment_params,
            bounds_params=bounds_params,
            snr_function=lambda t: snr[np.floor(t / step_size).astype(np.int64)] / coh_samples,
            dtype=np.complex64,
            clobber=clobber,
        )
    except FileExistsError:
        pass

    if clobber and gmf_path.is_dir():
        shutil.rmtree(gmf_path)

    analyse(
        path=drf_path,
        config=config_path,
        method=AnalysisMethod.target_estimation,
        method_lib=mf_method,
        implementation=mf_implementation,
        clobber=clobber,
        output=gmf_path,
        progress=True,
        # noise_power=2 * simulation_params["noise_sigma"] ** 2, # TODO ADD TO ANALYSE
    )

    results = {
        "delta_r": np.full((samples * snr_len,), np.nan, dtype=np.float64),
        "delta_v": np.full((samples * snr_len,), np.nan, dtype=np.float64),
        "delta_a": np.full((samples * snr_len,), np.nan, dtype=np.float64),
        "delta_snr": np.full((samples * snr_len,), np.nan, dtype=np.float64),
    }
    data_generator: Any = load_analysed_data(gmf_path)
    index = 0
    for out_data, exp_params, cfg_params, pro_params in data_generator:
        data_len = len(out_data.r_vec)
        print(f"loading {data_len} results")
        dr = out_data.r_vec - range0
        dv = out_data.v_vec - vel0
        da = out_data.a_vec - acel0
        results["range"] = out_data.r_vec
        results["range_rate"] = out_data.v_vec
        results["acceleration"] = out_data.a_vec
        results["snr"] = np.max(out_data.snr, axis=1)
        dsnr = np.max(out_data.snr, axis=1)
        for ind in range(snr_len):
            dsnr[(ind * samples) : ((ind + 1) * samples)] = -snr[ind]
        results["delta_r"][index : (index + data_len)] = dr
        results["delta_v"][index : (index + data_len)] = dv
        results["delta_a"][index : (index + data_len)] = da
        results["delta_snr"][index : (index + data_len)] = dsnr
        index += data_len
    results["cov"] = np.cov(np.stack([out_data.r_vec, out_data.v_vec, out_data.a_vec]))
    return results
