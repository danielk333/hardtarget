import os
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
from radardef import RadarDef
from radardef.radar_stations.eiscat.experiments import load_radar_code
from radardef.types import BoundParams, ExpDef

from hardtarget.data_handling import load_config_params
from hardtarget.data_simulation import DRFSimParams, simulate_drf

try:
    config = Path(__file__).parent.parent.absolute() / "cfg" / "sim_test.ini"
except NameError:
    config = Path(os.path.abspath("")) / "docs" / "examples" / "cfg" / "sim_test.ini"


def sim_data(
    output_path: Path,
    exp_params: Optional[ExpDef] = None,
    bounds_params: Optional[BoundParams] = None,
    config: Optional[Path] = None,
    range0: float = 2000e3,
    vel0: float = -0.4e3,
    acel0: float = 0.20e3,
    zero_noise=False,
):

    SNRdB = 40.0
    samples = 200

    if exp_params is None:
        exp_params = ExpDef(
            name="test",
            radar_frequency=929.6,
            t_ipp_usec=20000,
            t_samp_usec=1,
            rx_channels=["sim"],
            t_tx_start_usec=82.0,
            t_tx_end_usec=2002.0,
            t_rx_start_usec=0,
            t_rx_end_usec=20000,
            tx_channel="sim",
            t_cal_on_usec=19900.0,
            t_cal_off_usec=19997.0,
            code=load_radar_code("leo_bpark"),
            baud_length_usec=30,
            samples_per_file=12800000,
        )

    if bounds_params is None:
        bounds_params = BoundParams(
            ts_start_usec=1445511612.8,
            ts_end_usec=1445551228.8,
        )

    if config is None:
        try:
            config = Path(__file__).parent.parent.absolute() / "cfg" / "sim_test.ini"
        except NameError:
            config = Path(os.path.abspath("")) / "docs" / "examples" / "cfg" / "sim_test.ini"

    # Load user configuration and calculate start and end time
    cfg_params = load_config_params(config)
    coh_samples = (
        (exp_params.t_tx_end_usec - exp_params.t_tx_start_usec) / exp_params.t_samp_usec
    ) * cfg_params.n_ipp
    SNR = 10 ** (SNRdB / 10.0)
    if zero_noise:
        noise_sigma = 0
    else:
        noise_sigma = np.sqrt(coh_samples / (2 * SNR))
    t_start = 0
    coh_int_time = cfg_params.n_ipp * exp_params.t_ipp_usec
    echo_len = coh_int_time * samples
    t_abs = np.zeros(samples)
    t_rel = t_abs - t_start

    # Define a function to get the time of the current coherent integration

    def time_modder(t):
        return t % (coh_int_time * 1e-6)

    # range function needed to shape the signal

    range0 = 2000e3
    vel0 = -0.4e3
    acel0 = 0.20e3

    def range_function(t):
        _t = time_modder(t - t_start)
        return range0 + vel0 * _t + acel0 * 0.5 * _t**2

    # Simulation parameters to define the start, end, noise and transmitted power.

    simulation_params = DRFSimParams(
        epoch="2021-04-12T12:15:40",
        start_time_us=t_start,
        end_time_us=int(echo_len),
        target_start_time_us=t_start,
        target_end_time_us=int(echo_len),
        noise_sigma=noise_sigma,
        tx_amp=1,
    )

    # Create the simulated signal

    simulate_drf(
        output_path=output_path,
        range_function=range_function,
        sim_params=simulation_params,
        experiment_params=exp_params,
        bounds_params=bounds_params,
        dtype=np.complex64,
        clobber=True,
    )

    return range0, vel0, acel0, t_rel, t_abs, SNR, echo_len, exp_params


def download_test_data(path: Path) -> Path:

    download_location = path / "MUI.000000.000000"
    if not download_location.is_file():
        url = "https://cloud.irf.se/public.php/dav/files/pXR6iYARobLxd2f/?accept=zip"
        urllib.request.urlretrieve(url, download_location)
    return download_location


def convert_test_data(path: Path, dst: Path) -> list[Path]:
    radars = RadarDef()
    source_format = radars.get_source_format(path)
    target_formats = radars.available_target_formats(source_format)
    converted_files = RadarDef().convert(path, target_formats[0], str(dst))
    assert converted_files is not None, "No available files after conversion"
    return converted_files
