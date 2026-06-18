"""
Digital RF simulator, used to simulate raw data in the digital rf format.
"""

import configparser
import logging
import pathlib
import re
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

import digital_rf as drf
import numpy as np
import numpy.typing as npt
import scipy.constants
from radardef.types import BoundParams, ExpDef, Expparam, Metaparam

from hardtarget.data_simulation.utils import noise_generator, waveform_generator

from .utils import DRFSimParams

logger = logging.getLogger(__name__)


def simulate_drf(
    output_path: pathlib.Path | None,
    range_function: Callable,
    sim_params: DRFSimParams,
    exp_def: ExpDef,
    bounds_params: BoundParams,
    snr_function: Optional[Callable] = None,
    compression_level: int = 0,
    dir_cadence_secs: int = 3600,
    file_cadence_millisecs: int = 1000,
    clobber: bool = False,
    include_tx_signal: bool = True,
    dtype: Any = np.complex128,
) -> npt.NDArray:
    """
    Simulates a digital rf signal.

    Args:
        output_path: Path to directory to store the simulated measurement.
        range_function: Range function, used to model the range input: time point [seconds] output: range [km]
        sim_params: Simulation parameters, specifics to control the simulation.
        exp_def: Experiment parameters used to model the signal.
        bounds_params: Epoch bounds, start and end of measurement.
        snr_function (optional): Signal to noise ratio function, used to simulate the snr.
        compression_level (optional): Compression level on the raw data.
        dir_cadence_secs (optional): int = 3600,
        file_cadence_millisecs (optional): int = 1000,
        clobber (optional): If old files should be overwritten, set to true.
        dtype (optional): Data type

    """

    # ------- Channel data --------

    sample_rate = exp_def.sample_rate

    rx_start_usec = exp_def.t_rx_start_usec
    rx_end_usec = exp_def.t_rx_end_usec
    tx_start_usec = exp_def.t_tx_start_usec
    tx_end_usec = exp_def.t_tx_end_usec

    ipp_samp = exp_def.ipp_samps
    wavelength = exp_def.wavelength
    rx_start_samp = np.round(rx_start_usec / exp_def.t_samp_usec).astype(np.int64)
    rx_end_samp = np.round(rx_end_usec / exp_def.t_samp_usec).astype(np.int64)
    n_rx_samps = rx_end_samp - rx_start_samp
    rx_select = np.full((ipp_samp,), False, dtype=bool)
    rx_select[rx_start_samp:rx_end_samp] = True

    tx_start_samp = np.round(tx_start_usec / exp_def.t_samp_usec).astype(np.int64)
    tx_end_samp = np.round(tx_end_usec / exp_def.t_samp_usec).astype(np.int64)
    n_tx_samps = tx_end_samp - tx_start_samp
    tx_select = np.full((ipp_samp,), False, dtype=bool)
    tx_select[tx_start_samp:tx_end_samp] = True

    code = exp_def.code
    codes = exp_def.code.shape[0]

    samp_t0 = sim_params.start_time_us * 1e-6 * sample_rate
    samp_t0 = np.round((samp_t0 // ipp_samp) * ipp_samp).astype(np.int64)
    samp_t1 = sim_params.end_time_us * 1e-6 * sample_rate
    samp_t1 = np.round((samp_t1 // ipp_samp) * ipp_samp).astype(np.int64)

    sig_t0 = sim_params.target_start_time_us
    samp_sig_t0 = sig_t0 * 1e-6 * sample_rate
    sig_t1 = sim_params.target_end_time_us
    samp_sig_t1 = sig_t1 * 1e-6 * sample_rate

    sim_pulses = int((samp_t1 - samp_t0) / ipp_samp)
    simulated_signal = np.empty((samp_t1 - samp_t0,), dtype=dtype)

    rf_writer = None
    dstdir = None
    if output_path is not None:
        output_path = pathlib.Path(output_path)
        output_path.mkdir(exist_ok=True)

        dstdir = output_path / str(exp_def.rx_channels[0])
        if dstdir.is_dir() and clobber:
            logger.info(f"'{dstdir}' exists and clobber is on: removing dir")
            shutil.rmtree(dstdir)
        elif dstdir.is_dir():
            raise FileExistsError(f"Directory '{dstdir}' exists")
        dstdir.mkdir(exist_ok=False)

        rf_writer = drf.DigitalRFWriter(
            str(dstdir),  # directory
            dtype,  # dtype
            dir_cadence_secs,  # subdir cadence secs    => one dir per hour
            file_cadence_millisecs,  # file cadence millisecs => one file per second
            samp_t0,  # start global index
            sample_rate,  # sample rate numerator
            1,  # sample rate denominator
            uuid_str="tbd",
            compression_level=compression_level,
            checksum=False,
            num_subchannels=1,
            is_continuous=True,
            marching_periods=False,
        )

    t_tx = np.arange(n_tx_samps) / sample_rate
    for pid in range(sim_pulses):
        samp0 = pid * ipp_samp + samp_t0
        signal = np.zeros((ipp_samp,), dtype=dtype)

        if sim_params.noise_sigma > 0:
            signal[rx_select] += noise_generator(
                sim_params.noise_sigma,
                (n_rx_samps,),
                dtype=dtype,
            )

        tx_wave = waveform_generator(
            n_tx_samps=n_tx_samps,
            sample_rate=exp_def.sample_rate,
            baud_length_sec=exp_def.baud_length_usec,
            code=code[pid % codes],
            dtype=dtype,
        )
        tx_amp0 = sim_params.tx_amp

        if include_tx_signal:
            # This assumes rx streches over tx, generalize
            signal[tx_select] += tx_amp0 * tx_wave

        s0 = samp0 + tx_start_samp
        s1 = s0 + n_tx_samps / sample_rate
        t0 = s0 / sample_rate
        if s0 >= samp_sig_t0 and s1 <= samp_sig_t1:
            r0 = range_function(np.array([t0], dtype=np.float64))[0]
            sn0 = snr_function(t0 + t_tx) if snr_function is not None else 1.0
            rg0 = np.round((r0 / scipy.constants.c) * sample_rate).astype(np.int64)
            rg_samp0 = rg0 + tx_start_samp

            if rg_samp0 >= rx_start_samp and rg_samp0 <= rx_end_samp:
                if sim_params.noise_sigma > 0:
                    amp0 = np.sqrt(sn0 * 2 * sim_params.noise_sigma**2)
                else:
                    amp0 = np.sqrt(sn0)
                ranges = range_function(t0 + t_tx)
                phase = np.mod(ranges / wavelength, 1) * np.pi * 2

                rx_wave = tx_wave * amp0 * np.exp(1j * phase)
                signal[rg_samp0 : (rg_samp0 + n_tx_samps)] += rx_wave

                simulated_signal[(pid * ipp_samp) : ((pid + 1) * ipp_samp)] = signal

        if rf_writer:
            rf_writer.rf_write(signal)

    if output_path is not None:
        # ------------ pointing --------------

        pntdir = output_path / "pointing"

        if pntdir.is_dir() and clobber:
            logger.info(f"'{pntdir}' exists and clobber is on: removing dir")
            shutil.rmtree(pntdir)
        elif pntdir.is_dir():
            raise FileExistsError(f"Directory '{pntdir}' exists")
        pntdir.mkdir(exist_ok=False)

        meta_writer = drf.DigitalMetadataWriter(str(pntdir), 3600, 3600, 1, 2, "meta")  # directory

        # TODO: fix so that pointing data is correct
        pointing_data = {
            "azimuth": 0,
            "elevation": 1,
        }

        meta_writer.write(tx_start_samp, pointing_data)

        # ------------ Metadata ---------------
        write_metadata(exp_def, bounds_params, dstdir)

        if rf_writer:
            rf_writer.close()

    return simulated_signal


def write_metadata(exp_def: ExpDef, bounds_params: BoundParams, dstdir: Optional[Path] = None) -> None:
    meta = configparser.ConfigParser()
    meta.add_section(Metaparam.EXPERIMENT)
    meta.add_section(Metaparam.BOUNDS)
    exp = meta[Metaparam.EXPERIMENT]
    bounds = meta[Metaparam.BOUNDS]

    for key, value in asdict(exp_def).items():
        if key == "name":
            result = re.search(r"^(\D*)(.*)$", value)
            exp[Expparam.NAME] = result.group(1)[:-1] if result else "unknown"
            exp[Expparam.VERSION] = result.group(2) if result else "unknown"
            continue
        if value is not None:
            exp[key] = str(value)

    for key, value in bounds_params._asdict().items():
        if value is not None:
            bounds[key] = str(value)
    # write metadata file
    if dstdir:
        metafile = dstdir.parent / "metadata.ini"
        with open(metafile, "w") as f:
            meta.write(f)
