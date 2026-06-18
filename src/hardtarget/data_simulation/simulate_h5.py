"""
Simulate multichannel h5 data.
"""

import datetime as dt
from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np
import numpy.typing as npt
from pyant.models.array import Array, ArrayParams
from radardef.types import ExpDef

from hardtarget.data_simulation.utils import TrajectoryFunction, noise_generator, waveform_generator
from hardtarget.utils.range_conversion import range_to_range_gate


def default_trajectory_function(t: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Args:
        t: timepoints in seconds

    Returns:
        Two way range (len(t),) and Trajectory as a (3,len(t)) numpy array

    """

    # Initial values
    r0: float = 200e3
    v0: float = -2.0e3
    a0: float = -0.20e3

    # Initial position over radar
    k0 = np.array([0.1, 0.1, 0.8])

    # 3d position
    x_start = k0 * r0

    # Velocity vector, object moving along x-axis
    v_vec = np.array([1, 0, 0])

    distance_traveled = v0 * t + a0 * 0.5 * t**2

    trajectory = x_start[:, None] + v_vec[:, None] * distance_traveled[None, :]
    r = np.linalg.norm(trajectory, axis=0)
    two_way_range = 2 * r
    rx_k_vecs = trajectory / r

    # trajectory
    return two_way_range, rx_k_vecs


def simulate_h5(
    output_dir: Path,
    exp_def: ExpDef,
    start_time: dt.datetime | int,
    end_time: dt.datetime | int,
    target_start_time: Optional[dt.datetime | int] = None,
    target_end_time: Optional[dt.datetime | int] = None,
    target_relative_time: bool = False,
    trajectory_function: TrajectoryFunction = default_trajectory_function,
    beam: Optional[Array] = None,
    beam_params: Optional[ArrayParams] = None,
    snr_function: Optional[Callable] = None,
    noise_sigma: Optional[float] = None,
) -> Path:
    """
    Simulate H5 data from a multichannel radar, note only rx samples are stored. The dataloader is padding the data later.

    Args:
        output_dir: Output directory.
        exp_def: Experiment definition.
        start_time: Start time of measurement, either in datetime or usec.
        end_time: End time of measurement, either in datetime or usec.
        target_start_time (optional): Start time of object in the measurement, either in datetime or usec.
        target_end_time (optional): End time of object in the measurement, either in datetime or usec.
        target_relative_time (optional): If the target times are relative to the start time.
        trajectory_function (optional): Function modeling the trajectory of a object over time.
        beam (optional): Beam of the radar stations. Only optional if no object is present.
        beam_params (optional): Beam paramters of the radar stations. Only optional if no object is present.
        snr_function (optional): Function modeling the signal to noise ratio over time.
        noise_sigma (optional): Noise.
    """

    # Extract timepoints

    if isinstance(target_start_time, dt.datetime) and isinstance(target_end_time, dt.datetime):
        target_start_us = target_start_time.timestamp() * 1e6
        target_end_us = target_end_time.timestamp() * 1e6
    elif isinstance(target_start_time, int) and isinstance(target_end_time, int):
        target_start_us = target_start_time
        target_end_us = target_end_time
    else:
        raise ValueError("Target start and end time should be in the same format")

    if isinstance(start_time, dt.datetime) and isinstance(end_time, dt.datetime):
        start_time_us = start_time.timestamp() * 1e6
        end_time_us = end_time.timestamp() * 1e6
    elif isinstance(start_time, int) and isinstance(end_time, int):
        start_time_us = start_time
        end_time_us = end_time
        start_time = dt.datetime.fromtimestamp(start_time * 1e-6)
        end_time = dt.datetime.fromtimestamp(end_time * 1e-6)
    else:
        raise ValueError("Measurement start and end time should be in the same format")

    n_ipps = int((end_time_us - start_time_us) // exp_def.t_ipp_usec)
    if target_relative_time:
        target_start_ipp = target_start_us // exp_def.t_ipp_usec
        target_end_ipp = target_end_us // exp_def.t_ipp_usec
    else:
        target_start_ipp = (target_start_us - start_time_us) // exp_def.t_ipp_usec
        target_end_ipp = (target_end_us - start_time_us) // exp_def.t_ipp_usec

    # Create output dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate rx and tx sections
    rx_start_samp = int(exp_def.t_rx_start_usec / exp_def.t_samp_usec)
    rx_end_samp = int(exp_def.t_rx_end_usec / exp_def.t_samp_usec)
    rx_samps = rx_end_samp - rx_start_samp
    tx_start_samp = int(exp_def.t_tx_start_usec / exp_def.t_samp_usec)
    tx_end_samp = int(exp_def.t_tx_end_usec / exp_def.t_samp_usec)

    # Generate stream of data
    ipp_data = np.zeros((len(exp_def.rx_channels), n_ipps, rx_samps), dtype=np.complex64)
    for i in range(n_ipps):
        # Generate tx wave
        tx_wave = waveform_generator(
            n_tx_samps=tx_end_samp - tx_start_samp,
            sample_rate=exp_def.sample_rate,
            baud_length_sec=exp_def.baud_length_usec * 1e-6,
            code=exp_def.code,
            dtype=np.complex64,
        )

        ipp_data[:, i, :] = generate_rx_vectors(
            exp=exp_def,
            ipp_n=i,
            is_object_present=(i >= target_start_ipp and i <= target_end_ipp),
            tx_wave=tx_wave,
            beam=beam,
            beam_params=beam_params,
            noise_sigma=noise_sigma,
            trajectory_function=trajectory_function,
            snr_function=snr_function,
        )

        if noise_sigma:
            ipp_data[:, i] += noise_generator(
                noise_sigma,
                ipp_data[:, i].shape,
                dtype=np.complex64,
            )

    # Split data to files
    n_samps = n_ipps * exp_def.ipp_samps
    n_files = -(n_samps // -exp_def.samples_per_file)
    ipps_per_file = exp_def.samples_per_file // exp_def.ipp_samps
    for i in range(n_files):
        # Extract data for relevant ipps
        data = ipp_data[:, i * ipps_per_file : (i + 1) * ipps_per_file]

        # Calculate file start and end
        file_start_time = start_time + dt.timedelta(
            microseconds=i * exp_def.samples_per_file * exp_def.t_samp_usec
        )
        file_end_time = file_start_time + dt.timedelta(
            microseconds=(data.shape[1] * exp_def.ipp_samps * exp_def.t_samp_usec)
        )
        # Declare file name and location
        output_file = output_dir / (
            file_start_time.strftime("%Y-%m-%dT%H.%M.%S.%f").replace(":", ".") + "000000.h5"
        )

        # Write data
        with h5py.File(output_file, "w") as h5file:
            # Add attributes
            h5file.attrs["filename"] = output_file.name
            h5file.attrs["date"] = file_start_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[0:10]
            h5file.attrs["path"] = str(output_file)
            h5file.attrs["record_start_time"] = file_start_time.strftime("%Y-%m-%dT%H:%M:%S.%f")
            h5file.attrs["record_end_time"] = file_end_time.strftime("%Y-%m-%dT%H:%M:%S.%f")

            # Create dataset
            h5file.create_dataset("data", data=data)
            h5file.create_dataset("rx_channels", data=exp_def.rx_channels)

    return output_dir


def generate_rx_vectors(
    exp: ExpDef,
    ipp_n: int,
    is_object_present: bool,
    tx_wave: npt.NDArray[np.complex64],
    trajectory_function: TrajectoryFunction,
    beam: Optional[Array] = None,
    beam_params: Optional[ArrayParams] = None,
    noise_sigma: Optional[float] = None,
    snr_function: Optional[Callable] = None,
) -> npt.NDArray:
    """
    Generate rx vectors for each channel.

    Args:
        exp: Experiment definition.
        ipp_n: Ipp index
        is_object_present: Should an object be present in the data.
        tx_wave: Transmitted wave.
        trajectory_function: Function modeling the trajectory over time.
        beam (optional): Beam of the radar stations. Only optional if no object is present.
        beam_params (optional): Beam paramters of the radar stations. Only optional if no object is present.
        noise_sigma (optional): Noise.
        snr_function (optional): Function modeling the signal to noise ratio over time.
    Returns:
        Rx waves for one ipp in the shape (Channels, rx_samps)
    """

    rx_samps = int((exp.t_rx_end_usec - exp.t_rx_start_usec) / exp.t_samp_usec)
    rx_vector = np.zeros((len(exp.rx_channels), rx_samps), np.complex64)

    if is_object_present:
        samp_ind = ipp_n * exp.ipp_samps
        tx_start_ind = samp_ind + (exp.t_tx_start_usec / exp.t_samp_usec)
        t_tx_seconds = (np.arange(len(tx_wave)) / exp.sample_rate) + (tx_start_ind / exp.sample_rate)
        two_way_range, rx_k_vecs = trajectory_function(t_tx_seconds)
        if not isinstance(two_way_range, np.ndarray):
            two_way_range = np.array([two_way_range])
        # Check so that range is noticable within the signal, otherwise skip
        if is_range_within_rx_gate(range=two_way_range[0], exp=exp):
            # Generate rx_wave per channel
            rx_waves = np.zeros((len(exp.rx_channels), len(tx_wave)), np.complex64)
            rx_waves[:] = generate_rx_wave(
                exp=exp,
                tx_wave=tx_wave,
                snr=snr_function(t_tx_seconds) if snr_function else 1.0,
                ranges=two_way_range,
                noise_sigma=noise_sigma,
            )
            # If beam available add channel specifics
            if beam and beam_params:
                chs = beam.channel_signals(rx_k_vecs, beam_params)
                rx_waves *= chs if chs.ndim > 1 else chs.reshape(len(exp.rx_channels), 1)

            # allocate the wave correctly within the rx interval
            tx_rx_start_delta = int((exp.t_rx_start_usec - exp.t_tx_start_usec) / exp.t_samp_usec)
            range_gate = range_to_range_gate(two_way_range[0], exp.sample_rate)
            rx_wave_start_samp = range_gate - tx_rx_start_delta

            # Check so that signal fits, else just extract the samps that is within the rx interval
            if (rx_wave_start_samp + rx_waves.shape[1]) > rx_samps:
                n_samps = rx_samps - rx_wave_start_samp
            else:
                n_samps = rx_waves.shape[1]

            rx_vector[:, rx_wave_start_samp : rx_wave_start_samp + n_samps] += rx_waves[:, :n_samps]

    return rx_vector


def generate_rx_wave(
    exp: ExpDef,
    tx_wave: npt.NDArray[np.complex64],
    ranges: npt.NDArray,
    snr: float,
    noise_sigma: Optional[float] = None,
) -> npt.NDArray[np.complex64]:
    """
    Generate an rx wave from the existing tx wave based on what range the object is at.

    Args:
        exp: Experiment definition
        tx_wave: the transmitted wave
        ranges: array of range of object.
        snr: signal to noise ratio.
        noise_sigma (optional): noise
    Returns:
        reciver wave
    """

    if noise_sigma:
        amp = np.sqrt(snr * 2 * noise_sigma**2)
    else:
        amp = np.sqrt(snr)
    # Calculate phase
    phase = np.mod(ranges / exp.wavelength, 1) * np.pi * 2
    # Generate rx wave
    rx_wave = tx_wave * amp * np.exp(1j * phase)

    return rx_wave


def is_range_within_rx_gate(range: float, exp: ExpDef) -> bool:
    """
    For any given range, would it be detected within the rx interval relative to tx start

    Args:
        range: range oj object
        exp: experiment parameters
    Returns:
        If the range gate would be detected within the rx interval at the given range.
    """
    # Convert range to amount of samples
    range_gate = range_to_range_gate(range, exp.sample_rate)
    # What is the sample index of the range gate relative to the tx start
    sample_index = (exp.t_tx_start_usec / exp.t_samp_usec) + range_gate
    # is that sample index within the rx interval? Otherwise it can not be read
    return sample_index >= (exp.t_rx_start_usec / exp.t_samp_usec) and sample_index <= (
        exp.t_rx_end_usec / exp.t_samp_usec
    )
