"""Plotting tools for raw data"""

import logging
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from radardef import DataLoader, ExpDef
from scipy.fft import fft, fftfreq

from hardtarget.process.utils import sample_interval_to_closest_ipp
from hardtarget.types import Bounds
from hardtarget.utils.range_conversion import unit_to_range_gate
from hardtarget.utils.time_conversion import time_interval_to_sample_bound, ts_from_str

logger = logging.getLogger(__name__)


def rti(
    ax: Axes,
    data_loader: DataLoader,
    start_time: Optional[np.datetime64 | float | int | str] = None,
    end_time: Optional[np.datetime64 | float | int | str] = None,
    relative_time: bool = False,
    keep_tx: bool = False,
    axis_units: bool = False,
    log: bool = False,
    start_range_gate: Optional[int | float] = None,
    end_range_gate: Optional[int | float] = None,
    range_gate_unit: str = "sample",
    monostatic: bool = False,
    colorbar: bool = True,
    pcolormesh_kw: dict = {},
) -> tuple[Axes, list[QuadMesh]]:
    """
    Simple function to plot the range-time intensity information of complex raw voltage data.

    The start and stop times can be in format of:
    absolute time: (np.datetime64, datetime.datetime, string [ISOT])
    time relative lower bound: (float [seconds])
    if None, use drf-bound. To use relative time, toggle the flag `relative_time`

    Args:
        ax: Axes,
        data_loader: Data loader object to acess the data.
        start_time (optional): Start time to inspect.
        end_time(optional): End time to inspect.
        relative_time(optional): Should relative time be used.
        keep_tx(optional): Keep tx signal, if false the tx samples will be set to 0.
        axis_units(optional): Add units to axis.
        log(optional): Should power be calculated via absolute value or logarithmic.
        start_range_gate(optional): Start range gate to inspect.
        end_range_gate(optional): End range gate to inspect.
        range_gate_unit(optional): Unit of range gate.
        monostatic(optional): Receiver and tranceiver located at the same location.
        colorbar(optional): Adds a colorbar to the plot.
        pcolormesh_kw(optional): pcolormesh kwargs.

    Returns:
        Axis and pmesh
    """

    if isinstance(start_time, str):
        try:
            ts_from_str(start_time)
            start_time = int(ts_from_str(start_time) * 1e6)
        except ValueError:
            start_time = int(start_time)
    elif isinstance(start_time, float):
        start_time = int(start_time * 1e6)

    if isinstance(end_time, str):
        try:
            ts_from_str(end_time)
            end_time = int(ts_from_str(end_time) * 1e6)
        except ValueError:
            end_time = int(end_time)
    elif isinstance(end_time, float):
        end_time = int(end_time * 1e6)

    # Extract bounds
    if start_time or end_time:
        request_bounds = time_interval_to_sample_bound(
            time_bounds=Bounds(
                int(data_loader.epoch_bounds.ts_start_usec), int(data_loader.epoch_bounds.ts_end_usec)
            ),
            start_time=start_time,
            end_time=end_time,
            sample_rate=data_loader.exp_def.sample_rate,
            relative_time=relative_time,
        )
    else:
        request_bounds = Bounds(*data_loader.bounds(data_loader.exp_def.rx_channels[0]))

    samp_bounds = sample_interval_to_closest_ipp(request_bounds, data_loader.exp_def.ipp_samps)

    # Extract data within bounds
    n_samp = samp_bounds.end - samp_bounds.start
    if n_samp == 0:
        raise ValueError(
            f"Number of samples cannot be 0 for RTI plot ({samp_bounds.start=} {samp_bounds.end=})"
        )

    data_vec = data_loader.read(
        channel=data_loader.exp_def.rx_channels, start_sample=samp_bounds.start, vector_length=n_samp
    )

    if data_vec.ndim > 1:
        data_vec = np.sum(data_vec, axis=0)

    # Define experiment tx and rx intervals
    def usec_to_sample(t_usec: int) -> int:
        return int(t_usec / data_loader.exp_def.t_samp_usec)

    t_rx_start_samp = usec_to_sample(data_loader.exp_def.t_rx_start_usec)
    t_rx_end_samp = usec_to_sample(data_loader.exp_def.t_rx_end_usec)
    t_tx_start_samp = usec_to_sample(data_loader.exp_def.t_tx_start_usec)
    t_tx_end_samp = usec_to_sample(data_loader.exp_def.t_tx_end_usec)
    t_cal_on_samp = (
        usec_to_sample(int(data_loader.exp_def.t_cal_on_usec))
        if data_loader.exp_def.t_cal_on_usec is not None
        else 0
    )
    t_cal_off_samp = (
        usec_to_sample(int(data_loader.exp_def.t_cal_off_usec))
        if data_loader.exp_def.t_cal_off_usec is not None
        else 0
    )

    samp_vec = np.arange(data_loader.exp_def.ipp_samps)
    rg_vec = np.arange(t_rx_start_samp, t_rx_end_samp, 1) - t_tx_start_samp
    rt_vec = rg_vec / data_loader.exp_def.sample_rate

    if monostatic:
        rt_vec *= 0.5
        if start_range_gate is not None:
            start_range_gate *= 2
        if end_range_gate is not None:
            end_range_gate *= 2

    mat_shape = (data_vec.size // data_loader.exp_def.ipp_samps, data_loader.exp_def.ipp_samps)
    data_ipp_vec = data_vec.reshape(mat_shape).T

    il0_rg0, il0_rg1 = extract_requested_range_gates(
        start_range_gate, end_range_gate, range_gate_unit, data_loader.exp_def
    )
    assert il0_rg0 >= t_rx_start_samp, (
        f"requested start range gate {il0_rg0} before measurement start {t_rx_start_samp}"
    )
    assert il0_rg0 <= t_rx_end_samp, (
        f"requested start range gate {il0_rg0} after measurement end {t_rx_end_samp}"
    )

    assert il0_rg1 >= t_rx_start_samp, (
        f"requested end range gate {il0_rg1} before measurement start {t_rx_start_samp}"
    )
    assert il0_rg1 <= t_rx_end_samp, (
        f"requested end range gate {il0_rg1} after measurement end {t_rx_end_samp}"
    )

    data_ipp_vec = data_ipp_vec[il0_rg0:il0_rg1, :]
    samp_vec = samp_vec[il0_rg0:il0_rg1]
    rt_vec = rt_vec[il0_rg0:il0_rg1]

    # Remove tx-signal (if it exists) and null calibration signal
    if not keep_tx:
        tx_samps = np.logical_and(samp_vec <= t_tx_end_samp, samp_vec >= t_tx_start_samp)
        data_ipp_vec[tx_samps, :] = 0
        data_ipp_vec[t_cal_on_samp:t_cal_off_samp, :] = 0

    # Calculate signal power
    with warnings.catch_warnings(action="ignore", category=RuntimeWarning):
        powsum = np.log10(np.abs(data_ipp_vec) ** 2) if log else np.abs(data_ipp_vec) ** 2

    # Plot data
    if not axis_units:
        X, Y = np.meshgrid(
            np.arange(data_ipp_vec.shape[1]),
            samp_vec,
        )
        ax.set_xlabel("IPP")
        ax.set_ylabel("Level-0 sample")
    else:
        X, Y = np.meshgrid(
            np.arange(data_ipp_vec.shape[1]) * data_loader.exp_def.t_ipp_usec * 1e-6,
            1e-3 * rt_vec * constants.c,
        )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Range [km]")

    pmesh = ax.pcolormesh(X, Y, powsum, **pcolormesh_kw)

    if colorbar:
        cbar = plt.colorbar(pmesh, ax=ax)
        cbar.set_label("Power [arbitrary units]")

    return ax, [pmesh]


def fti(
    ax: Axes,
    data_loader: DataLoader,
    start_time: Optional[np.datetime64 | int | str] = None,
    end_time: Optional[np.datetime64 | int | str] = None,
    relative_time: bool = False,
    keep_tx: bool = False,
    axis_units: bool = False,
    log: bool = False,
    start_range_gate: Optional[int] = None,
    end_range_gate: Optional[int] = None,
    range_gate_unit: str = "sample",
    monostatic: bool = False,
    colorbar: bool = True,
    pcolormesh_kw: dict = {},
) -> tuple[Axes, list[QuadMesh]]:
    """
    Simple function to plot the frequency-time intensity information of complex raw voltage data.

    The start and stop times can be in format of:
    absolute time: (np.datetime64, datetime.datetime, string [ISOT])
    time relative lower bound: (float [seconds])
    if None, use drf-bound. To use relative time, toggle the flag `relative_time`

    Args:
        ax: Axes,
        data_loader: Data loader object to acess the data.
        start_time (optional): Start time to inspect.
        end_time(optional): End time to inspect.
        relative_time(optional): Should relative time be used.
        keep_tx(optional): Keep tx signal, if false the tx samples will be set to 0.
        axis_units(optional): Add units to axis.
        log(optional): Should power be calculated via absolute value or logarithmic.
        start_range_gate(optional): Start range gate to inspect.
        end_range_gate(optional): End range gate to inspect.
        range_gate_unit(optional): Unit of range gate.
        monostatic(optional): Receiver and tranceiver located at the same location.
        colorbar(optional): Adds a colorbar to the plot.
        pcolormesh_kw(optional): pcolormesh kwargs.

    Returns:
        Axis and pmesh
    """

    if isinstance(start_time, str):
        try:
            ts_from_str(start_time)
            start_time = int(ts_from_str(start_time) * 1e6)
        except ValueError:
            start_time = int(start_time)

    if isinstance(end_time, str):
        try:
            ts_from_str(end_time)
            end_time = int(ts_from_str(end_time) * 1e6)
        except ValueError:
            end_time = int(end_time)

    # Extract bounds
    if start_time or end_time:
        request_bounds = time_interval_to_sample_bound(
            time_bounds=Bounds(
                int(data_loader.epoch_bounds.ts_start_usec), int(data_loader.epoch_bounds.ts_end_usec)
            ),
            start_time=start_time,
            end_time=end_time,
            sample_rate=data_loader.exp_def.sample_rate,
            relative_time=relative_time,
        )
    else:
        request_bounds = Bounds(*data_loader.bounds(data_loader.exp_def.rx_channels[0]))

    samp_bounds = sample_interval_to_closest_ipp(request_bounds, data_loader.exp_def.ipp_samps)

    # Extract data within bounds
    n_samp = samp_bounds.end - samp_bounds.start
    data_vec = data_loader.read(
        channel=data_loader.exp_def.rx_channels, start_sample=samp_bounds.start, vector_length=n_samp
    )

    if data_vec.ndim > 1:
        data_vec = np.sum(data_vec, axis=0)

    # Define experiment tx and rx intervals
    def usec_to_sample(t_usec: int) -> int:
        return int(t_usec / data_loader.exp_def.t_samp_usec)

    t_rx_start_samp = usec_to_sample(data_loader.exp_def.t_rx_start_usec)
    t_rx_end_samp = usec_to_sample(data_loader.exp_def.t_rx_end_usec)
    t_tx_start_samp = usec_to_sample(data_loader.exp_def.t_tx_start_usec)
    t_tx_end_samp = usec_to_sample(data_loader.exp_def.t_tx_end_usec)
    t_cal_on_samp = (
        usec_to_sample(int(data_loader.exp_def.t_cal_on_usec))
        if data_loader.exp_def.t_cal_on_usec is not None
        else 0
    )
    t_cal_off_samp = (
        usec_to_sample(int(data_loader.exp_def.t_cal_off_usec))
        if data_loader.exp_def.t_cal_off_usec is not None
        else 0
    )

    range_T = t_tx_start_samp / data_loader.exp_def.sample_rate
    samp_vec = np.arange(data_loader.exp_def.ipp_samps)
    rt_vec = np.arange(t_rx_end_samp - t_rx_start_samp) * data_loader.exp_def.t_samp_usec - range_T

    if monostatic:
        rt_vec *= 0.5
        if start_range_gate is not None:
            start_range_gate *= 2
        if end_range_gate is not None:
            end_range_gate *= 2

    mat_shape = (data_vec.size // data_loader.exp_def.ipp_samps, data_loader.exp_def.ipp_samps)
    data_ipp_vec = data_vec.reshape(mat_shape).T

    il0_rg0, il0_rg1 = extract_requested_range_gates(
        start_range_gate, end_range_gate, range_gate_unit, data_loader.exp_def
    )
    assert il0_rg0 >= t_rx_start_samp, (
        f"requested start range gate {il0_rg0} before measurement start {t_rx_start_samp}"
    )
    assert il0_rg0 <= t_rx_end_samp, (
        f"requested start range gate {il0_rg0} after measurement end {t_rx_end_samp}"
    )

    assert il0_rg1 >= t_rx_start_samp, (
        f"requested end range gate {il0_rg1} before measurement start {t_rx_start_samp}"
    )
    assert il0_rg1 <= t_rx_end_samp, (
        f"requested end range gate {il0_rg1} after measurement end {t_rx_end_samp}"
    )

    data_ipp_vec = data_ipp_vec[il0_rg0:il0_rg1, :]
    samp_vec = samp_vec[il0_rg0:il0_rg1]

    # Remove tx-signal (if it exists) and null calibration signal
    if not keep_tx:
        tx_samps = np.logical_and(samp_vec <= t_tx_end_samp, samp_vec >= t_tx_start_samp)
        data_ipp_vec[tx_samps, :] = 0
        data_ipp_vec[t_cal_on_samp:t_cal_off_samp, :] = 0

    # Calculate spectrums
    powspec = np.abs(np.fft.fftshift(fft(data_ipp_vec, axis=0), axes=(0,)))
    freqs = np.fft.fftshift(fftfreq(data_ipp_vec.shape[0], 1.0 / data_loader.exp_def.sample_rate))
    samp_freqs = np.fft.fftshift(fftfreq(data_ipp_vec.shape[0]))
    if log:
        powspec = np.log10(powspec)

    # Plot data
    if not axis_units:
        X, Y = np.meshgrid(
            np.arange(data_ipp_vec.shape[1]),
            samp_freqs,
        )
        ax.set_xlabel("IPP")
        ax.set_ylabel("Frequency [1/samples]")
    else:
        X, Y = np.meshgrid(
            np.arange(data_ipp_vec.shape[1]) * data_loader.exp_def.t_ipp_usec * 1e-6,
            freqs,
        )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")

    pmesh = ax.pcolormesh(X, Y, powspec, **pcolormesh_kw)

    if colorbar:
        cbar = plt.colorbar(pmesh, ax=ax)
        cbar.set_label("Power [arbitrary units]")

    return ax, [pmesh]


def extract_requested_range_gates(
    start_range_gate: int | float | None,
    end_range_gate: int | float | None,
    range_gate_unit: str,
    exp_def: ExpDef,
) -> tuple[int, int]:
    if start_range_gate is None:
        il0_rg0 = exp_def.t_rx_start_usec / exp_def.t_samp_usec
    else:
        rg0 = unit_to_range_gate(
            val=start_range_gate,
            unit=range_gate_unit,
            sample_rate=exp_def.sample_rate,
        )
        il0_rg0 = rg0 + (exp_def.t_tx_start_usec / exp_def.t_samp_usec)

    if end_range_gate is None:
        il0_rg1 = exp_def.t_rx_end_usec / exp_def.t_samp_usec
    else:
        rg1 = unit_to_range_gate(
            val=end_range_gate,
            unit=range_gate_unit,
            sample_rate=exp_def.sample_rate,
        )
        il0_rg1 = rg1 + (exp_def.t_tx_start_usec / exp_def.t_samp_usec)

    return int(il0_rg0), int(il0_rg1)
