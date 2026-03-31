"""Plotting tools for raw data"""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from radardef import DataLoader

from hardtarget.types import Bounds
from hardtarget.utils.range_conversion import unit_to_range_gate
from hardtarget.utils.time_conversion import time_interval_to_sample_bound

logger = logging.getLogger(__name__)


def rti(
    ax: Axes,
    data_loader: DataLoader,
    start_time: Optional[np.datetime64 | int] = None,
    end_time: Optional[np.datetime64 | int] = None,
    relative_time: bool = False,
    keep_tx: bool = False,
    axis_units: bool = False,
    log: bool = False,
    start_range_gate: Optional[float] = None,
    end_range_gate: Optional[float] = None,
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
    time_bounds_us = Bounds(
        int(data_loader.epoch_bounds.ts_start_usec), int(data_loader.epoch_bounds.ts_end_usec)
    )
    request_bounds = time_interval_to_sample_bound(
        time_bounds=time_bounds_us,
        start_time=start_time,
        end_time=end_time,
        sample_rate=data_loader.experiment.sample_rate,
        relative_time=relative_time,
    )

    ipp_samps = data_loader.experiment.ipp_samps
    start, end = data_loader.bounds(data_loader.channels[0])
    sample_bounds = Bounds(start, end)
    ipp_n0 = (request_bounds.start - sample_bounds.start) // ipp_samps
    ipp_n1 = (request_bounds.end - sample_bounds.start) // ipp_samps

    samp_start = ipp_n0 * ipp_samps + sample_bounds.start
    samp_end = ipp_n1 * ipp_samps + sample_bounds.start
    if samp_end > end:
        samp_end = end

    samp_bounds = Bounds(samp_start, samp_end)

    n_samp = samp_bounds.end - samp_bounds.start
    data_vec = np.zeros((n_samp,), dtype=np.complex128)
    for chnl in data_loader.channels:
        data_vec += data_loader.read(chnl, samp_bounds.start, n_samp)

    data_vec.flatten()

    T_ipp = data_loader.experiment.t_ipp_usec * 1e-6
    sample_rate = data_loader.experiment.sample_rate

    T_samp = data_loader.experiment.t_samp_usec

    # Use np.round and case to int to avoid floating point errors in floor
    T_rx_start_samp = np.round(
        data_loader.experiment.t_rx_start_usec / data_loader.experiment.t_samp_usec
    ).astype(np.int64)
    T_rx_end_samp = np.round(
        data_loader.experiment.t_rx_end_usec / data_loader.experiment.t_samp_usec
    ).astype(np.int64)

    T_tx_start_samp = np.round(
        data_loader.experiment.t_tx_start_usec / data_loader.experiment.t_samp_usec
    ).astype(np.int64)
    T_tx_end_samp = np.round(
        data_loader.experiment.t_tx_end_usec / data_loader.experiment.t_samp_usec
    ).astype(np.int64)

    t_cal_on_usec = (
        data_loader.experiment.t_cal_on_usec if data_loader.experiment.t_cal_on_usec is not None else 0
    )
    t_cal_off_usec = (
        data_loader.experiment.t_cal_off_usec if data_loader.experiment.t_cal_off_usec is not None else 0
    )

    T_cal_start_samp = np.round(t_cal_on_usec / data_loader.experiment.t_samp_usec).astype(np.int64)
    T_cal_end_samp = np.round(t_cal_off_usec / data_loader.experiment.t_samp_usec).astype(np.int64)

    range_T = T_tx_start_samp / sample_rate
    samp_vec = np.arange(ipp_samps)
    rt_vec = np.arange(T_rx_end_samp - T_rx_start_samp) * T_samp - range_T
    if monostatic:
        rt_vec *= 0.5
    if monostatic and start_range_gate is not None:
        start_range_gate *= 2
    if monostatic and end_range_gate is not None:
        end_range_gate *= 2

    mat_shape = (data_vec.size // ipp_samps, ipp_samps)
    data_ipp_vec = data_vec.reshape(mat_shape).T

    if start_range_gate is None:
        il0_rg0 = T_rx_start_samp
    else:
        rg0 = unit_to_range_gate(
            val=start_range_gate,
            unit=range_gate_unit,
            sample_rate=sample_rate,
        )
        il0_rg0 = rg0 + T_tx_start_samp
    assert il0_rg0 >= T_rx_start_samp, (
        f"requested start range gate {il0_rg0} before measurement start {T_rx_start_samp}"
    )
    assert il0_rg0 <= T_rx_end_samp, (
        f"requested start range gate {il0_rg0} after measurement end {T_rx_end_samp}"
    )

    if end_range_gate is None:
        il0_rg1 = T_rx_end_samp
    else:
        rg1 = unit_to_range_gate(
            val=end_range_gate,
            unit=range_gate_unit,
            sample_rate=sample_rate,
        )
        il0_rg1 = rg1 + T_tx_start_samp
    assert il0_rg1 >= T_rx_start_samp, (
        f"requested end range gate {il0_rg1} before measurement start {T_rx_start_samp}"
    )
    assert il0_rg1 <= T_rx_end_samp, (
        f"requested end range gate {il0_rg1} after measurement end {T_rx_end_samp}"
    )

    data_ipp_vec = data_ipp_vec[il0_rg0:il0_rg1, :]
    samp_vec = samp_vec[il0_rg0:il0_rg1]

    # Remove tx-signal (if it exists) and null calibration signal
    if not keep_tx:
        tx_samps = np.logical_and(samp_vec <= T_tx_end_samp, samp_vec >= T_tx_start_samp)
        data_ipp_vec[tx_samps, :] = 0
        data_ipp_vec[T_cal_start_samp:T_cal_end_samp, :] = 0

    powsum = np.log10(np.abs(data_ipp_vec) ** 2) if log else np.abs(data_ipp_vec) ** 2

    if not axis_units:
        X, Y = np.meshgrid(
            np.arange(data_ipp_vec.shape[1]),
            samp_vec,
        )
        ax.set_xlabel("IPP")
        ax.set_ylabel("Level-0 sample")
    else:
        X, Y = np.meshgrid(
            np.arange(data_ipp_vec.shape[1]) * T_ipp,
            1e-3 * rt_vec * constants.c,
        )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Range [km]")

    pmesh = ax.pcolormesh(X, Y, powsum, **pcolormesh_kw)

    if colorbar:
        cbar = plt.colorbar(pmesh, ax=ax)
        cbar.set_label("Power [arbitrary units]")

    return ax, [pmesh]
