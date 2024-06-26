import matplotlib.pyplot as plt
import logging
import numpy as np
import scipy.constants as constants

from hardtarget.cli.utils import unit_to_range_gate
from hardtarget.drf_utils import time_interval_to_samples

logger = logging.getLogger(__name__)


def rti(
    ax,
    drf_reader,
    drf_params,
    start_time=None,
    end_time=None,
    relative_time=False,
    keep_tx=False,
    axis_units=False,
    log=False,
    start_range_gate=None,
    end_range_gate=None,
    range_gate_unit="sample",
    monostatic=False,
    colorbar=True,
    pcolormesh_kw={},
):
    """
    Simple function to plot the range-time intensity information of complex raw voltage data.

    The start and stop times can be in format of:
    absolute time: (np.datetime64, datetime.datetime, string [ISOT])
    time relative lower bound: (float [seconds])
    if None, use drf-bound. To use relative time, toggle the flag `relative_time`

    """

    channels = drf_reader.get_channels()

    rxchn = drf_params["rx_channel"]
    assert (
        rxchn in channels
    ), f"drf data does not have requested '{rxchn}', existing channels: {channels}"

    # check sample rate
    props = drf_reader.get_properties(rxchn)

    sample_rate = props["samples_per_second"].astype(np.int64)
    T_ipp = drf_params["ipp"] * 1e-6

    T_samp = 1.0 / sample_rate
    ipp_samps = int(T_ipp * sample_rate)

    # Use np.round and case to int to avoid floating point errors in floor
    T_rx_start_samp = np.round(drf_params["rx_start"] * 1e-6 * sample_rate).astype(np.int64)
    T_rx_end_samp = np.round(drf_params["rx_end"] * 1e-6 * sample_rate).astype(np.int64)

    T_tx_start_samp = np.round(drf_params["tx_start"] * 1e-6 * sample_rate).astype(np.int64)
    T_tx_end_samp = np.round(drf_params["tx_end"] * 1e-6 * sample_rate).astype(np.int64)

    T_cal_start_samp = np.round(drf_params["cal_on"] * 1e-6 * sample_rate).astype(np.int64)
    T_cal_end_samp = np.round(drf_params["cal_off"] * 1e-6 * sample_rate).astype(np.int64)

    drf_bounds = drf_reader.get_bounds(rxchn)

    request_bounds = time_interval_to_samples(
        start_time, end_time, drf_bounds, sample_rate, relative_time=relative_time
    )

    ipp_n0 = (request_bounds[0] - drf_bounds[0]) // ipp_samps
    ipp_n1 = (request_bounds[1] - drf_bounds[0]) // ipp_samps

    bounds = [
        (ipp_n0 + 1) * ipp_samps + drf_bounds[0],
        ipp_n1 * ipp_samps + drf_bounds[0],
    ]

    # The following assumes continuous samples in the DRF
    # check blocks rx channel
    blocks = drf_reader.get_continuous_blocks(bounds[0], bounds[1], rxchn)
    if len(blocks) > 1:
        logger.warning(f"multiple continuous blocks: {len(blocks)}")

    data_vec = drf_reader.read_vector_1d(bounds[0], bounds[1] - bounds[0], rxchn)

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
    data_vec = data_vec.reshape(mat_shape).T

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

    data_vec = data_vec[il0_rg0:il0_rg1, :]
    samp_vec = samp_vec[il0_rg0:il0_rg1]
    rt_vec = rt_vec[il0_rg0:il0_rg1]

    # Remove tx-signal (if it exists) and null calibration signal
    if not keep_tx:
        if T_rx_start_samp < T_tx_end_samp:
            tx_samps = np.logical_and(
                samp_vec <= T_tx_end_samp,
                samp_vec >= T_tx_start_samp
            )
            data_vec[tx_samps, :] = 0
        data_vec[T_cal_start_samp:T_cal_end_samp, :] = 0

    powsum = np.log10(np.abs(data_vec) ** 2) if log else np.abs(data_vec) ** 2

    """
    Sets pyplot to classic rendering, then renders the powersum unto it. We then add
    a colorbar and the y and x-label before saving it. At the moment it only saves
    to plot.png.
    """

    if not axis_units:
        X, Y = np.meshgrid(
            np.arange(data_vec.shape[1]),
            samp_vec,
        )
        ax.set_xlabel("IPP")
        ax.set_ylabel("Level-0 sample")
    else:
        X, Y = np.meshgrid(
            np.arange(data_vec.shape[1]) * T_ipp,
            1e-3 * rt_vec * constants.c,
        )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Range [km]")

    pmesh = ax.pcolormesh(X, Y, powsum, **pcolormesh_kw)

    if colorbar:
        cbar = plt.colorbar(pmesh, ax=ax)
        cbar.set_label("Power [arbitrary units]")

    return ax, [pmesh]
