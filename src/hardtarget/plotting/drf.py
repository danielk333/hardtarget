import matplotlib.pyplot as plt
import logging
import numpy as np
import scipy.constants as constants

from hardtarget.utilities import read_vector_c81d

logger = logging.getLogger(__name__)


def rti(
    ax,
    drf_reader,
    channel,
    start_time=None,
    end_time=None,
    axis_font_size=15,
    title_font_size=11,
    tick_font_size=11,
    title="",
    index_axis=True,
    log=False,
    colorbar=True,
    pcolormesh_kw={},
):
    """
    Simple function to plot the range-time intensity information of complex raw voltage data.
    """

    channels = drf_reader.get_channels()
    if channel not in channels:
        raise ValueError(
            f"drf data does not have requested '{channel}', existing channels: {channels}"
        )

    # check sample rate
    props = drf_reader.get_properties(channel)
    sample_rate = props["samples_per_second"].astype(np.int64)

    T_ipp = props.get("ipp", 0.002)
    T_samp = 1.0 / sample_rate
    ipp_samps = int(T_ipp * sample_rate)

    T_rx_start = props.get("rx_start", 0.001)
    T_rx_start_samp = int(T_rx_start * sample_rate)

    T_rx_end = props.get("rx_end", 0.002)
    T_rx_end_samp = int(T_rx_end * sample_rate)

    bounds = list(drf_reader.get_bounds(channel))

    if start_time is not None:
        if not isinstance(start_time, np.datetime64):
            dt64_t0 = np.datetime64(start_time)
        else:
            dt64_t0 = start_time
        unix_t0 = dt64_t0.astype("datetime64[s]").astype("int64")
        _b0 = unix_t0 * sample_rate
        assert _b0 >= bounds[0], "Given start time is before input data start"
        bounds[0] = _b0

    if end_time is not None:
        if not isinstance(end_time, np.datetime64):
            dt64_t1 = np.datetime64(end_time)
        else:
            dt64_t1 = end_time
        unix_t1 = dt64_t1.astype("datetime64[s]").astype("int64")
        _b1 = int(unix_t1 * sample_rate)
        breakpoint()
        assert _b1 <= bounds[1], "Given end time is after input data end"
        bounds[1] = _b1

    # check blocks rx channel
    blocks = drf_reader.get_continuous_blocks(bounds[0], bounds[1], channel)
    if len(blocks) > 1:
        logger.warning(f"multiple continuous blocks: {len(blocks)}")

    data_vec = read_vector_c81d(drf_reader, bounds[0], bounds[1] - bounds[0], channel)
    # TODO: view based on tx_start and tx_end
    mat_shape = (
        data_vec.size // (ipp_samps - T_rx_start_samp),
        (ipp_samps - T_rx_start_samp),
    )
    data_vec = data_vec.reshape(mat_shape)

    powsum = np.log10(np.abs(data_vec) ** 2) if log else np.abs(data_vec) ** 2

    """
    Sets pyplot to classic rendering, then renders the powersum unto it. We then add
    a colorbar and the y and x-label before saving it. At the moment it only saves
    to plot.png.
    """

    if index_axis:
        X, Y = np.meshgrid(
            np.arange(mat_shape[0]),
            np.arange(mat_shape[1]),
        )
        ax.set_xlabel("IPP", fontsize=axis_font_size)
        ax.set_ylabel("Sample", fontsize=axis_font_size)
    else:
        X, Y = np.meshgrid(
            np.arange(mat_shape[1]) * T_ipp,
            0.5e-3 * (np.arange(mat_shape[0]) * T_samp + T_rx_start) * constants.c,
        )
        ax.set_xlabel("Time [s]", fontsize=axis_font_size)
        ax.set_ylabel("Range [km]", fontsize=axis_font_size)

    pmesh = ax.pcolormesh(X, Y, powsum, **pcolormesh_kw)

    if len(title) > 0:
        ax.set_title(title, fontsize=title_font_size)
    if colorbar:
        cbar = plt.colorbar(pmesh, ax=ax)
        cbar.set_label("Power [arbitrary units]", size=axis_font_size)
        cbar.ax.tick_params(labelsize=tick_font_size)

    for ax_label in ["x", "y"]:
        ax.tick_params(axis=ax_label, labelsize=tick_font_size)

    return ax, [pmesh]
