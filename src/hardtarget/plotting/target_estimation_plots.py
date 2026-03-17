"""Plotting tools for analysed output"""

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.lines import Line2D

from hardtarget.target_estimation.types import (
    ExtendedTargetEstimationProParams,
    MFOutArgs,
    TargetEstimationCfgParams,
)
from hardtarget.types import ExpParams

from .utils import _convert


def plot_peaks(
    axes: npt.NDArray,  # of type Axes
    out_data: MFOutArgs,
    exp: ExpParams,
    cfg: TargetEstimationCfgParams,
    pro: ExtendedTargetEstimationProParams,
    monostatic: bool = True,
    snr_dB_limit: float = 15.0,
) -> tuple[npt.NDArray, None]:  # of type Axes
    """
    Plot peaks

    Args:
        axes: Matplotlib axes matrix for subplotting
        out_data: Matched filter output data
        exp: Experiment parameters
        cfg: Configuration parameters
        pro: Process parameters
        monostatic (optional): Receiver and tranceiver located at the same location.
        snr_dB_limit (optional): Signal to noise decibel limit, filter out indexes that does not meet the
                                 limit.

    Returns:
        Updated axes
    """

    r_inds = np.argmax(out_data.snr, axis=1)
    coh_inds = np.arange(out_data.vals.shape[0])

    min_acc = cfg.min_acceleration
    max_acc = cfg.max_acceleration

    snr = out_data.snr

    # TODO: use a interpolation of nf-range to determine the SNR of the optimized results
    # snr = noise.snr(data["gmf_optimized"], data["nf_range"])
    snr = snr[coh_inds, r_inds]
    snrdb = 10 * np.log10(snr)

    inds = snrdb > snr_dB_limit
    not_inds = np.logical_not(inds)

    _inds_sty = dict(marker=".", ls="none", color="r")
    _not_inds_sty = dict(marker=".", ls="none", color="b")

    t = out_data.t
    r = _convert(out_data.r_vec, monostatic=monostatic)
    v = _convert(out_data.v_vec, monostatic=monostatic)
    a = _convert(out_data.a_vec, monostatic=monostatic, km=False)

    axes[0, 0].plot(t[inds], r[inds], **_inds_sty)
    axes[0, 0].plot(t[not_inds], r[not_inds], **_not_inds_sty)
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("range [km]")

    axes[0, 1].plot(t[inds], v[inds], **_inds_sty)
    axes[0, 1].plot(t[not_inds], v[not_inds], **_not_inds_sty)
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("range rate [km/s]")

    axes[1, 0].plot(t[inds], a[inds], **_inds_sty)
    axes[1, 0].plot(t[not_inds], a[not_inds], **_not_inds_sty)
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("acceleration [m/s/s]")
    axes[1, 0].set_ylim([min_acc, max_acc])

    axes[1, 1].plot(t[inds], np.sqrt(snr[inds]), **_inds_sty)
    axes[1, 1].plot(t[not_inds], np.sqrt(snr[not_inds]), **_not_inds_sty)
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("sqrt(SNR)")

    return axes, None


def plot_detections(
    axes: npt.NDArray,  # of type axes
    out_data: MFOutArgs,
    exp: ExpParams,
    cfg: TargetEstimationCfgParams,
    pro: ExtendedTargetEstimationProParams,
    monostatic: bool = True,
    snr_dB_limit: float = 15.0,
) -> tuple[npt.NDArray, list[list[Line2D]]]:  # of type Axes
    """
    Plot detections

    Args:
        axes: Matplotlib axes matrix for subplotting
        out_data: Matched filter output data
        exp: Experiment parameters
        cfg: Configuration parameters
        pro: Process parameters
        monostatic (optional): Receiver and tranceiver located at the same location.
        snr_dB_limit (optional): Signal to noise decibel limit, filter out indexes that does not meet the
                                 limit.

    Returns:
        Updated axes and handles of the axis
    """

    r_inds = np.argmax(out_data.snr, axis=1)
    coh_inds = np.arange(out_data.vals.shape[0])

    min_acc = cfg.min_acceleration
    max_acc = cfg.max_acceleration

    nf_vec = np.nanmedian(out_data.dc, axis=0)
    nf_vec = nf_vec.reshape((1, nf_vec.size))

    snr = out_data.snr
    snr = snr[coh_inds, r_inds]
    snrdb = 10 * np.log10(snr)

    inds = snrdb > snr_dB_limit

    _style = dict(ls="none", marker=".")

    h00 = axes[0, 0].plot(out_data.t[inds], _convert(out_data.r_vec[inds], monostatic=monostatic), **_style)
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("range [km]")

    h01 = axes[0, 1].plot(out_data.t[inds], _convert(out_data.v_vec[inds], monostatic=monostatic), **_style)
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("range rate [km/s]")

    h10 = axes[1, 0].plot(
        out_data.t[inds],
        _convert(out_data.a_vec[inds], km=False, monostatic=monostatic),
        **_style,
    )
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("acceleration [m/s/s]")
    axes[1, 0].set_ylim([min_acc, max_acc])

    h11 = axes[1, 1].plot(out_data.t[inds], snrdb[inds], **_style)
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("SNR [dB]")

    range_data = _convert(out_data.r_vec)
    range_rate_data = _convert(out_data.v_vec)
    acceleration_data = _convert(out_data.a_vec, km=False)

    h11 = axes[0, 2].plot(range_data, range_rate_data, **_style)
    axes[0, 2].set_xlabel("range [km]")
    axes[0, 2].set_ylabel("range rate [km/s]")

    h11 = axes[1, 2].plot(range_data, acceleration_data, **_style)
    axes[1, 2].set_xlabel("range [km]")
    axes[1, 2].set_ylabel("acceleration [km/s/s]")
    axes[1, 2].set_ylim([min_acc, max_acc])

    handles = [[h00, h01], [h10, h11]]
    return axes, handles


def plot_map(
    axes: list[Axes],
    out_data: MFOutArgs,
    exp: ExpParams,
    cfg: TargetEstimationCfgParams,
    pro: ExtendedTargetEstimationProParams,
) -> tuple[list[Axes], list[QuadMesh | list[Line2D]]]:
    """
    Plot map

    Args:
        axes: Matplotlib axes matrix for subplotting
        out_data: Matched filter output data
        exp: Experiment parameters
        cfg: Configuration parameters
        pro: Process parameters

    Returns:
        Updated axes and handles of the axis
    """

    # GMF
    min_range = cfg.min_range_gate
    max_range = cfg.max_range_gate
    gmf_data_dB = 10 * np.log10(np.abs(out_data.vals.T))

    range_num, coh_int_num = gmf_data_dB.shape
    coh_ints = np.arange(0, coh_int_num)
    range_gates = np.arange(min_range, max_range, cfg.range_gate_step / cfg.range_gate_sub_resolution)

    X, Y = np.meshgrid(coh_ints, range_gates)
    h00 = axes[0].pcolormesh(X, Y, gmf_data_dB)
    axes[0].set_xlabel("Integration number")
    axes[0].set_ylabel("Range gates")
    axes[0].set_title("GMF decoded power [dB]")

    # Noise data
    nf_data_dB = 10 * np.log10(out_data.dc.T)
    h01 = axes[1].pcolormesh(X, Y, nf_data_dB)
    axes[1].set_xlabel("Integration number")
    axes[1].set_ylabel("Range gates")
    axes[1].set_title("Estimated noise power [dB]")

    # Noise floor
    nf_vec = np.nanmedian(out_data.dc, axis=0)
    nf_vec = nf_vec.reshape((1, nf_vec.size))
    nf_range = np.nanmedian(nf_vec, axis=0)

    h10 = axes[2].plot(range_gates, nf_range)
    axes[2].set_xlabel("Ranges [km]")
    axes[2].set_ylabel("Median noise power [arb. unit]")
    axes[2].set_title("Range dependant noise floor")

    handles = [h00, h01, h10]
    return axes, handles  # type: ignore[return-value]
