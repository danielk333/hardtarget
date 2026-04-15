from typing import Optional

import numpy as np
import numpy.typing as npt

from hardtarget.optimization.types import MFOptimizeOutArgs, OptimizeCfgParams, OptimizeProParams
from hardtarget.plotting.utils import _convert, to_relative_range_gate
from hardtarget.types import ExpDef
from hardtarget.utils import noise


def plot_optimization_peaks(
    axes: npt.NDArray,  # of type Axes
    out_data: MFOptimizeOutArgs,
    exp: ExpDef,
    cfg: OptimizeCfgParams,
    pro: OptimizeProParams,
    monostatic: bool = True,
    snr_dB_limit: Optional[float] = None,
) -> npt.NDArray:  # of type Axes
    """
    Plot peaks after optimization.
    If limit is set data points above limit will be highlighted.
    """

    nf_vec = np.nanmedian(out_data.dc, axis=0)
    nf_vec = nf_vec.reshape((1, nf_vec.size))
    nf_range = np.nanmedian(nf_vec, axis=0)

    range_gates = to_relative_range_gate(out_data.r_vec_opt, cfg, exp)

    snr = noise.snr(out_data.peak_vals, nf_range, range_gates=range_gates)

    snrdb = 10 * np.log10(snr)
    if snr_dB_limit:
        inds = snrdb > snr_dB_limit
    else:
        inds = np.zeros(out_data.t.shape, dtype=np.bool)
    not_inds = np.logical_not(inds)

    r0 = _convert(out_data.r_vec_opt, monostatic=monostatic)
    v0 = _convert(out_data.v_vec_opt, monostatic=monostatic)
    a0 = _convert(out_data.a_vec_opt, monostatic=monostatic, km=False)

    _inds0_sty = dict(marker="x", alpha=0.5, ls="none", color="r")
    _not0_inds_sty = dict(marker="x", alpha=0.5, ls="none", color="b")
    axes[0, 0].plot(out_data.t[inds], r0[inds], **_inds0_sty)
    axes[0, 1].plot(out_data.t[inds], v0[inds], **_inds0_sty)
    axes[1, 0].plot(out_data.t[inds], a0[inds], **_inds0_sty)
    axes[1, 1].plot(out_data.t[inds], np.sqrt(snr[inds]), **_inds0_sty)
    axes[1, 1].plot(out_data.t[not_inds], np.sqrt(snr[not_inds]), **_not0_inds_sty)

    return axes
