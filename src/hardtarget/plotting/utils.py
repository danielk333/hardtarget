from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import constants

from hardtarget.types import CfgParams, ExpDef


def _convert(data: Any, km: bool = True, monostatic: bool = True) -> npt.NDArray[Any]:
    """Assume data is [m] and two-way range"""
    _data = data.copy()
    if km:
        _data *= 0.001
    if monostatic:
        _data *= 0.5
    return _data


def to_relative_range_gate(
    ranges: npt.NDArray[np.float64 | np.int64],
    cfg: CfgParams,
    exp: ExpDef,
) -> npt.NDArray[np.int64]:
    sample_rate = exp.sample_rate
    tx_start_samp = exp.t_tx_start_usec / exp.t_samp_usec
    il0_r_samp = (ranges / constants.c) * sample_rate + tx_start_samp
    il0_min_range_gate = cfg.min_range_gate if cfg.min_range_gate > 0 else tx_start_samp + 1
    return il0_r_samp.astype(np.int64) - il0_min_range_gate
