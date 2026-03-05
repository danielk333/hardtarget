"""General Matched Filter Types, usable for all matched filter processes"""

from typing import NamedTuple

import numpy as np
import numpy.typing as npt


class MFVariables(NamedTuple):
    """Container for compacting the variables set by the GMF Grid function."""

    vals: npt.NDArray[np.float32]  # match function values reduced over the requested axis
    dc: npt.NDArray[np.float32]  # 0-frequency gmf output as a function of range
    v_ind: npt.NDArray[np.int32]  # best fitting range-rate
    a_ind: npt.NDArray[np.int32]  # best fitting range-rate change
    tx_pwr: npt.NDArray[np.floating]  # tx power


class MFOutArgs(NamedTuple):
    """Container for compacting the variables set by the GMF function."""

    num_cohints_per_file: int
    ranges: npt.NDArray[np.float64]
    range_rates: npt.NDArray[np.float64]
    accelerations: npt.NDArray[np.float64]
    sample_numbers: npt.NDArray[np.int32]
    vals: npt.NDArray[np.float32]
    dc: npt.NDArray[np.float32]
    v_ind: npt.NDArray[np.int32]
    a_ind: npt.NDArray[np.int32]
    tx_pwr: npt.NDArray[np.floating]
    snr: npt.NDArray[np.floating]
    r_vec: npt.NDArray[np.float64]
    v_vec: npt.NDArray[np.float64]
    a_vec: npt.NDArray[np.float64]
    g_vec: npt.NDArray[np.float32]
    pointing_vec: npt.NDArray[np.float32]
    t: npt.NDArray[np.float32]
    epoch: float


class MFOptimizeVariables(NamedTuple):
    """Container for compacting the variables set by the GMF Optimize function."""

    peak: npt.NDArray[np.float64]  # peak location
    peak_val: npt.NDArray[np.float64]  # peak magnitude


class MFOptimizeOutArgs(NamedTuple):
    peaks: npt.NDArray[np.float64]
    peak_vals: npt.NDArray[np.float64]
