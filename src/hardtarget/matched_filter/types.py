"""General Matched Filter Types, usable for all matched filter processes"""

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from hardtarget.types.types import (
    CfgParams,
    ProParams,
)


@dataclass(frozen=True)
class TargetEstimationCfgParams(CfgParams):
    """
    MF specific configuration parameters, can be configured under the mf section in the .ini file

    Args:
        TODO
    """

    range_gate_sub_resolution: int = 1
    frequency_decimation: int = 1
    clutter_length: int = 0
    min_acceleration: float = -200.0
    max_acceleration: float = 200.0
    optimization: bool = False


@dataclass(frozen=True)
class TargetEstimationProParams(ProParams):
    """
    MF specific process parameters

    Args:
        TODO
    """

    decimated_read_length: int = 0
    il0_rgs: npt.NDArray[np.int32] = field(default_factory=lambda: np.empty(2, dtype=np.int32))
    ranges: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(2, dtype=np.float64))
    il1_rx_window_indices: npt.NDArray[np.int32] = field(default_factory=lambda: np.empty(2, dtype=np.int32))
    il0_rx_window_indices: npt.NDArray[np.int32] = field(default_factory=lambda: np.empty(2, dtype=np.int32))
    il0_dec_rx_window_indices: npt.NDArray[np.int32] = field(
        default_factory=lambda: np.empty(2, dtype=np.int32)
    )
    range_rates: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(2, dtype=np.float64))


@dataclass(frozen=True)
class ExtendedTargetEstimationProParams(TargetEstimationProParams):
    inds_accelerations: npt.NDArray[np.int32] = field(default_factory=lambda: np.empty(2, dtype=np.int32))
    accelerations: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(2, dtype=np.float64))
    acceleration_phasors: npt.NDArray[np.complexfloating] = field(
        default_factory=lambda: np.empty(2, dtype=np.complex64)
    )
    fgmf_acceleration_phasors: npt.NDArray[np.complexfloating] = field(
        default_factory=lambda: np.empty(2, dtype=np.complex64)
    )


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
