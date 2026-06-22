"""General Matched Filter Types, usable for all matched filter processes"""

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from hardtarget.types import (
    CfgParams,
    ProParams,
)


@dataclass(frozen=True)
class TargetEstimationCfgParams(CfgParams):
    """
    Target estimation specific configuration parameters, can be configured under the target_estimation
    section in the .ini file

    Args:
        range_gate_sub_resolution: For increased accuracy subresolution can be used, increases resolution by
                generating sub samples between then available samples. This will increase computational time.
        frequency_decimation: How many samples to step in range in the first stage of the course search.
        clutter_length: How many samples to remove due to ground clutter
        min_acceleration: Minimum acceleration to search for (m/s^2)
        max_acceleration: Maximum acceleration to search for (m/s^2)
    """

    range_gate_sub_resolution: int = 1
    refine_doppler: bool = True
    refine_acceleration: bool = False
    frequency_decimation: int = 1
    clutter_length: int = 0
    min_acceleration: float = -200.0
    max_acceleration: float = 200.0


@dataclass(frozen=True)
class TargetEstimationProParams(ProParams):
    """
    Target estimation general process parameters

    Args:
        decimated_read_length: Decimated read_length based on frequency decimation.
        il0_rgs: Index level 0 range gates.
        ranges: Range gates (including subresolution) in meter, true ranges.
        il1_rx_window_indices: Index level 1 receiver window indices.
        il0_rx_window_indices: Index level 0 receiver window indices.
        il0_dec_rx_window_indices: Index level 0 decimated receiver window indices
                                    (il0_rx_window_indices with frequency decimation).
        range_rates: Range rates
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
    fft_frequencies: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(2, dtype=np.float64))
    # TODO: i shimmied this in here for now, it should probably be propagated down some other way?
    sample_rate: float = 0


# TODO: the name "Extended" should probably be changed as extended targets are a common phrase in
# the radar community and it means a target that is much larger than the range-gates used which is
# probably not what this means
@dataclass(frozen=True)
class ExtendedTargetEstimationProParams(TargetEstimationProParams):
    """
    Target estimation shared process parameters but with seperate implementations
    """

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
    v: npt.NDArray[np.float64]  # best fitting range-rate
    a: npt.NDArray[np.float64]  # best fitting range-rate change
    phi: npt.NDArray[np.float64]  # best fitting phase
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
    tx_pwr: npt.NDArray[np.floating]
    snr: npt.NDArray[np.floating]
    v: npt.NDArray[np.float64]
    a: npt.NDArray[np.float64]
    phi: npt.NDArray[np.float64]
    r_vec: npt.NDArray[np.float64]
    v_vec: npt.NDArray[np.float64]
    a_vec: npt.NDArray[np.float64]
    g_vec: npt.NDArray[np.float32]
    pointing_vec: npt.NDArray[np.float32]
    t: npt.NDArray[np.float32]
    epoch_us: float
