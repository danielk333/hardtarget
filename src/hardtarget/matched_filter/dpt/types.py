"""Discrete Polynomial-phase Transform, or DPT type specifics"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from hardtarget.types.types import CfgParams, ProParams


@dataclass(frozen=True)
class DPTCfgParams(CfgParams):
    """DPT specific configuration parameters, can be configured under the dpt section in the .ini file

    Args:
        ipp_delay_parameter: How many IPPs to use for estimating the numerical derivative of the phase of
                             the coherent signal if using the polynomial phase transform method.
                             Also used by the Fast GMF method to calculate the acceleration resolution to
                             sample. A larger number means smaller accelerations can be determined at higher
                             resolution. Note this can never be less or equal then the n_ipp param.

    """

    ipp_delay_parameter: int = 10


@dataclass(frozen=True)
class DPTProParams(ProParams):
    """
    DPT specific process parameters

    Args: TODO
        decimated_ipp_delay_parameter:
        inds_accelerations:
        accelerations:
        acceleration_phasors:
        fgmf_acceleration_phasors:
    """

    decimated_ipp_delay_parameter: int = 1
    inds_accelerations: npt.NDArray[np.int32] = field(default_factory=lambda: np.empty(2, dtype=np.int32))
    accelerations: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(2, dtype=np.float64))
    acceleration_phasors: npt.NDArray[np.complexfloating] = field(
        default_factory=lambda: np.empty(2, dtype=np.complex64)
    )
    fgmf_acceleration_phasors: npt.NDArray[np.complexfloating] = field(
        default_factory=lambda: np.empty(2, dtype=np.complex64)
    )
