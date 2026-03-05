"""General Matched Filter, or GMF type specifics"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from hardtarget.types.types import CfgParams, ProParams


@dataclass(frozen=True)
class GMFCfgParams(CfgParams):
    """
    GMF specific configuration parameters, can be configured under the gmf section in the .ini file

    Args:
        acceleration_steps: Number of acceleration steps between max and min acceleration.
    """

    acceleration_steps: int = 1


@dataclass(frozen=True)
class GMFProParams(ProParams):
    """
    GMF specific process parameters

    Args: TODO
        inds_accelerations:
        accelerations:
        acceleration_phasors:
        fgmf_acceleration_phasors:
    """

    inds_accelerations: npt.NDArray[np.int32] = field(default_factory=lambda: np.empty(2, dtype=np.int32))
    accelerations: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(2, dtype=np.float64))
    acceleration_phasors: npt.NDArray[np.complexfloating] = field(
        default_factory=lambda: np.empty(2, dtype=np.complex64)
    )
    fgmf_acceleration_phasors: npt.NDArray[np.complexfloating] = field(
        default_factory=lambda: np.empty(2, dtype=np.complex64)
    )
