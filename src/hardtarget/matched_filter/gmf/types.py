"""General Matched Filter, or GMF type specifics"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from hardtarget.matched_filter.types import ExtendedTargetEstimationProParams, TargetEstimationCfgParams


@dataclass(frozen=True)
class GMFCfgParams(TargetEstimationCfgParams):
    """
    GMF specific configuration parameters, can be configured under the gmf section in the .ini file

    Args:
        acceleration_steps: Number of acceleration steps between max and min acceleration.
    """

    acceleration_steps: int = 1


GMFProParams = ExtendedTargetEstimationProParams
