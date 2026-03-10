"""General Matched Filter, or GMF type specifics"""

from dataclasses import dataclass

from hardtarget.target_estimation.types import ExtendedTargetEstimationProParams, TargetEstimationCfgParams


@dataclass(frozen=True)
class GMFCfgParams(TargetEstimationCfgParams):
    """
    GMF specific configuration parameters, can be configured under the gmf section in the .ini file.
    """

    acceleration_steps: int = 1


GMFProParams = ExtendedTargetEstimationProParams
