from .process import Process  # isort: off

from hardtarget.target_estimation import DPTProcess, GMFProcess, TargetEstimationProcess
from hardtarget.optimization import OptimizeProcess
from hardtarget.echo_search import EchoSearchProcess
from hardtarget.interferometry import DOAProcess
from hardtarget.constants import AnalysisMethod, TargetEstimationMethod, MethodLib
from typing import Optional

# ---- Processes ----
PROCESSES: dict[AnalysisMethod, type[Process] | dict[MethodLib, type[Process]]] = {
    AnalysisMethod.target_estimation: {
        TargetEstimationMethod.fgmf: GMFProcess,
        TargetEstimationMethod.fdpt: DPTProcess,
    },
    OptimizeProcess.method: OptimizeProcess,  # type: ignore[has-type]
    EchoSearchProcess.method: EchoSearchProcess,  # type: ignore[has-type]
    DOAProcess.method: DOAProcess,  # type: ignore[has-type]
}


def get_analysis_process(method: AnalysisMethod, method_lib: Optional[MethodLib] = None) -> type[Process]:
    """
    Get process for the intended method and specific method_lib if requested

    Args:
        method: Analysis method
        method_lib (optional): Specific library for the intented method
    Returns:
        Process compatible with the method

    """

    try:
        process = PROCESSES[method]

        if isinstance(process, dict):
            if method_lib:
                return process[method_lib]
            else:
                return GMFProcess  # TODO: Better handling of default value
        else:
            return process
    except KeyError:
        raise KeyError(f"No available process for method: {method}")
