from .process import Process  # isort: off

from hardtarget.target_estimation import TargetEstimationProcess, DPTProcess, GMFProcess
from hardtarget.optimization import OptimizeProcess
from hardtarget.echo_search import XCorrProcess
from hardtarget.interferometry import DOAProcess
from hardtarget.constants import Processes, AnalysisMethod, TargetEstimationMethod, MethodLib
from typing import Optional

# ---- Processes ----
PROCESSES: dict[Processes, type[Process]] = {
    Processes.GMF: GMFProcess,
    Processes.DPT: DPTProcess,
    Processes.Optimization: OptimizeProcess,
    Processes.XCORR: XCorrProcess,
    Processes.DOA: DOAProcess,
}


def get_analysis_process(method: AnalysisMethod, method_lib: Optional[MethodLib] = None) -> type[Process]:

    if method == AnalysisMethod.target_estimation:
        if method_lib:
            if method_lib == TargetEstimationMethod.fdpt:
                return PROCESSES[Processes.DPT]
            else:
                return PROCESSES[Processes.GMF]
        else:
            return PROCESSES[Processes.GMF]
    elif method == AnalysisMethod.optimize:
        return PROCESSES[Processes.Optimization]
    elif method == AnalysisMethod.echo_search:
        return PROCESSES[Processes.XCORR]
    elif method == AnalysisMethod.direction_of_arrival:
        return PROCESSES[Processes.DOA]
    else:
        raise Exception(f"No available process for method: {method}, lib: {method_lib}")
