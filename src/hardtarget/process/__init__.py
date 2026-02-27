from .process import Process  # isort: off

from hardtarget.matched_filter.target_estimation_process import TargetEstimationProcess
from hardtarget.matched_filter.dpt import DPTProcess
from hardtarget.matched_filter.gmf import GMFProcess
from hardtarget.matched_filter.optimize import OptimizeProcess
from hardtarget.matched_filter.xcorr import XCorrProcess
from hardtarget.interferometry import DOAProcess
from hardtarget.types.constants import Processes, AnalysisMethod, TargetEstimationMethod, MethodLib
from typing import Optional
from typing import Callable

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
    elif method == AnalysisMethod.event_detection:
        return PROCESSES[Processes.XCORR]
    elif method == AnalysisMethod.direction_of_arrival:
        return PROCESSES[Processes.DOA]
    else:
        raise Exception(f"No available process for method: {method}, lib: {method_lib}")
