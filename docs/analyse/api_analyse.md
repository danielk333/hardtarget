# Analysis

---

This describes functionality related to the analysis tool. The analyse function is the main path for any analysis to happen, it supports multiple methods. The ones supported as of now are:

::: hardtarget.constants.AnalysisMethod

And each of these methods have (at times) different implementations of the method, these can be found under the processes.

The analysis is done through extracting the available data from the datasource,
decide what process is suitable for the given task, split the workload and then run the analysis.

## Analyse

---

The analyse function works for any given method (of the available ones). And is the main entrypoint for any analysis.

::: hardtarget.analyse.analyse
