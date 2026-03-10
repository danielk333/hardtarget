# Optimization Process

---

This describes functionality related to the **Optimization Process**. Optimization is based on the results
from the target estimation analysis. Thus the normal process would be to first run a target estimation
analysis and then a optimization, the optimization configuration requires a path to the target estimation
output. What the optimization does is running a maximum likelihood method with the target estimation maximas
as seed values. The different supported libraries for the optimization are:

::: hardtarget.constants.OptimizationMethod

## Analysis

---

To simplify or rather clarify the usage of [analyse](../api_analyse.md) for target estimation optimization a [wrapper](../../reference/hardtarget/analyse.md#hardtarget.analyse.optimize) is available that only supports **optimization**. More information about the running process can be found under [Process](#process)

::: hardtarget.analyse.optimize

## Process

---

::: hardtarget.optimization.OptimizeProcess

## Type specifics

---

::: hardtarget.optimization.types
