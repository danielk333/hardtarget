# Target estimation

---

Target estimation is done through the Generalized Matched Filter (GMF). The GMF for a certain
signal model is proportional to the Likelihood function for that signal models
parameters given a measured signal. It is called a matched filter because signal
power is transmitted trough the filter (i.e. the function) where the model
matches the recorded signal. If the measured signal follows the signal model,
the peak of the GMF appears at the location of the parameters of the true signal
perturbed by noise. For multiple targets it is usually possible to find multiple
peaks in the GMF each corresponding the a unique target.

The supported ways to calculate or approximate the global maximum of the GMF are:
::: hardtarget.constants.TargetEstimationMethod

Because of the difference between these two methods they have been devided into two processes, the [GMFProcess](#gmf-process) and the [DPTProcess](#dpt-process), both derived from the [TargetEstimationProcess](../../reference/hardtarget/target_estimation/target_estimation_process.md#hardtarget.target_estimation.target_estimation_process.TargetEstimationProcess)

## Analysis

---

To simplify or rather clarify the usage of [analyse](../api_analyse.md) for target estimation a [wrapper](../../reference/hardtarget/analyse.md#hardtarget.analyse.target_estimation) is available that only supports **Target Estimation**. More information about the running process can be found under [GMF Process](#gmf-process) and the [DPTProcess](#dpt-process).

::: hardtarget.analyse.target_estimation

## GMF process

This describes functionality related to the **General Match Filter Process**.

::: hardtarget.target_estimation.gmf.GMFProcess

### Type specifics

---

::: hardtarget.target_estimation.gmf.types

## DPT process

This describes functionality related to the **Discrete Polynomial-phase Transform**.

::: hardtarget.target_estimation.dpt.DPTProcess

### Type specifics

---

::: hardtarget.target_estimation.dpt.types
