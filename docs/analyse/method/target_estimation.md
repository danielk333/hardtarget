# Target estimation

Given that the targets phase function can be completely described by at most a second order
polynomial, there are at least three ways to process a matched filter output:

1. Directly compute the output as a function of all three polynomial coefficients
2. Perform one FFT over one of the components, and directly compensate for the other
2. Perform two FFT's over two of the components, and directly compensate for the other

We here implement all three possible variants, where option 1) is named echo search (??? fix),
options 2) is called GMF, and option 3) is called DPT.

Technically we also have a version of 1) that is usable as a refinement method, named "optimization", which
basically does a gradient ascent or similar optimization on the full matched filter function to
refine an initial result found by the other methods (which usually use grid-evaluation to find peaks
in the matched filter).


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
