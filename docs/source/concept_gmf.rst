

..  _concept_gmf:

==========================
Generalized Matched Filter
==========================


This documents context and key concepts in **GMF** processing


Coherent integration
----------

Target echoes that are too weak to detect in the raw signal but scatter at a
predictable phase is termed **coherent**, and can be brought out by a technique
called **coherent integration**.

From a fundamental perspective, it is a filtering operation.  Echoes (complex
amplitude data) from a collection of pulses are summed, taking into account the
expected phase behavior.  A coherent target which is expected to be stationary
with respect to the radar should scatter with exactly the same phase from one
pulse to the next, and its contribution to the combined signal increases
linerarly with the number of pulses added.  Noise-like contributions, however,
will contribute with randomly varying phase, and will average to zero.

The SNR increase is sometimes referred to as the **gain** of the filter.


Matched Filter
--------

When a coherent target's position relative to the radar is not stationary, but
expected to follow some predictable trajectory, the phase contribution due to
the motion can be compensated for before the complex amplitude data are
summed. This is called a **matched filter** operation, the phase function in
the filter is **matched** to the expected phase history of the target. If the
phase is not correctly predicted, the gain of the filter will be reduced, and
a target may evade detection.

The Generalized Matched Filter
--------

In our case, the parameters that are used to predict the phase of a target are
not known in advance --- in fact, these are exactly the parameters we want to
estimate for the purpose of orbit determination.

The target will not be detectable at all unless we have the correct matched
filter, so it seems the only option is to try **every feasible** matched
filter, and pick the one that gives the highest gain, an approach that is
often referred to as a **filter bank**.

Implementations
--------

The brute-force filter bank operation has an algorithmic complexity which is
proportional to the product of the number of grid search points for each
parameter. For a quadratic model (equivalent to constant radial acceleration),
three parameters must be correctly guessed, and the complexity is (broadly
speaking) cubic in the typical linear dimension of the search space. The GMF
is then easily the most computationally expensive step of the hard target
detection process, and optimizations must be found to produce results without
unacceptable delays.


The Fast GMF, of FGMF
++++++++

The processing chain developed at SGO over several years of EISCAT space
debris observations is centered on a GMF implementation where a number of
clever tricks are exploited to bring down the processing time. One important
improvement was to realise that summing over the phase contribution caused by
a linear motion is equivalent to a DFT, which is implemented very efficiently
by means of the FFT.  This means that one axis of the multi-dimensional filter
bank is not necessary, greatly reducing the size of the problem..

The resulting GMF approach is referred to as the **Fast GMF**, or FGMF.
Implementations of FGMF exist in numpy, numba(?) and C.


The Discrete Polynomial-phase Transform, or DPT
++++++++

Given that the detection of very weak hard target echoes has been the purpose
of radars since they were first invented during WWII, it should not come as a
surprise that clever techniques had been invented already.  In the radar
literature, the kind of signal we hope to detect is referred to as a
**polynomial-phase signal**, and the problem we try to solve is a **polynomial
phase estimation** problem. We found a paper :cite:`Peleg1995` which describes
a very efficient algorithm for estimating polynomial phase coefficients for
the strongest target echo in a signal, one at a time, where the cost of
extracting a single coefficient is on the order of an FFT of size equal to the
signal's length in samples.



TODO [Daniel, Tom]
