

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
will contribute with randomly varying pulse, and will average to zero.


TODO [Daniel, Tom]
