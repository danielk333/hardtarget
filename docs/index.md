# Welcome to Hardtarget's documentation!

!!! note
    This project is under development.

---
## Introduction

The Hardtarget library contains a set of tools used to detect and characterize
coherent echoes in raw (complex amplitude-level) data from a radar.  At the
heart of the library are routines for implementing a generalised match filter
(MF) which is used to detect targets with a quadratic phase behaviour.

The library is used for detections of meteors in atmospheric radar data, and
for detecting resident space objects (RSOs) such as satellites and space debris
with ground-based radars.

The library operates on level 2 beamformed data streams that are stored on disk
in any format supported by the [radardef](https://pypi.org/project/radardef/) library.
The program assumes that there is a complex baseband
recording of the transmit waveform and the received complex voltage containing
the radar echoes. The program can be adapted to various different radars and
radar experiment parameters. This is done by editing a configuration file, as seen in
[config parameters](analyse/config_parameters.md).

If there is an available [CUDA compiler](https://developer.nvidia.com/cuda-zone)
in the environment, then both a CUDA library and a C library will be compiled.
If no CUDA compiler is found, then only the C code will be compiled.

There is also an end-to-end test, which includes a raw voltage simulator that
creates a dataset and analyses it. This can be used to test the performance of
the analysis program and to validate the results. This also includes use of an
additional refinement step that is used to refine the detected targets:


---
## Quick install

To install, ensure dependencies `gcc`, `libfftw3-dev`, and `libhdf5-dev` are
installed for the C implementations and/or `cuda` for the GPU implementations.
Then install with

```bash
   pip install hardtarget
```
or the nightly build

```bash
   git clone --branch develop git@github.com:danielk333/hardtarget.git
   cd hardtarget
   pip install .
```

Read the CLI description

```bash
   hardtarget -h
```

---
## History

The hardtarget library has grown out of research and developments by several
scientists in diverse fields, over some time.

EISCAT/IRF observations of meteors since 1994 or thereabouts.

First EISCAT observations of space debris (2000-2001)

ESA space debris projects SGO, IRF, UiT, NORCE (2000-2024)


---
## Getting Help

If you have questions about using Hardtarget please open a [GitHub Issue](https://github.com/danielk333/hardtarget/issues) or email the [Hardtarget
developers](mailto:daniel.kastinen@irf.se)


---
## Acknowledgements

SGO, IRF, UiT, NORCE, IKS, ESA, EISCAT


* ESOC Contract No. 13945/99/D/CD, 2002.
* ESOC Contract No. 16646/02/D/HK(CS), 2005.