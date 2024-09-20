.. Hardtarget documentation master file, created by
   sphinx-quickstart on Thu Feb 22 11:50:50 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

====================================================
Welcome to Hardtarget's documentation!
====================================================


.. note::

   This project is under development.


:Release: |release|
:Date: |today|


Introduction
"""""""""""""

The Hardtarget library contains a set of tools used to detect and characterize
coherent echoes in raw (complex amplitude-level) data from a radar.  At the
heart of the library are routines for implementing a generalised match filter
(GMF) which is used to detect targets with a quadratic phase behaviour.

The library is used for detections of meteors in atmospheric radar data, and
for detecting resident space objects (RSOs) such as satellites and space debris
with ground-based radars.

The library operates on level 2 beamformed data streams that are stored on disk
in `Digital RF <https://github.com/MITHaystack/digital_rf>`_ format.
The program assumes that there is a complex baseband
recording of the transmit waveform and the received complex voltage containing
the radar echoes. The program can be adapted to various different radars and
radar experiment parameters. This is done by editing a configuration file. An
example configuration file is shown below. The example file includes comments
that describe each parameter.

If there is an available `CUDA compiler <https://developer.nvidia.com/cuda-zone>`_
in the environment, then both a CUDA library and a C library will be compiled.
If no CUDA compiler is found, then only the C code will be compiled.

There is also an end-to-end test, which includes a raw voltage simulator that
creates a dataset and analyzes it. This can be used to test the performance of
the analysis program and to validate the results. This also includes use of an
additional refinement step that is used to refine the detected targets:


Quick install
"""""""""""""

To install, ensure dependencies `gcc` and `libfftw3-dev` are installed for the 
C implementations and/or `cuda` for the GPU implementations. Then install with

.. code-block:: bash
   
   pip install hardtarget

or the nightly build

.. code-block:: bash

   git clone --branch develop git@github.com:danielk333/hardtarget.git
   cd hardtarget
   pip install .

Read the CLI description

.. code-block:: bash

   hardtarget -h


History
"""""""
The hardtarget library has grown out of research and developments by several
scientists in diverse fields, over some time.

EISCAT/IRF observations of meteors since 1994 or thereabouts.

First EISCAT observations of space debris (2000-2001)

ESA space debris projects SGO, IRF, UiT, NORCE (2000-2024)


Getting Help
""""""""""""
If you have questions about using Hardtarget please open a `GitHub Issue
<https://github.com/danielk333/hardtarget/issues>`_ or email the `Hardtarget
developers <mailto:daniel.kastinen@irf.se>`_.


Acknowledgements
"""""""""""""""""
SGO, IRF, UiT, NORCE, IKS, ESA, EISCAT


* ESOC Contract No. 13945/99/D/CD, 2002.
* ESOC Contract No. 16646/02/D/HK(CS), 2005.

TODO [Daniel]

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. Hidden TOCs


.. toctree::
   :caption: Usage
   :maxdepth: 2
   :hidden:

   install
   quickstart
   cli

.. toctree::
   :caption: API
   :maxdepth: 2
   :hidden:

   api_eiscat
   api_gmf

.. toctree::
   :caption: Concepts
   :maxdepth: 2
   :hidden:

   concept_gmf

.. toctree::
   :caption: Reference
   :maxdepth: 2
   :hidden:

   params_experiment
   params_processing
   format_drf
   format_gmf



