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

What Hardtarget is, and what is it used for.

TODO [Daniel]

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
Hardtarget has grown out of research by ... in diverse fields,
over some time.

TODO [Daniel]

Getting Help
""""""""""""
If you have questions about using Hardtarget please open a `GitHub Issue
<https://github.com/danielk333/hardtarget/issues>`_ or email the `Hardtarget
developers <mailto:daniel.kastinen@irf.se>`_.


Acknowledgements
"""""""""""""""""
IRF, NORCE, IKS, ESA, EISCAT

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
   develop

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



