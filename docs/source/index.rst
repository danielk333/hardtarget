hardtarget
==========

:Release: |release|
:Date: |today|


Hardtarget is ... TODO



Getting started
-----------------

To install

.. code-block:: bash

   pip install hardtarget

or the nightly build

.. code-block:: bash

   git clone --branch develop git@github.com:danielk333/hardtarget.git
   cd hardtarget
   pip install .


Tutorials
---------

.. toctree::
   :maxdepth: 2

   notebooks/getting_started


Examples
---------

Example gallery of the different modular functionality of the toolbox.

.. toctree::
   :maxdepth: 2

   autogallery/index


API Reference
==============

The typical data-flow of processing

.. image:: static/hardtarget_gmf_arrays.svg
  :alt: The data flow trough different array indexing schemas
  :class: light-theme-image

TODO: a flow-diagram with DPT acceleration

.. irf_autopackages:: package
   :template: autosummary/module.rst
   :toctree: autosummary
   :exclude: hardtarget.version

   hardtarget


Extensions
----------

TODO: make this work, package documentation:
- https://sphinx-c-autodoc.readthedocs.io/en/latest/configuration.html


GMF C implementation

.. autocfunction:: gmf.c::gmf

TODO: this does not work because it does not find the symbol???
maybe because of the "extern C"

.. autocfunction:: gmfgpu.cu::gmf


Developing
==========

Please refer to the style and contribution guidelines documented in the
`IRF Software Contribution Guide <https://danielk.developer.irf.se/software_contribution_guide/>`_.
Generally external code-contributions are made trough a "Fork-and-pull"
workflow, while internal contributions follow the branching strategy outlined
in the contribution guide.

Docs
~~~~

To make the docs, use the `Makefile` by running

.. code-block:: bash

   make html


Notebooks
~~~~~~~~~

To develop notebooks for documentation in Jupyter-lab, install the following

.. code-block:: bash

   pip install notebook jupytext

Then run notebooks in the appropriate folder `docs/source/notebooks` using `jupyter-notebook` and
pair the new notebook with a MyST file.

For more information on how to pair notebooks in order to have persistent plain-text versions,
see the `jupytext docs <https://jupytext.readthedocs.io/en/latest/paired-notebooks.html>`_.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
