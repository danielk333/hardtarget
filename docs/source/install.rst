=======
Install
=======

.. note::
   Hardtarget instuallation in only supported on Linux.


.. contents:: Table of Contents
   :local:



Prerequisites
-------------

Hardtarget depends on *gcc* and *libfftw3-dev*.

.. code-block:: bash

   sudo apt install gcc libfftw3-dev


It is recommended to install Hardtarget in a virtual environment, e.g. *virtualenv* or *conda*.

.. code-block:: bash

   $ python3.X -m venv .ht
   $ source .ht/bin/activate
   (.ht) $ pip install --upgrade pip


Installation from PyPi
----------------------

.. warning::

   Installation from PyPi not supported yet!

.. code-block:: bash

   (.ht) $ pip install hardtarget


Installation from Git
--------------------------------------

Basic install.

.. code-block:: bash

   $ git clone git@github.com:danielk333/hardtarget.git
   $ cd hardtarget
   (.ht) $ pip install .[plotting]

Nightly build.

.. code-block:: bash

   $ git clone --branch develop git@github.com:danielk333/hardtarget.git
   $ cd hardtarget
   (.ht) $ pip install .


Full developer install.

.. code-block:: bash

   (.ht) $ pip install -e .[mpi,develop,plotting,profiling]
