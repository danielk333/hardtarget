..  _cli:

======================
CLI
======================


The Command Line Interface of Hardtarget is exposed under a single executable **hardtarget**.

..  code-block:: bash

    (.ht) hardtarget -h


The **hardtarget** executable provides access to 6 tools, each with subcommands.
The option **-v** turns on verbose mode.


Download
--------

Use the **download** tool to download radar data. Currently, **download**
functionality is only supported for Eiscat data.

..  code-block:: bash

    (.ht) $ hardtarget download -h
    (.ht) $ hardtarget download eiscat -h    
    (.ht) $ hardtarget -v download eiscat 20220408 leo_bpark_2.1u_NO uhf /data --progress

Convert
-------

Use the **convert** tool to convert radar data to :ref:`format_drf`. Currently, **convert**
functionality is only supported for Eiscat data.

.. code-block:: bash

    (.ht) $ hardtarget convert -h
    (.ht) $ hardtarget convert eiscat -h    
    (.ht) $ hardtarget -v convert eiscat /data/leo_bpark_2.1u_NO@uhf /data/drf --progress


Inspect
-------

Use the **inspect** tool to inspect a data product of given type. The tool
prints key properties and metadata to the screen. Currently, **inspect**
functionality is supported for **drf** (:ref:`format_drf`) and **gmf**
(:ref:`format_gmf`). 

.. code-block:: bash

    (.ht) $ hardtarget inspect -h
    (.ht) $ hardtarget inspect drf -h    
    (.ht) $ hardtarget inspect gmf -h    
    (.ht) $ hardtarget inspect drf /data/drf
    (.ht) $ hardtarget inspect gmf /data/gmf

Plot
----

Use the **plot** tool to plot a data product of given type. Currently, **plot**
functionality is supported for **drf** (:ref:`format_drf`) and **gmf**
(:ref:`format_gmf`). 

.. code-block:: bash

    (.ht) $ hardtarget plot -h
    (.ht) $ hardtarget plot drf -h    
    (.ht) $ hardtarget plot gmf -h    
    (.ht) $ hardtarget plot drf /data/drf -s 2022-04-08T08:40:00 -e 2022-04-08T08:40:30
    (.ht) $ hardtarget plot gmf /data/gmf -s 2022-04-08T08:40:00 -e 2022-04-08T08:40:30


.. note::
    
    Plot only small a small time range. Use the **inspect** tool to figure out time ranges.


Check
-----

Use the **check** tool to perform certain checks. Currently, checks are included for
cuda support and range gate configuration.

.. code-block:: bash

    (.ht) $ hardtarget check -h
    (.ht) $ hardtarget check cuda
    (.ht) $ hardtarget check range-gates -h



Analyze
-------

Use the **analyze** tool to launch gmf processing. 

.. code-block:: bash

   (.ht) $ hardtarget -v analyze /data/drf uhf --config cfg.ini --progress -o /data/gmf








