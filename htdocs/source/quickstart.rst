..  _quickstart:

===========
Quickstart
===========

This page walks through the basic steps of radar analysis using Hardtarget. 


Download Eiscat radar data
--------------------------

In this example we will use data from `Eiscat Scientific Association
<eiscatlink_>`_. Eiscat radar data may be downloaded manually from `Eiscat
portal <eiscatdownloadlink_>`_. Alternatively, the hardtarget package provides
support for script based download.

.. code-block:: bash

   (.ht) $ hardtarget download eiscat 20220408 leo_bpark_2.1u_NO uhf  --o /data


The script will create a folder ``/data/20220408_leo_bpark_2.1u_NO_uhf`` containing the downloaded data.



.. note::

    Download from Eiscat is geo-restricted to IP-adresses in Scandinavia.
    Location-based download also uses a non-standard port number, which may be
    blocked by agressive firewalls.


Convert Eiscat radar data
--------------------------

..  _drflink: https://pypi.org/project/digital-rf/
..  _eiscatlink: https://eiscat.se/
..  _eiscatdownloadlink: https://portal.eiscat.se/

Hardtarget expects radar data in a specific format known as :ref:`drf`. This is
essentially the `digital-rf <drflink_>`_ format, yet extended with an extra metadata
file. Eiscat radar data may be converted to :ref:`drf` using the following script.

.. code-block:: bash

   (.ht) $ hardtarget convert eiscat /data/leo_bpark_2.1u_NO20220408/leo_bpark_2.1u_NO@uhf -o /data/drf






