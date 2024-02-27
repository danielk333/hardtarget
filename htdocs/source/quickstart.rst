..  _quickstart:

===========
Quickstart
===========

These are the basic steps working with radar data using Hardtarget. 


Download Eiscat radar data
--------------------------

In this example we will use data from `Eiscat Scientific Association
<eiscatlink_>`_. Eiscat radar data may be downloaded manually from `Eiscat
portal <eiscatdownloadlink_>`_. Alternatively, the hardtarget package provides
support for script based download.

.. code-block:: bash

   (.ht) $ hardtarget -v download eiscat 20220408 leo_bpark_2.1u_NO uhf --progress -o /data


The script will make the result available in the `/data` folder. Using the
`--update` option, multiple downloads can be placed in the same folder hierarchy.

.. code-block:: bash

    (.ht) $ tree /data -L 2
    (.ht) $ /data/
    (.ht) $ ├── leo_bpark_2.1u_NO@uhf
    (.ht) $ │   ├── 20220408_08
    (.ht) $ │   └── 20220408_09
    (.ht) $ └── leo_bpark_2.1u_NO@uhf_information
    (.ht) $     └── 20220408


.. note::

    Download from Eiscat is geo-restricted to IP-addresses in Scandinavia.
    Location-based download also uses a non-standard port number, and may be
    blocked by an agressive firewall.


Convert Eiscat radar data
--------------------------

..  _drflink: https://pypi.org/project/digital-rf/
..  _eiscatlink: https://eiscat.se/
..  _eiscatdownloadlink: https://portal.eiscat.se/

Hardtarget expects radar data in a specific format known as :ref:`drf`. This is
essentially the `digital-rf <drflink_>`_ format, yet extended with an extra metadata
file. Eiscat radar data may be converted to :ref:`drf` using the following script.

.. code-block:: bash

   (.ht) $ hardtarget -v convert eiscat /data/leo_bpark_2.1u_NO@uhf --progress -o /data/drf


The script will make the result available in the given folder.

.. code-block:: bash

   (.ht) $ /data/drf
   (.ht) $ ├── metadata.ini
   (.ht) $ └── uhf
   (.ht) $     ├── 2022-04-08T08-00-00
   (.ht) $     ├── 2022-04-08T09-00-00
   (.ht) $     └── drf_properties.h5


Analyze DRF Data
--------------------------

.. code-block:: bash

   (.ht) $ hardtarget -v analyze 20210412/drf uhf --config ~/Dev/Git/hardtarget/examples/cfg/test.ini --progress -o 20210412/gmf -s "2021-04-12T12:15:40" -e "2021-04-12T12:16:10"  -m fgmf -i numpy


Plot GMF Data
--------------------------

hardtarget plot gmf 20210412/gmf3 -s "2021-04-12T12:15:40" -e "2021-04-12T12:16:10"






