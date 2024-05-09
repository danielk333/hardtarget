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

   (.ht) $ hardtarget -v download eiscat 20220408 leo_bpark_2.1u_NO uhf /data --progress


The script will download a zip archive to the `/data` folder. After unzip and
removal of the zip file, two folders are available.

.. code-block:: bash

   (.ht) $ tree /data -L 2
   (.ht) $ /data/
   (.ht) $ ├── leo_bpark_2.1u_NO@uhf
   (.ht) $ │   ├── 20220408_08
   (.ht) $ │   └── 20220408_09
   (.ht) $ └── leo_bpark_2.1u_NO@uhf_information
   (.ht) $     └── 20220408


.. note::

   The download link used in the script is geo-restricted to IP-addresses in
   Scandinavia. Location-based download also uses a non-standard port number,
   and may be blocked by an agressive firewall. For data access outside
   Scandinavia, see `Eiscat Scientific Association <eiscatlink_>`_. 


Convert Eiscat radar data
--------------------------

..  _drflink: https://pypi.org/project/digital-rf/
..  _eiscatlink: https://eiscat.se/
..  _eiscatdownloadlink: https://portal.eiscat.se/

Hardtarget expects radar data in a specific format known as :ref:`format_drf`. This is
essentially the `digital-rf <drflink_>`_ format, yet extended with an extra metadata
file. Eiscat radar data may be converted to :ref:`format_drf` using the following script.

.. code-block:: bash

   (.ht) $ hardtarget -v convert eiscat /data/leo_bpark_2.1u_NO@uhf /data/drf --progress


The script will make the result available in the given folder. The file `metadata.ini` includes
meta-data about the radar experiment from which the :ref:`format_drf` product was derived. 

.. code-block:: bash

   (.ht) $ tree /data/leo_bpark_2.1u_NO@uhf_drf -L 2
   (.ht) $ /data/leo_bpark_2.1u_NO@uhf_drf
   (.ht) $ ├── metadata.ini
   (.ht) $ └── uhf
   (.ht) $     ├── 2022-04-08T08-00-00
   (.ht) $     ├── 2022-04-08T09-00-00
   (.ht) $     └── drf_properties.h5

Meta-data may also be obtained using the inspect script.

.. code-block:: bash

   (.ht) $ hardtarget -v inspect drf /data/drf
   (.ht) $ ...
   (.ht) $ ... ('start', '2022-04-08 08:32:00')
   (.ht) $ ... ('end', '2022-04-08 09:01:39.999999')
   (.ht) $ ...



Analyze DRF Data
--------------------------

Run GMF analysis using the following script. The config file `cfg.ini` describes
processing parameters.

.. code-block:: bash

   (.ht) $ hardtarget -v analyze /data/drf uhf --config cfg.ini --progress -o /data/gmf

The script will make the result available in the given folder. 

.. code-block:: bash

   (.ht) $ tree /data/gmf -L 2
   (.ht) $ 20210412/gmf
   (.ht) $ └── 2021-04-12T12-00-00
   (.ht) $     ├── gmf-1618229740000000.h5
   (.ht) $     ├── .......................
   (.ht) $     └── gmf-1618229768000000.h5


For large products, analysis may take some time. It is also possible to limit
the conversion to a time range.

.. code-block:: bash

   (.ht) $ hardtarget -v analyze /data/drf uhf --config cfg.ini --progress -o /data/gmf -s 2022-04-08T08:40:00 -e 2022-04-08T08:40:30



Plot GMF Data
--------------------------

GMF data may be presented using the Hardtarget plotting tool.

.. code-block:: bash

   (.ht) $ hardtarget plot gmf /data/gmf -s 2022-04-08T08:40:00 -e 2022-04-08T08:40:30






