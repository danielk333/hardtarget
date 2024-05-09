..  _format_drf:

=====================
Hardtarget DRF Format
=====================


This describes **Hardtarget_DRF**, the data format used by :ref:`api_gmf`.

 

..  _drflink: https://pypi.org/project/digital-rf/

The **Hargtarget_DRF** format is an instance of the **Digital_RF** format for
reading and writing radio frequency data. **Digital_RF** is defined by the
Python module `digital-rf <drflink_>`_, and is based on the **HDF5** format, a
self-documenting format intended for efficient data storage and random access
for data processing. As **Digital_RF** does not allow arbitrary meta
information, **Hardtarget_DRF** additionally specifies an extra metadata file
**metadata.ini** at the top level of the folder tree. The metadata file defines
key :ref:`radarexperimentparams` experiment parameters associatated with the
experiment data.


..  code-block:: bash

    drf
    ├── metadata.ini
    └── uhf
        ├── 2021-04-12T11-00-00
        │   ├── rf@1618228761.000.h5
        │   ├── rf@1618228762.000.h5
        │   ├── rf@1618228763.000.h5
        │   ├── rf@1618228764.000.h5
        │   ├── ...
        └── drf_properties.h5





