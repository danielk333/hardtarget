..  _drf:

==============
Hardtarget DRF
==============



The **Hardtarget_DRF** data format is used as input to GMF Analysis.

 



..  _drflink: https://pypi.org/project/digital-rf/

The **Hargtarget_DRF** format is an instance of the **Digital_RF** format for
reading and writing radio frequency data. **Digital_RF** is defined by the
Python module `digital-rf <drflink_>`_, and is based on the **HDF5** format, a
self-documenting format intended for efficient data storage and random access
for data processing. As **Digital_RF** does not allow arbitrary meta
information, **Hardtarget_DRF** additionally specifies an extra metadata file
**metadata.ini** at the top level of the folder tree. The metadata file defines
key experiment parameters associatated with the experiment data.




..  Block comment:

    Should DRF metadata.inf include start,end?

    See Radar-experiment-parameters.


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



Experiment Parameters
---------------------


.. code-block:: ini

    [Experiment]
    name = leo_bpark
    version = 2.1u
    sample_rate = 1000000
    ipp = 20000
    file_secs = 12.8
    tx_pulse_length = 1920.0
    doppler_sign = -1.0
    round_trip_range = false
    rx_channel = uhf
    rx_start = 2.0
    rx_end = 19997.0
    tx_channel = uhf
    tx_start = 82.0
    tx_end = 2002.0
    cal_on = 19900.0
    cal_off = 19997.0
    radar_frequency = 929.6


* name *(str)*
    Experiment identifier

* version *(str)*
    Experiment version number

* sample_rate *(int)*
    Number of samples per second

* ipp *(int)*
    Inter-pulse period length counted in samples

* file_secs *(float)*
    Seconds worth of data in each file?

* tx_pulse_length *(float)*
    Transmit-pulse length in samples

* doppler_sign *(float)*
    Doppler shift convention [1.0|-1.0]. (Positive means towards the radar?)

* round_trip_range *(boolean)*
    Monostatic (false) or bi-static (true)

* tx_channel *(string)*
    Name of channel with sampled transmit waveform, e.g. ("uhf")

* tx_start *(float)*
    Index of first transmit pulse in samples

* tx_end *(float)*
    Index of last transmit pulse in samples (or first "empty sample after end?")

* rx_channel *(string)*
    Name of channel with sampled echos, e.g. ("uhf")

* rx_start *(float)*
    Index of first receive pulse in samples

* rx_end *(float)*
    Index of last receive pulse in samples (or first "empty sample after end?")

* cal_on *(float)*
    ?

* cal_off *(float)*
    ?

* radar_frequency *(float)*
    Exact frequency of radar (unit?)





