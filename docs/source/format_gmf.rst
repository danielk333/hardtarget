
..  _format_gmf:

=====================
Hardtarget GMF Format
=====================


This describes **Hardtarget_GMF**, the data format produced by :ref:`api_gmf`.


Results from :ref:`api_gmf` are stored in folders, one folder per hour. Within
each folder, there is a set of *HDF5* files, each including relevant datasets,
and named by epoch timestamp in seconds.

..  code-block:: bash

    2021-04-12T11-00-00/
    └── gmf-1618228774000000.h5
    └── gmf-1618228776000000.h5
    └── ...


Dimensions
----------

* Sample Numbers
    Dimension defined by sample numbers from experiment

    * **path**: ["sample_numbers"]
    * **dtype**: int64

* Integration Index
    Integration index within this file relative the epoch

    * **path**: ["integration_index"]
    * **dtype**: int64

* Ranges
    Matched filter ranges. 
    
    Monostatic: distance (transmitter, object) 
    
    Bistatic : distance (transmitter, object) + (object, receiver)

    * **path**: ["ranges"]
    * **dtype**: float64
    * **unit**: "m"

* Range Rates
    Matched filter range rates.

    * **path**: ["range_rates"]
    * **dtype**: float64
    * **unit**: "m/s"

* Accelerations
    Matched filter range accelerations

    * **path**: ["accelerations"]
    * **dtype**: float64
    * **unit**: "m/s^2"

* Rx Window Index
    RX window index

    * **path**: ["accelerations"]
    * **dtype**: int64


Datasets
--------

* gmf
    Generalized Matched Filter output values

    * **path**: ["gmf"]
    * **dims**: ["integration_index", "ranges"]
    * **dtype**: float32

* gmf_zero_frequency
    Range dependent noise floor (0-frequency gmf output)

    * **path**: ["gmf_zero_frequency"]
    * **dims**: ["integration_index", "ranges"]
    * **dtype**: float32

* range_rate_index
    If range_rate is reduced, contains the best range rate index for each left over axis

    * **path**: ["range_rate_index"]
    * **dims**: ["integration_index", "ranges"]
    * **dtype**: float32

* acceleration_index
    If acceleration is reduced, contains the best acceleration index for each left over axis

    * **path**: ["acceleration_index"]
    * **dims**: ["integration_index", "ranges"]
    * **dtype**: int32

* tx_power
    Measured transmitted power

    * **path**: ["tx_power"]
    * **dims**: ["integration_index",]
    * **dtype**: float32
    * **unit**: "W"

* range_peak
    Range rate at peak GMF

    * **path**: ["range_rate_peak"]
    * **dims**: ["integration_index",]
    * **dtype**: float64

* acceleration_peak
    Acceleration at peak GMF

    * **path**: ["acceleration_peak"]
    * **dims**: ["integration_index",]
    * **dtype**: float64

* gmf_peak
    Peak GMF

    * **path**: ["gmf_peak"]
    * **dims**: ["integration_index",]
    * **dtype**: float32

* rgs
    TODO [Daniel]

    * **path**: ["vector_params"]["rgs"]
    * **dims**: ["ranges",]
    * **dtype**: int32

* fvec
    TODO [Daniel]

    * **path**: ["vector_params"]["fvec"]
    * **dims**: ["ranges",]
    * **dtype**: float64
    * **unit**: "Hz"

* acceleration_phasors
    TODO [Daniel]

    * **path**: ["vector_params"]["acceleration_phasors"]
    * **dims**: ["accelerations", "range_rates"]
    * **dtype**: complex64
    * **unit**: "rad"

* rx_stencil
    TODO [Daniel]

    * **path**: ["vector_params"]["rx_stencil"]
    * **dims**: ["sample_numbers",]
    * **dtype**: bool

* tx_stencil
    TODO [Daniel]

    * **path**: ["vector_params"]["tx_stencil"]
    * **dims**: ["sample_numbers",]
    * **dtype**: bool

* rx_window_indices
    TODO [Daniel]

    * **path**: ["vector_params"]["rx_window_indices"]
    * **dims**: ["rx_window_index",]
    * **dtype**: int32

* pointing
    Pointing data for radar antenna. 
    Vector of angle measurements {'azimuth': 40.0, 'elevation': 98.0}
    * **path**: ["pointing"]
    * **dims**: ["integration_index",]
    * **dtype**: float32
    * **unit**: "deg"


