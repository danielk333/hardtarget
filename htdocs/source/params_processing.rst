..  _gmfprocessingparams:

============================
GMF Processing Parameters
============================

This describes processing paramaters used to configure :ref:`api_gmf`.

.. code-block:: ini
    
    [signal-processing]
        n_ipp=10
        ipp_offset=0
        min_range_gate=6800
        max_range_gate=7280
        min_acceleration=-300.0
        max_acceleration=300.0
        range_gate_step=1
        frequency_decimation=16
        num_cohints_per_file=10
        node_gpus=1
        dpt_ipp_delay_parameter=5


* n_ipp *(int)*
    Number or interpulse periods to coherently integrate

* ipp_offset *(int)*
    TODO [Daniel]

* min_range_gate *(int)*
    TODO [Daniel]

* max_range_gate *(int)*
    TODO [Daniel]

* range_gate_step *(int)*
    TODO [Daniel]

* frequency_decimation *(str)*
    How many samples to step in range in the first stage of the course search.

* clutter_length *(int)*
    How many samples to remove due to ground clutter

* min_acceleration *(float)*
    Minimum acceleration to search for (m/s^2)

* max_acceleration *(float)*
    Maximum acceleration to search for (m/s^2)

* acceleration_resolution *(float)*
    How many radians of resolution for the acceleration search

* num_cohints_per_file *(int)*
    
    How many coherent integration periods to include in one output file.
    Smaller means that lower latency can be achieved.

* reduce_range *(bool)*
    TODO [Daniel]

* reduce_range_rate *(bool)*
    TODO [Daniel]

* reduce_acceleration *(bool)*
    TODO [Daniel]

* dpt_ipp_delay_paramater *(int)*
    TODO [Daniel]
