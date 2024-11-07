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
    Pulse offset to use when selecting transmitt pulse for coherent integration, used for finding range-aliased echoes. Value of 1 would mean first range alias, 2 second alias, ect.

* min_range_gate *(int)*
    Minimum range gate to process relative to the start of transmission for each pulse (each range gate is one receiver sample).

* max_range_gate *(int)*
    Maximum range gate to process relative to the start of transmission for each pulse (each range gate is one receiver sample).

* range_gate_step *(int)*
    The step in range-gates to use when processing, if baud-length is longer than the receiver sampling time this can be increased to sacrifice range-resolution for processing speed.

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

* dpt_ipp_delay_paramater *(int)*
    How many IPPs to use for estimating the numerical derivative of the phase of the coherent signal if using the polynomial phase transform method. Also used by the Fast GMF method to calculate the acceleration resolution to sample. A larger number means smaller accelerations can be determined at higher resolution.
