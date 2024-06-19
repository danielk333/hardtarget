Concepts
========

..  Block comment:

    The library operates on level 2 beamformed data streams that are stored on disk
    in Digital RF format. The program assumes that there is a complex baseband
    recording of the transmit waveform and the received complex voltage containing
    the radar echoes. The program can be adapted to various different radars and
    radar experiment parameters. This is done by editing a configuration file. An
    example configuration file is shown below. The example file includes comments
    that describe each parameter.

    If there is an available cuda compiler in the environment, then both a CUDA
    library and a C library will be compiled. If no CUDA compiler is found, then
    only the C code will be compiled.

    There is also an end-to-end test, which includes a raw voltage simulator that
    creates a dataset and analyzes it. This can be used to test the performance of
    the analysis program and to validate the results. This also includes use of an
    additional refinement step that is used to refine the detected targets:

