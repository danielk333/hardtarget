import numpy as np
from hardtarget.gmf import GMF_LIBS
import logging


logger = logging.getLogger(__name__)

####################################################################
# ANALYSE IPPS
####################################################################


def analyze_ipps(rx, tx, start_sample, gmf_params):
    """
    TODO: clean up this and all other docstrings when structure is done

    Analyse ipps runs the gmf function.

    Parameters
    ----------
    rx: tuple
        (DigitalRF.reader, str) the drf reader and name of the channel with the rx signal
    tx: tuple
        (DigitalRF.reader, str) the drf reader and name of the channel with the tx signal
    start_sample: int
        Start sample of integration period
    gmf_params: dict
        GMF parameters

    Returns
    -------
    gmf_vec: numpy array
        ?
    gmf_dc_vec: numpy array
        ?
    vvec: numpy array
        ?
    avec: numpy array
        ?
    tx_amp: numpy array
        ?
    """

    # gmf lib
    gmf_lib = gmf_params.get("gmflib", None)
    if gmf_lib is None or gmf_lib not in GMF_LIBS:
        gmf_lib = "c" if "c" in GMF_LIBS else "numpy"
    gmf = GMF_LIBS[gmf_lib]

    # parameters
    rx_stencil = gmf_params["rx_stencil"]
    tx_stencil = gmf_params["tx_stencil"]
    n_range_gates = gmf_params["n_ranges"]
    acc_phasors = gmf_params["acceleration_phasors"]
    rgs = gmf_params["rgs"]
    frequency_decimation = gmf_params["frequency_decimation"]
    rx_window_indices = gmf_params["rx_window_indices"]

    rx_reader, rx_channel = rx
    tx_reader, tx_channel = tx

    # read data vector with n_ipp + n_extra ipp's (to allow for searching across to subsequent pulses)
    z_rx = rx_reader.read_vector_1d(start_sample, gmf_params["read_length"], rx_channel)

    if tx_channel != rx_channel or tx_reader != rx_reader:
        z_tx = tx_reader.read_vector_1d(start_sample, gmf_params["read_length"], tx_channel)
    else:
        z_tx = np.copy(z_rx)

    # clean ground clutter, get separate transmit waveform and echo vectors
    # z_rx = z_rx * rx_stencil
    # z_tx = z_tx * tx_stencil
    z_rx = z_rx[rx_stencil]
    z_tx = z_tx[tx_stencil]

    # TODO: WHY THIS?
    # truncate the tx vector to be exactly the length of n_ipp
    # z_tx = z_tx[0: (n_ipp * ipp)]

    # conjugate, so that when matched filtering, it will cancel out phase of transmit waveform.
    # scale transmit waveform to unity power
    tx_amp = np.sqrt(np.sum(np.abs(z_tx) ** 2.0))
    z_tx = np.conj(z_tx) / tx_amp

    # maximum match function value
    gmf_vec = np.zeros(n_range_gates, dtype=np.float32)
    # best fitting range-rate
    v_vec = np.zeros(n_range_gates, dtype=np.int64)
    # best fitting range-rate change
    a_vec = np.zeros(n_range_gates, dtype=np.int64)
    # 0-frequency gmf output
    gmf_dc_vec = np.zeros(n_range_gates, dtype=np.float32)

    if tx_amp > 1.0:
        gmf(
            z_tx,
            z_rx,
            acc_phasors,
            rgs,
            frequency_decimation,
            gmf_vec,
            gmf_dc_vec,
            v_vec,
            a_vec,
            rx_window_indices,
        )

    return gmf_vec, gmf_dc_vec, v_vec, a_vec, tx_amp**2.0
