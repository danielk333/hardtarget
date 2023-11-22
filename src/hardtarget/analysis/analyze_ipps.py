import numpy as np
from hardtarget.gmf import GMF_LIBS
from hardtarget import gmf_utils

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
    rx : tuple
        (DigitalRF.reader, str) the drf reader and name of the channel with the rx signal
    tx : tuple
        (DigitalRF.reader, str) the drf reader and name of the channel with the tx signal
    start_sample : int
        Start sample of integration period
    gmf_params : dict
        GMF parameters

    Returns
    -------
    gmf_vars : hardtarget.gmf_utils.GMFVariables
        Named tuple container for compacting the variables set by the GMF function.
    tx_amp2 : float
        Squared summed tx amplitude
    """

    # gmf lib
    gmf_lib = gmf_params.get("gmflib", None)
    if gmf_lib is None or gmf_lib not in GMF_LIBS:
        gmf_lib = "c" if "c" in GMF_LIBS else "numpy"
    gmf = GMF_LIBS[gmf_lib]

    # TODO: implement this
    if not gmf_params["reduce_range_rate"] and gmf_params["reduce_acceleration"]:
        raise NotImplementedError("reduce settings not implemented")

    # parameters
    rx_stencil = gmf_params["rx_stencil"]
    tx_stencil = gmf_params["tx_stencil"]

    rx_reader, rx_channel = rx
    tx_reader, tx_channel = tx

    # read data vector with n_ipp + n_extra ipp's (to allow for searching across to subsequent pulses)
    z_rx = rx_reader.read_vector_1d(start_sample, gmf_params["read_length"], rx_channel)

    if tx_channel != rx_channel or tx_reader != rx_reader:
        z_tx = tx_reader.read_vector_1d(start_sample, gmf_params["read_length"], tx_channel)
    else:
        z_tx = np.copy(z_rx)

    # clean ground clutter, get separate transmit waveform and echo vectors
    z_rx = z_rx[rx_stencil]
    z_tx = z_tx[tx_stencil]

    # conjugate, so that when matched filtering, it will cancel out phase of transmit waveform.
    # scale transmit waveform to unity power
    tx_amp2 = np.sum(np.abs(z_tx) ** 2.0)
    tx_amp = np.sqrt(tx_amp2)
    z_tx = np.conj(z_tx) / tx_amp

    # TODO: There are better ways of estimating the background noise by
    #   removing all coherent echoes first and using the individual signal samples

    gmf_vars = gmf_utils.GMFVariables(
        vals = np.zeros(gmf_params["gmf_size"], dtype=np.float32),
        dc = np.zeros([gmf_params["n_ranges"], ], dtype=np.float32),
        r_ind = np.full(gmf_params["gmf_size"], -1, dtype=np.int32),
        v_ind = np.full(gmf_params["gmf_size"], -1, dtype=np.int32),
        a_ind = np.full(gmf_params["gmf_size"], -1, dtype=np.int32),
    )

    if tx_amp > 1.0:
        gmf(
            z_tx,
            z_rx,
            gmf_vars,
            gmf_params,
        )

    return gmf_vars, tx_amp2
