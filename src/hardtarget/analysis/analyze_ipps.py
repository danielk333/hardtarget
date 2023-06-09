import numpy as np
import time
from hardtarget.utilities import read_vector_c81d
from hardtarget.gmf import GMF_LIBS


####################################################################
# ANALYSE IPPS
####################################################################


def analyze_ipps(drf_reader, i0, params):
    """
    This runs the gmf function 
    """

    # gmf lib
    gmf_lib = params.get("gmflib", None)
    if gmf_lib is None or gmf_lib not in GMF_LIBS:
        gmf_lib = "c"
    gmf = GMF_LIBS[gmf_lib]

    # logger
    logger = params["logger"]
    job = params["job"]

    # parameters
    ipp = params["ipp"]
    n_ipp = params["n_ipp"]
    n_extra = params["n_extra"]
    rx_channel = params["rx_channel"]
    tx_channel = params["tx_channel"]
    rx_stencil = params["rx_stencil"]
    tx_stencil = params["tx_stencil"]
    n_range_gates = params["n_range_gates"]
    acc_phasors = params["acc_phasors"]
    rgs_float = params["rgs_float"]
    frequency_decimation = params["frequency_decimation"]
    ranges = params["ranges"]
    range_rates = params["range_rates"]
    accs = params["accs"]
    sample_rate = params["sample_rate"]

    # timestamp
    cput0 = time.time()

    # read data vector with n_ipps, and a little extra
    z = read_vector_c81d (drf_reader, i0, (n_ipp + n_extra) * ipp, rx_channel)

    # make a separate copy to hold transmit pulse, and the echo
    z_rx = np.copy(z)

    if tx_channel != rx_channel:
        z = read_vector_c81d (drf_reader, i0, (n_ipp + n_extra) * ipp, tx_channel)
    z_tx = np.copy(z)

    # clean ground clutter, get separate transmit waveform and echo vectors
    z_tx = z_tx*tx_stencil
    z_rx = z_rx*rx_stencil

    # truncate the tx vector to be exactly the length of n_ipp
    z_tx = z_tx[0:(n_ipp * ipp)]

    # conjugate, so that when matched filtering, it will cancel out phase of transmit waveform.
    # scale transmit waveform to unity power
    tx_amp = np.sqrt(np.sum(np.abs(z_tx)**2.0))
    z_tx = np.conj(z_tx)/tx_amp

    # maximum match function value
    gmf_vec = np.zeros(n_range_gates, dtype=np.float32)
    # best fitting range-rate
    v_vec = np.zeros(n_range_gates, dtype=np.float32)
    # best fitting range-rate change
    a_vec = np.zeros(n_range_gates, dtype=np.float32)
    # 0-frequency gmf output
    gmf_dc_vec = np.zeros(n_range_gates, dtype=np.float32)

    if tx_amp > 1.0:
        gmf(z_tx, z_rx, acc_phasors, rgs_float, frequency_decimation, gmf_vec, gmf_dc_vec, v_vec, a_vec)

    mri = np.argmax(gmf_vec)

    # timestamp
    cput1 = time.time()

    # logging
    info = {
        "job": job["idx"],
        "GMF": np.max(gmf_vec),
        "r_max": ranges[mri],
        "vel_max": range_rates[int(v_vec[mri])]/1e3,
        "a_max": accs[int(a_vec[mri])]
    }
    msg = "job {job} GMF={GMF:1.2g} r_max={r_max:1.2f} (km) vel_max={vel_max:1.2f} (km/s) a_max={a_max:1.2f} (m/s**2)"
    logger.debug(msg.format(**info))
    info = {
        "time": cput1-cput0,
        "real": (cput1-cput0)/(n_ipp*ipp/sample_rate)
    }
    msg = "time {time:1.2f} cpu/real {real:1.2f}"
    logger.debug(msg.format(**info))

    avec = accs[np.array(a_vec, dtype=np.int)]
    vvec = range_rates[np.array(v_vec, dtype=np.int)]

    return (gmf_vec, gmf_dc_vec, vvec, avec, tx_amp**2.0)
