import numpy as np
import numpy.typing as npt

from hardtarget.echo_search.types import (
    EchoSearchCfgParams,
    EchoSearchProParams,
    EchoSearchVars,
)
from hardtarget.libs import load_c_lib
from hardtarget.types import ExpDef

mfclib = load_c_lib()


def xcorr_c(
    tx: npt.NDArray[np.complex64],
    rx: npt.NDArray[np.complex64],
    exp_params: ExpDef,
    cfg_params: EchoSearchCfgParams,
    pro_params: EchoSearchProParams,
) -> EchoSearchVars:
    """
    Wrapper for crosscorrelation c implementation.

    Args:
        tx: Transmitted signal, one dimensional
        rx: Recived signal per channel
        exp_params: Experiment definition
        cfg_params: Configuration parameters
        pro_params: Echo search process specific parameters

    Returns:
        Data defining power of echo and more.
    """

    rx_sum = np.sum(rx, axis=0)
    correlation = np.zeros([pro_params.doppler_freq_size, (len(rx_sum) + len(rx_sum))], dtype=np.complex64)
    corr_normalized = np.zeros([pro_params.doppler_freq_size, (len(rx_sum) + len(tx))], dtype=np.complex64)
    corr_per_doppler = np.zeros([pro_params.doppler_freq_size], dtype=np.complex64)
    max_corr_ind = np.zeros([pro_params.doppler_freq_size], dtype=np.int32)

    error_code = mfclib.xcorr_echo_search(
        tx.flatten(),  # 1
        len(tx),  # 2
        rx_sum,  # 3
        rx_sum.size,  # 4
        cfg_params.doppler_freq_min,  # 5
        cfg_params.doppler_freq_max,  # 6
        cfg_params.doppler_freq_step,  # 7
        pro_params.doppler_freq_size,  # 8
        int(exp_params.t_samp_usec),  # 9
        correlation,  # 10
        np.array([pro_params.doppler_freq_size, len(rx_sum) + len(tx)], dtype=np.int32),  # 11
        corr_normalized,  # 12
        np.array([pro_params.doppler_freq_size, len(rx_sum) + len(tx)], dtype=np.int32),  # 13
        corr_per_doppler,  # 14
        corr_per_doppler.size,  # 15
        max_corr_ind,  # 16
        max_corr_ind.size,  # 17
    )

    best_value_index = np.argmax(corr_per_doppler)

    assert error_code == 0, f"Event search C-function returned error {error_code}"

    rx_samps = np.concatenate((rx.real, rx.imag), axis=None).astype(np.float64)

    return EchoSearchVars(
        max_corr=corr_per_doppler[best_value_index],
        max_corr_ind=max_corr_ind[best_value_index],
        best_doppler=cfg_params.doppler_freq_min + (best_value_index * cfg_params.doppler_freq_step),
        tot_pow=np.sum(np.square(abs(rx))),
        mean=np.mean(rx_samps),
        std_dev=np.std(rx_samps),
    )
