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
    exp_def: ExpDef,
    cfg_params: EchoSearchCfgParams,
    pro_params: EchoSearchProParams,
) -> EchoSearchVars:
    """
    Wrapper for crosscorrelation c implementation.

    Args:
        tx: Transmitted signal, one dimensional
        rx: Recived signal per channel, shape: (n_channels, n_rx_samps)
        exp_def: Experiment definition
        cfg_params: Configuration parameters
        pro_params: Echo search process specific parameters

    Returns:
        Data defining power of echo and more.
    """

    rx_sum = np.sum(rx, axis=0) if rx.ndim > 1 else rx
    rx_sum = rx_sum[:: cfg_params.range_gate_step].copy()
    tx = tx.flatten()[:: cfg_params.range_gate_step]

    correlation = np.zeros([len(pro_params.doppler_frequencies), (len(rx_sum) + len(tx))], dtype=np.complex64)
    corr_normalized = np.zeros(
        [len(pro_params.doppler_frequencies), (len(rx_sum) + len(tx))], dtype=np.complex64
    )
    corr_per_doppler = np.zeros([len(pro_params.doppler_frequencies)], dtype=np.complex64)
    max_corr_ind = np.zeros([len(pro_params.doppler_frequencies)], dtype=np.int32)

    error_code = mfclib.xcorr_echo_search(
        tx.flatten(),  # 1
        len(tx),  # 2
        rx_sum,  # 3
        rx_sum.size,  # 4
        pro_params.doppler_frequencies,  # 5
        len(pro_params.doppler_frequencies),  # 6
        cfg_params.range_gate_step,  # 7
        int(exp_def.t_samp_usec),  # 8
        correlation,  # 9
        np.array(correlation.shape, dtype=np.int32),  # 10
        corr_normalized,  # 11
        np.array(corr_normalized.shape, dtype=np.int32),  # 12
        corr_per_doppler,  # 13
        corr_per_doppler.size,  # 14
        max_corr_ind,  # 15
        max_corr_ind.size,  # 16
    )

    assert error_code == 0, f"Event search C-function returned error {error_code}"

    best_doppler_index = np.argmax(corr_per_doppler)
    max_corr = corr_per_doppler[best_doppler_index]
    max_corr_start_ind = max_corr_ind[best_doppler_index]
    delay = (np.arange(-len(tx), len(rx_sum)) * cfg_params.range_gate_step)[
        max_corr_start_ind
    ]  # TODO: max_corr_start_ind + 1 improves the result very much! WHY?
    max_pow = correlation[best_doppler_index, max_corr_start_ind]

    # If correlation present, calculate noise on non coherent samples
    if max_corr > cfg_params.corr_filter_limit:
        end_samp = delay + len(tx) + 1
        rx_real_filtered = np.delete(
            rx.real,
            np.arange(delay, end_samp if end_samp <= len(rx.real) else len(rx.real)),
            axis=1 if rx.ndim > 1 else 0,
        )
        rx_imag_filtered = np.delete(
            rx.imag,
            np.arange(delay, end_samp if end_samp <= len(rx.imag) else len(rx.imag)),
            axis=1 if rx.ndim > 1 else 0,
        )
        rx_samps = np.concatenate(
            (rx_real_filtered, rx_imag_filtered),
            axis=None,
        ).astype(np.float64)
    else:
        rx_samps = np.concatenate((rx.real, rx.imag), axis=None).astype(np.float64)

    return EchoSearchVars(
        max_corr=max_corr,
        max_corr_ind=max_corr_start_ind,
        max_corr_delay=delay + 1 / cfg_params.range_gate_step,
        best_doppler=cfg_params.doppler_freq_min + (best_doppler_index * cfg_params.doppler_freq_step),
        tot_pow=max_pow,
        mean=np.mean(rx_samps),
        std_dev=np.std(rx_samps),
    )
