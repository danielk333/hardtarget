import numpy as np
import numpy.typing as npt

from hardtarget.echo_search.types import (
    EchoSearchCfgParams,
    EchoSearchProParams,
    EchoSearchVars,
)
from hardtarget.types import ExpDef


def xcorr_numpy(
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

    # NOTE NOT READY!

    rx_sum = np.sum(rx, axis=0)

    # Extract n tx len windows sliding over rx
    rx_windows = np.zeros((len(rx_sum) + len(tx) - 1, len(tx)))
    # Calculate normalization coefficient for each rx window, TODO: Why is it +1? or should it be tx-1?
    rx_windows[len(tx) - 1 : len(rx_sum)] = np.lib.stride_tricks.sliding_window_view(rx_sum, len(tx))
    rx_windows[: len(tx) - 1] = rx_windows[len(tx) - 1]
    rx_windows[len(rx_sum) - 1 :] = rx_windows[len(rx_sum) - 1]
    rx_norm_coef = np.sqrt(np.sum(rx_windows * np.conj(rx_windows), axis=1))
    filt = rx_norm_coef < 0.000001
    rx_norm_coef[filt] = 1

    # Doppler freqs to search through
    doppler_freqs = np.arange(
        cfg_params.doppler_freq_min,
        cfg_params.doppler_freq_max + cfg_params.doppler_freq_step,
        cfg_params.doppler_freq_step,
    )

    # Calculate signal model
    doppler_freq_samp = (
        np.atleast_2d(doppler_freqs).T * 2 * np.pi * np.arange(1, len(tx) + 1) * exp_def.t_samp_usec * 1e-6
    )
    signal_model = tx.flatten() * (np.cos(doppler_freq_samp) + np.sin(doppler_freq_samp) * 1j)

    # tx/signal model normalization coefficient
    tx_norm_coef = np.sqrt(np.sum(signal_model * np.conj(signal_model), axis=1))

    # Correlate over each doppler frequency
    decoded = np.zeros((len(doppler_freqs), len(rx_sum) + len(tx) - 1))  # TODO
    for i, df in enumerate(doppler_freqs):
        decoded[i] = np.correlate(signal_model[i], rx_sum, mode="full")

    # Normalize correlation to [0,1]
    output_power = np.abs(decoded / (rx_norm_coef * np.atleast_2d(tx_norm_coef).T)) ** 2

    # Extract information
    max_corr_per_doppler = np.max(output_power, axis=1)
    best_doppler_index = np.argmax(max_corr_per_doppler)
    max_corr = max_corr_per_doppler[best_doppler_index]
    max_corr_start_ind = np.argmax(output_power[best_doppler_index])
    tot_pow = decoded[best_doppler_index, max_corr_start_ind]

    delay = np.arange(-len(tx), len(rx_sum))[max_corr_start_ind]

    # If correlation present, calculate noise on non coherent samples
    if max_corr > cfg_params.corr_filter_limit:
        end_samp = delay + len(tx) + 1
        rx_real_filtered = np.delete(
            rx.real, np.arange(delay, end_samp if end_samp <= len(rx.real) else len(rx.real)), axis=1
        )
        rx_imag_filtered = np.delete(
            rx.imag, np.arange(delay, end_samp if end_samp <= len(rx.imag) else len(rx.imag)), axis=1
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
        max_corr_delay=delay,
        best_doppler=doppler_freqs[best_doppler_index],
        tot_pow=tot_pow,
        mean=np.mean(rx_samps),
        std_dev=np.std(rx_samps),
    )
