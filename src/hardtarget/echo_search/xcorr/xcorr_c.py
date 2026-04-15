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

    pows = np.zeros([pro_params.doppler_freq_size, (len(rx) + len(tx))], dtype=np.complex64)
    pows_normalized = np.zeros([pro_params.doppler_freq_size, (len(rx) + len(tx))], dtype=np.complex64)
    max_pow_per_doppler = np.zeros([pro_params.doppler_freq_size], dtype=np.complex64)
    maxpowind = np.zeros([pro_params.doppler_freq_size], dtype=np.int32)

    # TODO Fix for subresolution, is it needed??
    tx_no_sub_res = tx[:, 0].flatten()
    error_code = mfclib.xcorr_echo_search(
        tx_no_sub_res,  # 1
        tx_no_sub_res.size,  # 2
        rx,  # 3
        rx.size,  # 4
        cfg_params.doppler_freq_min,  # 5
        cfg_params.doppler_freq_max,  # 6
        cfg_params.doppler_freq_step,  # 7
        pro_params.doppler_freq_size,  # 8
        int(exp_params.t_samp_usec),  # 9
        pows,  # 10
        np.array([pro_params.doppler_freq_size, len(rx) + len(tx)], dtype=np.int32),  # 11
        pows_normalized,  # 12
        np.array([pro_params.doppler_freq_size, len(rx) + len(tx)], dtype=np.int32),  # 13
        max_pow_per_doppler,  # 14
        max_pow_per_doppler.size,  # 15
        maxpowind,  # 16
        maxpowind.size,  # 17
    )

    best_value_index = np.argmax(max_pow_per_doppler)

    assert error_code == 0, f"Event search C-function returned error {error_code}"

    return EchoSearchVars(
        max_pow=np.max(pows, axis=0),
        max_pow_norm=np.max(pows_normalized, axis=0),
        max_peak=max_pow_per_doppler[best_value_index],
        max_pow_ind=maxpowind[best_value_index],
        best_doppler=cfg_params.doppler_freq_min + (best_value_index * cfg_params.doppler_freq_step),
        ipps_pow=np.sum(np.square(abs(rx))),
    )
