import numpy as np
import scipy.constants
import digital_rf as drf
import pathlib
import configparser
import shutil
import logging
from hardtarget.version import __version__

logger = logging.getLogger(__name__)


def noise_generator(noise_sigma, shape, dtype=np.complex128):
    return noise_sigma * (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(dtype)


def waveform_generator(t, baud_length, frequency, code, dtype=np.complex128):
    t_ind = (t // baud_length).astype(np.int64)
    signal = np.zeros(t.shape, dtype=dtype)
    inds = np.logical_and(t >= 0, t <= baud_length * len(code))
    signal[inds] = code[t_ind[inds]]
    return signal


def simulate_drf(
    output_path,
    range_function,
    sim_params,
    experiment_params,
    snr_function=None,
    compression_level=0,
    dir_cadence_secs=3600,
    file_cadence_millisecs=1000,
    chnl="sim",
    clobber=False,
    dtype=np.complex128,
):
    """
    # TODO: docstring
    """
    if output_path is not None:
        output_path = pathlib.Path(output_path)
        output_path.mkdir(exist_ok=True)

        dstdir = output_path / chnl
        if dstdir.is_dir() and clobber:
            logger.info(f"'{dstdir}' exists and clobber is on: removing dir")
            shutil.rmtree(dstdir)
        elif dstdir.is_dir():
            raise FileExistsError(f"Directory '{dstdir}' exists")
        dstdir.mkdir(exist_ok=False)

    sample_rate = experiment_params["sample_rate"]
    codes = experiment_params["code"].shape[0]

    dt64_epoch = np.datetime64(sim_params["epoch"])
    unix_epoch = dt64_epoch.astype("datetime64[us]").astype("int64") * 1e-6
    samp_epoch = unix_epoch * sample_rate

    ipp_samp = np.round(experiment_params["ipp"] * 1e-6 * sample_rate).astype(np.int64)
    wavelength = scipy.constants.c / (experiment_params["frequency"] * 1e6)
    T_rx_start_samp = np.round(experiment_params["rx_start"] * 1e-6 * sample_rate).astype(np.int64)
    T_rx_end_samp = np.round(experiment_params["rx_end"] * 1e-6 * sample_rate).astype(np.int64)
    T_rx_samps = T_rx_end_samp - T_rx_start_samp
    T_rx_select = np.full((ipp_samp,), False, dtype=bool)
    T_rx_select[T_rx_start_samp:T_rx_end_samp] = True

    T_tx_start_samp = np.round(experiment_params["tx_start"] * 1e-6 * sample_rate).astype(np.int64)
    T_tx_end_samp = np.round(experiment_params["tx_end"] * 1e-6 * sample_rate).astype(np.int64)
    T_tx_samps = T_tx_end_samp - T_tx_start_samp

    samp_t0 = samp_epoch + sim_params["start_time"] * sample_rate
    samp_t0 = np.round((samp_t0 // ipp_samp) * ipp_samp).astype(np.int64)
    samp_t1 = samp_epoch + sim_params["end_time"] * sample_rate
    samp_t1 = np.round((samp_t1 // ipp_samp) * ipp_samp).astype(np.int64)

    sig_t0 = sim_params["target_start_time"]
    samp_sig_t0 = samp_epoch + sig_t0 * sample_rate
    sig_t1 = sim_params["target_end_time"]
    samp_sig_t1 = samp_epoch + sig_t1 * sample_rate

    sim_pulses = int((samp_t1 - samp_t0) / ipp_samp)

    if output_path is None:
        simulated_signal = np.empty((samp_t1 - samp_t0,), dtype=dtype)
    else:
        rf_writer = drf.DigitalRFWriter(
            str(dstdir),  # directory
            dtype,  # dtype
            dir_cadence_secs,  # subdir cadence secs    => one dir per hour
            file_cadence_millisecs,  # file cadence millisecs => one file per second
            samp_t0,  # start global index
            sample_rate,  # sample rate numerator
            1,  # sample rate denominator
            uuid_str="tbd",
            compression_level=compression_level,
            checksum=False,
            num_subchannels=1,
            is_continuous=True,
            marching_periods=False,
        )

    t_tx = np.arange(T_tx_samps) / sample_rate
    for pid in range(sim_pulses):
        samp0 = pid * ipp_samp + samp_t0
        signal = np.zeros((ipp_samp,), dtype=dtype)

        if sim_params["noise_sigma"] > 0:
            signal[T_rx_select] += noise_generator(
                sim_params["noise_sigma"],
                (T_rx_samps,),
                dtype=dtype,
            )

        tx_wave = waveform_generator(
            t_tx,
            experiment_params["baud_length"] * 1e-6,
            experiment_params["frequency"] * 1e6,
            experiment_params["code"][pid % codes],
            dtype=dtype,
        )
        tx_amp0 = sim_params["tx_amp"] if "tx_amp" in sim_params else 1.0

        # TODO: this assumes rx streches over tx, generalize
        signal[T_tx_start_samp:T_tx_end_samp] += tx_amp0*tx_wave

        s0 = samp0 + T_tx_start_samp
        s1 = s0 + T_tx_samps / sample_rate
        t0 = (s0 - samp_epoch) / sample_rate
        if s0 >= samp_sig_t0 and s1 <= samp_sig_t1:
            r0 = range_function(np.array([t0], dtype=np.float64))[0]
            sn0 = snr_function(t0 + t_tx) if snr_function is not None else 1.0
            rg0 = np.round((r0 / scipy.constants.c) * sample_rate).astype(np.int64)
            rg_samp0 = rg0 + T_tx_start_samp

            if rg_samp0 >= T_rx_start_samp and rg_samp0 <= T_rx_end_samp:
                if sim_params["noise_sigma"] > 0:
                    amp0 = np.sqrt(sn0*2*sim_params["noise_sigma"]**2)
                else:
                    amp0 = np.sqrt(sn0)
                ranges = range_function(t0 + t_tx)
                phase = np.mod(ranges / wavelength, 1) * np.pi * 2

                rx_wave = tx_wave * amp0 * np.exp(1j * phase)
                signal[rg_samp0:(rg_samp0 + T_tx_samps)] += rx_wave

        if output_path is None:
            simulated_signal[(pid * ipp_samp):((pid + 1) * ipp_samp)] = signal
        else:
            rf_writer.rf_write(signal)

    EXP_SECTION = "Experiment"
    meta = configparser.ConfigParser()
    meta.add_section(EXP_SECTION)
    exp = meta[EXP_SECTION]
    exp["name"] = "simulation"
    exp["version"] = __version__

    # forward values from experiment config file
    props = [
        "sample_rate",
        "ipp",
        "tx_pulse_length",
        "rx_start",
        "rx_end",
        "tx_start",
        "tx_end",
        "cal_on",
        "cal_off",
    ]

    for prop in props:
        if prop in experiment_params:
            exp[prop] = str(experiment_params[prop])
    exp["rx_channel"] = chnl
    exp["tx_channel"] = chnl
    # add
    exp["radar_frequency"] = str(experiment_params["frequency"])

    if output_path is None:
        return simulated_signal
    else:
        rf_writer.close()
        # write metadata file
        metafile = dstdir.parent / "metadata.ini"
        with open(metafile, 'w') as f:
            meta.write(f)
