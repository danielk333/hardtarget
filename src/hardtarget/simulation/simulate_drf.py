import numpy as np
import scipy.constants
import scipy.interpolate as scint
import digital_rf as drf
import pathlib
import configparser
import shutil
from hardtarget.version import __version__


def noise_generator(noise_sigma, shape):
    return noise_sigma * (np.random.randn(*shape) + 1j * np.random.randn(*shape))


def waveform_generator(t, baud_length, frequency, code, dtype=np.complex128):
    t_ind = (t // baud_length).astype(np.int64)
    signal = np.zeros(t.shape, dtype=dtype)
    inds = np.logical_and(t >= 0, t <= baud_length * len(code))
    signal[inds] = code[t_ind[inds]]
    return signal


def simulate_drf(
    target, sim_data, sim_params, experiment_params, compression_level=0, chnl="sim", clobber=False
):
    """
    # TODO: docstring
    """
    if target is not None:
        target = pathlib.Path(target)
        target.mkdir(exist_ok=True)

        dstdir = target / chnl
        if dstdir.is_dir() and clobber:
            shutil.rmtree(dstdir)
        elif dstdir.is_dir():
            raise FileExistsError(f"Directory '{dstdir}' exists")
        dstdir.mkdir(exist_ok=False)

    interp_data = {}
    for key in sim_data.keys():
        if key == "t":
            continue
        interp_data[key] = scint.interp1d(sim_data["times"], sim_data[key])

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

    sig_t0 = np.min(sim_data["times"])
    samp_sig_t0 = samp_epoch + sig_t0 * sample_rate
    sig_t1 = np.max(sim_data["times"])
    samp_sig_t1 = samp_epoch + sig_t1 * sample_rate

    sim_pulses = int((samp_t1 - samp_t0) / ipp_samp)

    if target is None:
        simulated_signal = np.empty((samp_t1 - samp_t0,), dtype=np.complex64)
    else:
        rf_writer = drf.DigitalRFWriter(
            str(dstdir),  # directory
            np.complex64,  # dtype
            3600,  # subdir cadence secs    => one dir per hour
            1000,  # file cadence millisecs => one file per second
            samp_t0,  # start global index
            sample_rate,  # sample rate numerator
            1,  # sample rate denominator
            uuid_str="tbd",
            compression_level=compression_level,
            checksum=False,
            num_subchannels=1,
            is_continuous=True,
            marching_periods=True,
        )

    t_tx = np.arange(T_tx_samps) / sample_rate
    for pid in range(sim_pulses):
        samp0 = pid * ipp_samp + samp_t0
        signal = np.zeros((ipp_samp,), dtype=np.complex64)

        if sim_params["noise_sigma"] > 0:
            signal[T_rx_select] += noise_generator(sim_params["noise_sigma"], (T_rx_samps,))

        tx_wave = waveform_generator(
            t_tx,
            experiment_params["baud_length"] * 1e-6,
            experiment_params["frequency"] * 1e6,
            experiment_params["code"][pid % codes],
            dtype=np.complex64,
        )
        tx_amp0 = sim_params["tx_amp"] if "tx_amp" in sim_params else 1.0

        # TODO: this assumes rx streches over tx, generalize
        signal[T_tx_start_samp:T_tx_end_samp] += tx_amp0*tx_wave

        s0 = samp0 + T_tx_start_samp
        t0 = (s0 - samp_epoch) / sample_rate
        if s0 >= samp_sig_t0 and s0 <= samp_sig_t1:
            r0 = interp_data["ranges"](t0)
            v0 = interp_data["velocities"](t0)
            a0 = interp_data["accelerations"](t0)
            sn0 = interp_data["snr"](t0) if "snr" in interp_data else 1.0

            rg0 = np.round((r0 / scipy.constants.c) * sample_rate).astype(np.int64)
            rg_samp0 = rg0 + T_tx_start_samp
        else:
            rg_samp0 = None

        if rg_samp0 is not None and (
            rg_samp0 >= T_rx_start_samp and rg_samp0 <= T_rx_end_samp
        ):
            ranges = r0 + v0*t_tx + 0.5*a0*t_tx**2
            phase = np.mod(ranges / wavelength, 1) * np.pi * 2

            if sim_params["noise_sigma"] > 0 and "snr" in interp_data:
                amp0 = np.sqrt(sn0*2*sim_params["noise_sigma"]**2)
            else:
                amp0 = np.sqrt(sn0)

            rx_wave = tx_wave * amp0 * np.exp(1j * phase)
            signal[rg_samp0:(rg_samp0 + T_tx_samps)] += rx_wave

        if target is None:
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

    if target is None:
        return simulated_signal
    else:
        # write metadata file
        metafile = dstdir.parent / "metadata.ini"
        with open(metafile, 'w') as f:
            meta.write(f)
