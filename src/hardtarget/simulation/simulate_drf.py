import numpy as np
import scipy.constants
import scipy.interpolate as scint
import digital_rf as drf
import pathlib
import configparser
import shutil


def to_i2x16(zz):
    zz2x16 = np.empty((len(zz), 2), dtype=np.int16)
    zz2x16[:, 0] = zz.real.astype(np.int16)
    zz2x16[:, 1] = zz.imag.astype(np.int16)
    return zz2x16


def noise_generator(noise_sigma, shape):
    return noise_sigma * (np.random.randn(*shape) + 1j * np.random.randn(*shape))


def phase_function(t, range0, velocity0, acceleration0, wavelength):
    rng = range0 + t * velocity0 + 0.5 * t**2 * acceleration0
    return rng * np.pi * 2 / wavelength


def waveform_generator(t, phase, baud_length, frequency, code, dtype=np.complex64):
    t_ind = (t // baud_length).astype(np.int64)
    signal = np.zeros(t.shape, dtype=dtype)
    inds = np.logical_and(t >= 0, t <= baud_length * len(code))
    signal[inds] = np.exp(-1j * phase) * code[t_ind[inds]]
    return signal


def simulate_drf(
    target, sim_data, sim_params, experiment_params, compression_level=0, chnl="sign", clobber=False
):
    """ """
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
    samp_t0 = (samp_t0 // ipp_samp) * ipp_samp
    samp_t1 = samp_epoch + sim_params["end_time"] * sample_rate
    samp_t1 = (samp_t1 // ipp_samp) * ipp_samp

    sig_t0 = np.min(sim_data["times"])
    samp_sig_t0 = samp_epoch + sig_t0 * sample_rate
    sig_t1 = np.max(sim_data["times"])
    samp_sig_t1 = samp_epoch + sig_t1 * sample_rate

    sim_pulses = int((samp_t1 - samp_t0) / ipp_samp)

    rf_writer = drf.DigitalRFWriter(
        str(dstdir),  # directory
        np.int16,  # dtype
        3600,  # subdir cadence secs    => one dir per hour
        1000,  # file cadence millisecs => one file per second
        samp_t0,  # start global index
        sample_rate,  # sample rate numerator
        1,  # sample rate denominator
        uuid_str="tbd",
        compression_level=compression_level,
        checksum=False,
        is_complex=True,
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
            np.ones_like(t_tx),
            experiment_params["baud_length"],
            experiment_params["frequency"],
            experiment_params["code"][pid % codes],
        )
        # TODO: this assumes rx streches over tx, generalize
        signal[T_tx_start_samp:T_tx_end_samp] += tx_wave

        if samp0 <= samp_sig_t0 or samp0 >= samp_sig_t1:
            ipp_samples = to_i2x16(signal)
            rf_writer.rf_write(ipp_samples)
            continue

        t0 = (samp0 - samp_epoch) / sample_rate
        r0 = interp_data["ranges"](t0)
        rg0 = np.round((r0 / scipy.constants.c) * sample_rate).astype(np.int64)

        if rg0 < (T_rx_start_samp - T_tx_start_samp) or rg0 > (T_rx_end_samp - T_tx_start_samp):
            ipp_samples = to_i2x16(signal)
            rf_writer.rf_write(ipp_samples)
            continue

        sn = interp_data["snrs"](t0)
        phase = phase_function(
            t_tx,
            r0,
            interp_data["velocities"](t0),
            interp_data["accelerations"](t0),
            wavelength,
        )
        rx_wave = waveform_generator(
            t_tx,
            phase,
            experiment_params["baud_length"],
            experiment_params["frequency"],
            experiment_params["code"][pid % codes],
        )

        signal[(rg0 + T_tx_start_samp):(rg0 + T_tx_start_samp + T_tx_samps)] += rx_wave*sn

        ipp_samples = to_i2x16(signal)
        rf_writer.rf_write(ipp_samples)

    EXP_SECTION = "Experiment"
    meta = configparser.ConfigParser()
    meta.add_section(EXP_SECTION)
    exp = meta[EXP_SECTION]
    exp["name"] = "simulation"
    exp["version"] = "1.0"

    # forward values from experiment config file
    props = [
        "sample_rate", "ipp", "tx_pulse_length",
        "doppler_sign", "round_trip_range",
        "rx_start", "rx_end",
        "tx_start", "tx_end",
        "cal_on", "cal_off"
    ]

    for prop in props:
        if prop in experiment_params:
            exp[prop] = str(experiment_params[prop])
    exp["rx_channel"] = chnl
    exp["tx_channel"] = chnl
    # add
    exp["radar_frequency"] = str(experiment_params["frequency"])

    # write metadata file
    metafile = dstdir.parent / "metadata.ini"
    with open(metafile, 'w') as f:
        meta.write(f)
