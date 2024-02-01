import numpy as np
from hardtarget import noise


def _convert(data, km=True, monostatic=True):
    """Assume data is [m] and two-way range"""
    _data = data.copy()
    if km:
        _data *= 0.001
    if monostatic:
        _data *= 0.5
    return _data


def plot_peaks(axes, data, meta, monostatic=True, snr_dB_limit=12.0):
    r_inds = np.argmax(data["gmf"], axis=1)
    coh_inds = np.arange(data["gmf"].shape[0])

    # If we want monostatic but the data is already in monostatic, dont half data again
    if "round_trip_range" in meta["processing"]:
        if monostatic and not meta["processing"]["round_trip_range"]:
            monostatic = False

    snr = noise.snr(data["gmf"], data["nf_range"])
    snr = snr[coh_inds, r_inds]
    snrdb = 10 * np.log10(snr)

    inds = snrdb > snr_dB_limit
    not_inds = np.logical_not(inds)

    _inds_sty = dict(marker=".", ls="none", color="r")
    _not_inds_sty = dict(marker=".", ls="none", color="b")

    axes[0, 0].plot(
        data["t"][inds], _convert(data["range_peak"][inds], monostatic=monostatic), **_inds_sty
    )
    axes[0, 0].plot(
        data["t"][not_inds],
        _convert(data["range_peak"][not_inds], monostatic=monostatic),
        **_not_inds_sty,
    )
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("range [km]")

    axes[0, 1].plot(
        data["t"][inds],
        _convert(data["range_rate_peak"][inds], monostatic=monostatic),
        **_inds_sty
    )
    axes[0, 1].plot(
        data["t"][not_inds],
        _convert(data["range_rate_peak"][not_inds], monostatic=monostatic),
        **_not_inds_sty,
    )
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("range rate [km/s]")

    axes[1, 0].plot(
        data["t"][inds],
        _convert(data["acceleration_peak"][inds], km=False, monostatic=monostatic),
        **_inds_sty,
    )
    axes[1, 0].plot(
        data["t"][not_inds],
        _convert(data["acceleration_peak"][not_inds], km=False, monostatic=monostatic),
        **_not_inds_sty,
    )
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("acceleration [m/s/s]")

    axes[1, 1].plot(data["t"][inds], np.sqrt(snr[inds]), **_inds_sty)
    axes[1, 1].plot(data["t"][not_inds], np.sqrt(snr[not_inds]), **_not_inds_sty)
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("sqrt(SNR)")

    return axes, None


def plot_detections(axes, data, meta, monostatic=True, snr_dB_limit=12.0):
    r_inds = np.argmax(data["gmf"], axis=1)
    coh_inds = np.arange(data["gmf"].shape[0])

    snr = noise.snr(data["gmf"], data["nf_range"])
    snr = snr[coh_inds, r_inds]
    snrdb = 10 * np.log10(snr)

    inds = snrdb > snr_dB_limit

    # If we want monostatic but the data is already in monostatic, dont half data again
    if "round_trip_range" in meta["processing"]:
        if monostatic and not meta["processing"]["round_trip_range"]:
            monostatic = False

    _style = dict(ls="none", marker=".")

    h00 = axes[0, 0].plot(
        data["t"][inds], _convert(data["range_peak"][inds], monostatic=monostatic), **_style
    )
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("range [km]")

    h01 = axes[0, 1].plot(
        data["t"][inds], _convert(data["range_rate_peak"][inds], monostatic=monostatic), **_style
    )
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("range rate [km/s]")

    h10 = axes[1, 0].plot(
        data["t"][inds],
        _convert(data["acceleration_peak"][inds], km=False, monostatic=monostatic),
        **_style,
    )
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("acceleration [m/s/s]")

    h11 = axes[1, 1].plot(data["t"][inds], snrdb[inds], **_style)
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("SNR [dB]")

    range_data = _convert(data["range_peak"])
    range_rate_data = _convert(data["range_rate_peak"])
    acceleration_data = _convert(data["acceleration_peak"], km=False)

    h11 = axes[0, 2].plot(range_data, range_rate_data, **_style)
    axes[0, 2].set_xlabel("range [km]")
    axes[0, 2].set_ylabel("range rate [km/s]")

    h11 = axes[1, 2].plot(range_data, acceleration_data, **_style)
    axes[1, 2].set_xlabel("range [km]")
    axes[1, 2].set_ylabel("acceleration [km/s/s]")

    handles = [[h00, h01], [h10, h11]]
    return axes, handles


def plot_map(axes, data, meta):
    # GMF
    min_range = meta["processing"]["min_range_gate"]
    max_range = meta["processing"]["max_range_gate"]
    gmf_data_dB = 10 * np.log10(np.abs(data["gmf"].T))

    range_num, coh_int_num = gmf_data_dB.shape
    coh_ints = np.arange(0, coh_int_num)
    range_gates = np.arange(min_range, max_range)

    X, Y = np.meshgrid(coh_ints, range_gates)
    h00 = axes[0].pcolormesh(X, Y, gmf_data_dB)
    axes[0].set_xlabel("Integration number")
    axes[0].set_ylabel("Range gates")
    axes[0].set_title("GMF decoded power [dB]")

    # Noise data
    nf_data_dB = 10 * np.log10(data["gmf_zero_frequency"].T)
    h01 = axes[1].pcolormesh(X, Y, nf_data_dB)
    axes[1].set_xlabel("Integration number")
    axes[1].set_ylabel("Range gates")
    axes[1].set_title("Estimated noise power [dB]")

    # Noise floor
    ranges_data = _convert(meta["ranges"])
    h10 = axes[2].plot(ranges_data, data["nf_range"])
    axes[2].set_xlabel("Ranges [km]")
    axes[2].set_ylabel("Median noise power [arb. unit]")
    axes[2].set_title("Range dependant noise floor")

    handles = [h00, h01, h10]
    return axes, handles
