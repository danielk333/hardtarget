import numpy as np



def convert(data, km=True, monostatic=True):
    """Assume data is [m] and two-way range"""
    if km:
        data *= 0.001
    if monostatic:
        data *= 0.5
    return data


def plot_peaks(axes, data):
    snr = data["gmf_vec"]/data["nf_range"][None, :] - 1
    r_inds = np.argmax(data["gmf_vec"], axis=1)
    coh_inds = np.arange(data["gmf_vec"].shape[0])
    snr = snr[coh_inds, r_inds]

    h00 = axes[0, 0].plot(data["t_vecs"], convert(data["r_vecs"]))
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("range [km]")

    h01 = axes[0, 1].plot(data["t_vecs"], convert(data["v_vecs"]))
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("range rate [km/s]")

    h10 = axes[1, 0].plot(data["t_vecs"], convert(data["a_vecs"]))
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("acceleration [km/s/s]")

    h11 = axes[1, 1].plot(data["t_vecs"], np.sqrt(snr))
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("sqrt(ENR) [1]")

    handles = [[h00, h01], [h10, h11]]
    return axes, handles


def plot_detections(axes, data):
    snr = data["gmf_vec"]/data["nf_range"][None, :] - 1
    r_inds = np.argmax(data["gmf_vec"], axis=1)
    coh_inds = np.arange(data["gmf_vec"].shape[0])
    snr = snr[coh_inds, r_inds]
    snrdb = 10*np.log10(snr)
    inds = snrdb > 12.0

    # CONVERT DATA
    # Range data [m] is assumed to be bi-static (round_trip_range==True)
    # TODO - bring in the config parameter (round_trip_range) and do the right thing.

    h00 = axes[0, 0].plot(data["t_vecs"][inds], convert(data["r_vecs"][inds]), ls="none", marker='.')
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("range [km]")

    h01 = axes[0, 1].plot(data["t_vecs"][inds], convert(data["v_vecs"][inds]), ls="none", marker='.')
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("range_rate [km/s]")

    h10 = axes[1, 0].plot(data["t_vecs"][inds], convert(data["a_vecs"][inds]), ls="none", marker='.')
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_ylabel("acceleration [km/s/s]")

    h11 = axes[1, 1].plot(data["t_vecs"][inds], snrdb[inds], ls="none", marker='.')
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("SNR [dB]")

    range_data = convert(data["r_vecs"])
    range_rate_data = convert(data["v_vecs"])
    acceleration_data = convert(data["a_vecs"])

    h11 = axes[0, 2].plot(range_data, range_rate_data, ls="none", marker='.')
    axes[0, 2].set_xlabel("range [km]")
    axes[0, 2].set_ylabel("range_rate [km/s]")

    h11 = axes[1, 2].plot(range_data, acceleration_data, ls="none", marker='.')
    axes[1, 2].set_xlabel("range [km]")
    axes[1, 2].set_ylabel("acceleration [km/s/s]")

    handles = [[h00, h01], [h10, h11]]
    return axes, handles


def plot_map(axes, data):
    # TODO: there is a better way to get noise background which also gives
    # higher resolution in the estimates, (by estimating the noise sigma directly
    # from the receiver samples distribution as a function of range and time)
    # But this works as a placeholder

    # GMF
    min_y, max_y = data["min_range_gate"], data["max_range_gate"]
    gmf_data = np.log10(data["gmf_vec"].T)
    size_y, size_x = gmf_data.shape
    x = np.arange(0, size_x)
    y = np.arange(min_y, max_y)
    h00 = axes[0, 0].pcolormesh(x, y, gmf_data)
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Range gates")

    # NF
    nf_data = data["nf_vecs"].T
    nf_size_y, nf_size_x = nf_data.shape
    nf_x = np.arange(0, nf_size_x)
    nf_y = np.arange(min_y, max_y)
    h01 = axes[0, 1].pcolormesh(nf_x, nf_y, nf_data)
    axes[0, 1].set_xlabel("? [?]")
    # TODO - y-label should be number from min_range_gate to max_range_gate
    axes[0, 1].set_ylabel("Range gates")

    #
    ranges_data = convert(data["ranges"])
    h10 = axes[1, 0].plot(ranges_data, data["nf_range"])
    axes[1, 0].set_xlabel("Ranges [km]")
    axes[1, 0].set_ylabel("Time [s]")

    #
    h11 = axes[1, 1].plot(data["nf_ts"], data["nf_time"])
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("Mean noise level [?]")

    handles = [[h00, h01], [h10, h11]]
    return axes, handles
