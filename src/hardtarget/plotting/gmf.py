import pathlib
import h5py
import numpy as np
import datetime


def collect_data(paths):
    nf_vecs = []
    nf_ts = []
    data = None
    t_vec_pos = 0
    for path in paths:
        with h5py.File(path, "r") as hf:
            gmf = hf["gmf"][()]
            r_vec = hf["range_peak"][()]
            v_vec = hf["range_rate_peak"][()]
            a_vec = hf["acceleration_peak"][()]
            g_vec = hf["gmf_peak"][()]
            ranges = hf["ranges"][()]
            min_range_gate = hf["processing"]["min_range_gate"][()]
            max_range_gate = hf["processing"]["max_range_gate"][()]

            _t_conv = (hf["processing"]["n_ipp"][()]*hf["experiment"]["ipp"][()])*1e-6

        n_cohints = gmf.shape[0]
        t_vec = np.arange(n_cohints)*_t_conv + t_vec_pos
        t_vec_pos = t_vec[-1] + 1
        nf_vecs.append(np.nanmedian(gmf, axis=0))
        nf_ts.append(t_vec[0])

        if data is None:
            data = {}
            data["gmf_vec"] = gmf
            data["t_vecs"] = t_vec
            data["r_vecs"] = r_vec
            data["v_vecs"] = v_vec
            data["a_vecs"] = a_vec
            data["g_vecs"] = g_vec
        else:
            data["gmf_vec"] = np.append(data["gmf_vec"], gmf, axis=0)
            data["t_vecs"] = np.append(data["t_vecs"], t_vec)
            data["r_vecs"] = np.append(data["r_vecs"], r_vec)
            data["v_vecs"] = np.append(data["v_vecs"], v_vec)
            data["a_vecs"] = np.append(data["a_vecs"], a_vec)
            data["g_vecs"] = np.append(data["g_vecs"], g_vec)
        if "ranges" not in data:
            data["ranges"] = ranges

    data["nf_vecs"] = np.stack(nf_vecs, axis=0)
    data["nf_range"] = np.nanmedian(data["nf_vecs"], axis=0)
    data["nf_time"] = np.nanmedian(data["nf_vecs"], axis=1)
    data["nf_ts"] = np.array(nf_ts)
    data["min_range_gate"] = min_range_gate
    data["max_range_gate"] = max_range_gate

    return data


def yield_chunked_data(paths, chunk_size=None):
    paths.sort()
    pth_num = len(paths)
    if chunk_size is None:
        chunks = 1
        chunk_size = pth_num
    else:
        chunks = pth_num // chunk_size + 1
    for ind in range(chunks):
        sub_paths = paths[(ind*chunk_size):((ind + 1)*chunk_size)]
        yield collect_data(sub_paths)


def collect_paths(
    folder,
    start_time=None,
    end_time=None,
    relative_time=False,
):
    folder = pathlib.Path(folder)
    fl = list(folder.glob("**/gmf*.h5"))
    fl.sort()
    fl_epochs = [
        int(file.stem.split("-")[1])*1e-6
        for file in fl
    ]

    epoch_unix = fl_epochs[0]
    max_unix = fl_epochs[-1]

    if relative_time:
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = max_unix - epoch_unix
        unix_t0 = epoch_unix + start_time
        unix_t1 = epoch_unix + end_time
    else:
        if start_time is None:
            start_time = datetime.datetime.utcfromtimestamp(epoch_unix)
        if end_time is None:
            end_time = datetime.datetime.utcfromtimestamp(max_unix)

        dt64_t0 = start_time if isinstance(start_time, np.datetime64) else np.datetime64(start_time)
        unix_t0 = dt64_t0.astype("datetime64[us]").astype("int64")*1e-6

        dt64_t1 = end_time if isinstance(end_time, np.datetime64) else np.datetime64(end_time)
        unix_t1 = dt64_t1.astype("datetime64[us]").astype("int64")*1e-6

    fl = [
        file for file, ep in zip(fl, fl_epochs)
        if ep >= unix_t0 and ep <= unix_t1
    ]

    # TODO - provide useful feedback if only one file exists
    # and start_time, end_time is given, but not a perfect match with the file

    return fl


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

    # GMF
    min_y, max_y = data["min_range_gate"], data["max_range_gate"]
    gmf_data = np.log10(data["gmf_vec"].T)
    size_y, size_x = gmf_data.shape
    x = np.arange(0, size_x)
    y = np.arange(min_y, max_y)
    h00 = axes[0, 0].pcolormesh(x, y, gmf_data)
    axes[0, 0].set_xlabel("? [?]")
    axes[0, 0].set_ylabel("range gates")

    # NF
    nf_data = data["nf_vecs"].T
    nf_size_y, nf_size_x = nf_data.shape
    nf_x = np.arange(0, nf_size_x)
    nf_y = np.arange(min_y, max_y)
    h01 = axes[0, 1].pcolormesh(nf_x, nf_y, nf_data)
    axes[0, 1].set_xlabel("? [?]")
    # TODO - y-label should be number from min_range_gate to max_range_gate
    axes[0, 1].set_ylabel("range gates")

    # 
    ranges_data = convert(data["ranges"])
    h10 = axes[1, 0].plot(ranges_data, data["nf_range"])
    axes[1, 0].set_xlabel("ranges [km]")
    axes[1, 0].set_ylabel("? [?]")

    #
    h11 = axes[1, 1].plot(data["nf_ts"], data["nf_time"])
    axes[1, 1].set_xlabel("? [?]")
    axes[1, 1].set_ylabel("? [?]")

    handles = [[h00, h01], [h10, h11]]
    return axes, handles
