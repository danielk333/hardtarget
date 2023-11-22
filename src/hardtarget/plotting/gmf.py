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

        n_cohints = gmf.shape[0]
        t_vec = np.arange(n_cohints) + t_vec_pos
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
    return fl


def plot_peaks(axes, data):
    h00 = axes[0, 0].plot(data["t_vecs"], data["r_vecs"])
    h01 = axes[0, 1].plot(data["t_vecs"], data["v_vecs"])
    h10 = axes[1, 0].plot(data["t_vecs"], data["a_vecs"])
    h11 = axes[1, 1].plot(data["t_vecs"], data["g_vecs"])
    handles = [[h00, h01], [h10, h11]]
    return axes, handles


def plot_map(axes, data):
    h00 = axes[0, 0].pcolormesh(np.log10(data["gmf_vec"].T))
    h01 = axes[0, 1].pcolormesh(data["nf_vecs"].T)
    h10 = axes[1, 0].plot(data["ranges"]*1e-3, data["nf_range"])
    h11 = axes[1, 1].plot(data["nf_ts"], data["nf_time"])
    handles = [[h00, h01], [h10, h11]]
    return axes, handles
