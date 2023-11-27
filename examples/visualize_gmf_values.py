import hardtarget
import matplotlib.pyplot as plt
import numpy as np
import hardtarget.analysis.analysis_utils as analysis_utils
import hardtarget.analysis.analyze_gmf as analyze_gmf
import digital_rf as drf

target = "/home/danielk/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf/"
config = "/home/danielk/git/hard_target/examples/cfg/test.ini"

params = hardtarget.load_gmf_params(target, config)

reader = drf.DigitalRFReader(target)
chnl = "uhf"

bounds = analysis_utils.compute_bounds(
    reader,
    "uhf",
    params["sample_rate"],
    start_time="2021-04-12T12:15:57.5",
    end_time="2021-04-12T12:15:57.9",
    relative_time=False,
)

start_sample = bounds[0]
delta_samples = params["ipp_samp"] * params["n_ipp"]

reduce_axis = [
    False,
    False,
    False,
]
gmf_size = [
    params["n_ranges"],
    params["n_range_rates"],
    params["n_accelerations"],
]
params["gmf_size"] = gmf_size
params["reduce_axis"] = reduce_axis
params["gmflib"] = "numpy_no_reduce"


gmf_vars, gmf_tx = analyze_gmf.integrate_and_match_ipps(
    (reader, chnl),
    (reader, chnl),
    start_sample,
    params,
)

rv_gmf = np.max(gmf_vars.vals, axis=2).T
ra_gmf = np.max(gmf_vars.vals, axis=1).T
va_gmf = np.max(gmf_vars.vals, axis=0).T

fig = plt.figure(layout="constrained")
spec = fig.add_gridspec(2, 2)

axes = [
    fig.add_subplot(spec[:, 0]),
    fig.add_subplot(spec[0, 1]),
    fig.add_subplot(spec[1, 1]),
]

axes[0].pcolormesh(rv_gmf)
axes[0].set_xlabel("Range")
axes[0].set_ylabel("Range rate")

axes[1].pcolormesh(ra_gmf)
axes[1].set_xlabel("Range")
axes[1].set_ylabel("Acceleration")

axes[2].pcolormesh(va_gmf)
axes[2].set_xlabel("Range rate")
axes[2].set_ylabel("Acceleration")

plt.show()
