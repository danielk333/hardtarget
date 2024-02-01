import hardtarget
import matplotlib.pyplot as plt
import numpy as np
import hardtarget.analysis.analysis_utils as analysis_utils
import hardtarget.analysis.analyze_gmf as analyze_gmf
import digital_rf as drf
from matplotlib.animation import FuncAnimation
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ipp", type=int, default=-1)
args = parser.parse_args()


# target = "/home/danielk/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf/"
# config = "/home/danielk/git/hard_target/examples/cfg/test.ini"
# chnl = "uhf"
# start_time = "2021-04-12T12:15:57.0"
# end_time = "2021-04-12T12:15:59.0"
# relative_time = False

target = "/home/danielk/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf_sim/"
config = "/home/danielk/git/hard_target/examples/cfg/sim_test.ini"
chnl = "sim"
start_time = 5.0
end_time = 7.0
relative_time = True

params = hardtarget.load_gmf_params(target, config)

reader = drf.DigitalRFReader(target)


bounds = analysis_utils.compute_bounds(
    reader,
    chnl,
    params["EXP"]["sample_rate"],
    start_time=start_time,
    end_time=end_time,
    relative_time=relative_time,
)
cohint_anim = 19

start_sample = bounds[0]
delta_samples = params["EXP"]["ipp_samp"] * params["PRO"]["n_ipp"]

reduce_axis = [
    False,
    False,
    False,
]
gmf_size = [
    params["PRO"]["n_ranges"],
    params["PRO"]["n_range_rates"],
    params["PRO"]["n_accelerations"],
]
params["PRO"]["gmf_size"] = gmf_size
params["PRO"]["reduce_axis"] = reduce_axis
params["PRO"]["gmflib"] = "numpy_no_reduce"


def analyse_and_plot(axes, cohint_num):
    gmf_vars, gmf_tx = analyze_gmf.integrate_and_match_ipps(
        (reader, chnl),
        (reader, chnl),
        start_sample + cohint_num*delta_samples,
        params,
    )

    rv_gmf = np.max(gmf_vars.vals, axis=2).T
    ra_gmf = np.max(gmf_vars.vals, axis=1).T
    va_gmf = np.max(gmf_vars.vals, axis=0).T

    axes[0].pcolormesh(rv_gmf)
    axes[0].set_xlabel("Range")
    axes[0].set_ylabel("Range rate")
    axes[0].set_title(f"Integration number {cohint_num}")

    axes[1].pcolormesh(ra_gmf)
    axes[1].set_xlabel("Range")
    axes[1].set_ylabel("Acceleration")

    axes[2].pcolormesh(va_gmf)
    axes[2].set_xlabel("Range rate")
    axes[2].set_ylabel("Acceleration")


fig = plt.figure(layout="constrained")
spec = fig.add_gridspec(2, 2)
axes = [
    fig.add_subplot(spec[:, 0]),
    fig.add_subplot(spec[0, 1]),
    fig.add_subplot(spec[1, 1]),
]

if args.ipp < 0:
    ani = FuncAnimation(fig, lambda ind: analyse_and_plot(axes, ind), frames=range(cohint_anim))
else:
    analyse_and_plot(axes, args.ipp)

plt.show()
