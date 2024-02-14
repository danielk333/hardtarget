import hardtarget
import matplotlib.pyplot as plt
import hardtarget.analysis.utils as a_utils
import hardtarget.analysis.analyze_gmf as analyze_gmf
import digital_rf as drf
from matplotlib.animation import FuncAnimation
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ipp", type=int, default=1)
args = parser.parse_args()

raise NotImplementedError("Todo fix this example")

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


bounds = a_utils.compute_bounds(
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
    True,
    True,
    False,
]
gmf_size = [
    params["PRO"]["n_ranges"],
    params["PRO"]["n_range_rates"],
    params["PRO"]["n_accelerations"],
]
params["PRO"]["gmf_size"] = [z for ind, z in enumerate(gmf_size) if not reduce_axis[ind]]
params["PRO"]["reduce_axis"] = reduce_axis
params["PRO"]["gmf_grid_lib"] = "gmfnp_no_reduce"
params["PRO"]["gmf_fine_tune"] = False

print("GMF Output size: ", params["PRO"]["gmf_size"])


def analyse_and_plot(ax, cohint_num):
    gmf_vars, gmf_tx = analyze_gmf.integrate_and_match_ipps(
        (reader, chnl),
        (reader, chnl),
        start_sample + cohint_num*delta_samples,
        params,
    )

    ax.semilogy(gmf_vars.vals)
    ax.set_xlabel("Acceleration")
    ax.set_title(f"Integration number {cohint_num}")

    # ax.pcolormesh(gmf_vars.vals)
    # ax.set_xlabel("Range rate")
    # ax.set_ylabel("Acceleration")
    # ax.set_title(f"Integration number {cohint_num}")


fig, ax = plt.subplots()

if args.ipp < 0:
    ani = FuncAnimation(fig, lambda ind: analyse_and_plot(ax, ind), frames=range(cohint_anim))
else:
    analyse_and_plot(ax, args.ipp)

plt.show()
