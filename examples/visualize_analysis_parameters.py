"""

MAKE DOC HERE
=============

todo

"""

import numpy as np
import matplotlib.pyplot as plt
import hardtarget

target = "/home/danielk/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf/"
config = "/home/danielk/git/hard_target/examples/cfg/test.ini"

params = hardtarget.load_gmf_params(target, config)

t = np.arange(params["PRO"]["read_length"], dtype=np.float64) / params["EXP"]["sample_rate"]
t_ipp = np.arange(params["EXP"]["ipp_samp"], dtype=np.float64) / params["EXP"]["sample_rate"]
rgs_t = np.zeros_like(t)
rel_rgs = params["DER"]["il0_rgs"]
rgs_t[rel_rgs] = 0.5

all_rx = np.ones_like(t_ipp)*0.75
all_rx[np.logical_or(
    t_ipp < params["EXP"]["rx_start"] * 1e-6,
    t_ipp > params["EXP"]["rx_end"] * 1e-6
)] = 0
all_rx[0] = 0

vline_st = dict(ls="--", alpha=0.5)

fig, axes = plt.subplots(2, 2)

axes[0, 0].plot(t, params["DER"]["rx_stencil"], "-b")
axes[0, 0].plot(t, params["DER"]["tx_stencil"], "-r")
for ind in range(params["PRO"]["n_ipp"]):
    axes[0, 0].axvline((params["EXP"]["tx_start"] + params["EXP"]["ipp"] * ind) * 1e-6, c="r", **vline_st)
    axes[0, 0].axvline((params["EXP"]["tx_end"] + params["EXP"]["ipp"] * ind) * 1e-6, c="r", **vline_st)

    axes[0, 0].axvline((params["EXP"]["rx_start"] + params["EXP"]["ipp"] * ind) * 1e-6, c="b", **vline_st)
    axes[0, 0].axvline((params["EXP"]["rx_end"] + params["EXP"]["ipp"] * ind) * 1e-6, c="b", **vline_st)

    axes[0, 0].axvline((params["EXP"]["cal_on"] + params["EXP"]["ipp"] * ind) * 1e-6, c="g", **vline_st)
    axes[0, 0].axvline((params["EXP"]["cal_off"] + params["EXP"]["ipp"] * ind) * 1e-6, c="g", **vline_st)

    dt = params["EXP"]["ipp"] * ind * 1e-6
    axes[0, 0].plot(t_ipp + dt, all_rx, "-c")

axes[0, 0].set_title("Stencils")
axes[0, 0].set_xlabel("Time [s]")

axes[1, 0].plot(t, params["DER"]["rx_stencil"], "-b", label="RX stencil")
axes[1, 0].plot(t, params["DER"]["tx_stencil"], "-r", label="TX stencil")
axes[1, 0].plot(t_ipp, all_rx, "-c", label="RX")
axes[1, 0].plot(t, rgs_t, ".g", label="Range-gates")
axes[1, 0].plot(t, rgs_t, "-g", alpha=0.5)

axes[1, 0].axvline(params["EXP"]["tx_start"] * 1e-6, c="r", **vline_st)
axes[1, 0].axvline(params["EXP"]["tx_end"] * 1e-6, c="r", **vline_st)
axes[1, 0].axvline(params["EXP"]["rx_start"] * 1e-6, c="b", **vline_st)
axes[1, 0].axvline(params["EXP"]["rx_end"] * 1e-6, c="b", **vline_st)
axes[1, 0].axvline(params["EXP"]["cal_on"] * 1e-6, c="m", **vline_st)
axes[1, 0].axvline(params["EXP"]["cal_off"] * 1e-6, c="m", **vline_st)
axes[1, 0].set_xlim(0, params["EXP"]["ipp"] * 1e-6)
axes[1, 0].set_title("Stencils - single")
axes[1, 0].legend()
axes[1, 0].set_xlabel("Time [s]")

X, Y = np.meshgrid(
    np.arange(len(params["DER"]["decimated_sample_times"])),
    params["DER"]["fgmf_accelerations"],
)

axes[0, 1].pcolormesh(X, Y, np.real(params["DER"]["fgmf_acceleration_phasors"]))
axes[0, 1].set_xlabel("Decimated sample")
axes[0, 1].set_ylabel("Acceleration [m/s^2]")
axes[0, 1].set_title("Acceleration phasors - real")

axes[1, 1].pcolormesh(X, Y, np.imag(params["DER"]["fgmf_acceleration_phasors"]))
axes[1, 1].set_xlabel("Decimated sample")
axes[1, 1].set_ylabel("Acceleration [m/s^2]")
axes[1, 1].set_title("Acceleration phasors - imag")

plt.show()
