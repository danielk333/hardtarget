import numpy as np
import matplotlib.pyplot as plt
import hardtarget

target = "/home/danielk/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf/"
config = "/home/danielk/git/hard_target/examples/cfg/test.ini"

params = hardtarget.load_gmf_params(target, config)

t = np.arange(params["read_length"], dtype=np.float64)/params["sample_rate"]
rgs_t = np.zeros_like(t)
rel_rgs = params["rgs"] + params["stencil_start_samp"]
rgs_t[rel_rgs] = 0.5

vline_st = dict(ls="--", alpha=0.5)

fig, axes = plt.subplots(2, 2)

axes[0, 0].plot(t, params["rx_stencil"], "-b", label="RX")
axes[0, 0].plot(t, params["tx_stencil"], "-r", label="TX")
for ind in range(params["n_ipp"]):
    axes[0, 0].axvline((params["tx_start"] + params["ipp"]*ind)*1e-6, c="r", **vline_st)
    axes[0, 0].axvline((params["tx_end"] + params["ipp"]*ind)*1e-6, c="r", **vline_st)

    axes[0, 0].axvline((params["rx_start"] + params["ipp"]*ind)*1e-6, c="b", **vline_st)
    axes[0, 0].axvline((params["rx_end"] + params["ipp"]*ind)*1e-6, c="b", **vline_st)

    axes[0, 0].axvline((params["cal_on"] + params["ipp"]*ind)*1e-6, c="g", **vline_st)
    axes[0, 0].axvline((params["cal_off"] + params["ipp"]*ind)*1e-6, c="g", **vline_st)

axes[0, 0].set_title("Stencils")
axes[0, 0].legend()

axes[1, 0].plot(t, params["rx_stencil"], "-b", label="RX")
axes[1, 0].plot(t, params["tx_stencil"], "-r", label="TX")
axes[1, 0].plot(t, rgs_t, ".g", label="Range-gates")
axes[1, 0].plot(t, rgs_t, "-g", alpha=0.5)

axes[1, 0].axvline(params["tx_start"]*1e-6, c="r", **vline_st)
axes[1, 0].axvline(params["tx_end"]*1e-6, c="r", **vline_st)
axes[1, 0].axvline(params["rx_start"]*1e-6, c="b", **vline_st)
axes[1, 0].axvline(params["rx_end"]*1e-6, c="b", **vline_st)
axes[1, 0].axvline(params["cal_on"]*1e-6, c="m", **vline_st)
axes[1, 0].axvline(params["cal_off"]*1e-6, c="m", **vline_st)
axes[1, 0].set_xlim(0, params["ipp"]*1e-6)
axes[1, 0].set_title("Stencils - single")
axes[1, 0].legend()


axes[0, 1].pcolormesh(np.real(params["acceleration_phasors"]))
axes[0, 1].set_title("Acceleration phasors - real")

axes[1, 1].pcolormesh(np.imag(params["acceleration_phasors"]))
axes[1, 1].set_title("Acceleration phasors - imag")

plt.show()
