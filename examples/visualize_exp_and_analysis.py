import numpy as np
import matplotlib.pyplot as plt
import hardtarget

target = "/home/danielk/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf/"
config = "/home/danielk/git/hard_target/examples/cfg/test.ini"

params = hardtarget.load_gmf_params(target, config)

t = np.arange(params["read_length"], dtype=np.float64)/params["sample_rate"]

vline_st = dict(ls="--", alpha=0.5)

fig, axes = plt.subplots(2, 3)

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

plt.show()
