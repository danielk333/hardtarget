import numpy as np
import matplotlib.pyplot as plt
import hardtarget
import hardtarget.analysis.analysis_utils as analysis_utils
import digital_rf as drf

target = "/home/danielk/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf/"
config = "/home/danielk/git/hard_target/examples/cfg/test.ini"

params = hardtarget.load_gmf_params(target, config)

reader = drf.DigitalRFReader(target)

bounds = analysis_utils.compute_bounds(
    reader,
    "uhf",
    params["EXP"]["sample_rate"],
    start_time="2021-04-12T12:15:57.5",
    end_time="2021-04-12T12:15:57.9",
    relative_time=False,
)
range_sample = 410
acc_ind = 40

phasors = params["DER"]["acceleration_phasors"][acc_ind]
rg = params["DER"]["rgs"][range_sample]

start_sample = bounds[0]
delta_samples = params["EXP"]["ipp_samp"] * params["PRO"]["n_ipp"]

z_rx = reader.read_vector_1d(start_sample, delta_samples, "uhf")

samps = np.arange(len(z_rx))
t = samps / params["EXP"]["sample_rate"]

not_used_sig = np.full(t.shape, True, dtype=bool)
not_used_sig[params["DER"]["rx_stencil"]] = False
not_used_sig[params["DER"]["tx_stencil"]] = False

txs = z_rx[params["DER"]["tx_stencil"]]
rxs = z_rx[params["DER"]["rx_stencil"]]
sel_rxs = rxs[rg + params["DER"]["rx_window_indices"]]
comp_samps = np.arange(txs.size)
nfft = np.arange(phasors.size)

rx_window = np.full(rxs.shape, False, dtype=bool)
rx_window[rg + params["DER"]["rx_window_indices"]] = True
not_rx_window = np.logical_not(rx_window)

fig = plt.figure(layout="constrained")
spec = fig.add_gridspec(3, 2)

axes = [
    fig.add_subplot(spec[0, :]),
    [fig.add_subplot(spec[1, 0]), fig.add_subplot(spec[1, 1])],
    [fig.add_subplot(spec[2, 0]), fig.add_subplot(spec[2, 1])],
]

axes[0].semilogy(t[not_used_sig], np.abs(z_rx[not_used_sig]), ".k")
axes[0].semilogy(
    t[params["DER"]["rx_stencil"]][rx_window], np.abs(z_rx[params["DER"]["rx_stencil"]][rx_window]), ".g"
)
axes[0].semilogy(
    t[params["DER"]["rx_stencil"]][not_rx_window],
    np.abs(z_rx[params["DER"]["rx_stencil"]][not_rx_window]),
    ".b",
)
axes[0].semilogy(t[params["DER"]["tx_stencil"]], np.abs(z_rx[params["DER"]["tx_stencil"]])**2, ".r")
axes[0].set_title("Raw signal power")

axes[1][0].plot(comp_samps, np.real(sel_rxs) / np.sum(np.real(sel_rxs)), "-g", label="RX")
axes[1][0].plot(comp_samps, np.real(txs) / np.sum(np.real(txs)), "-r", label="TX")
axes[1][0].set_title("Stenciled signals")
axes[1][0].legend()

axes[1][1].plot(nfft, np.real(phasors))
axes[1][1].set_title("Acceleration phasors")

echo = np.sum((sel_rxs * txs).reshape(-1, params["PRO"]["frequency_decimation"]), axis=-1)

axes[2][0].plot(nfft, np.real(echo), "-k")
axes[2][1].plot(nfft, np.real(echo * phasors), "-b")

axes[2][0].set_title("Correlated echo")
axes[2][1].set_title("Acceleration corrected echo")


plt.show()
