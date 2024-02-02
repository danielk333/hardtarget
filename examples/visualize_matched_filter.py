import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import hardtarget
import hardtarget.analysis.analysis_utils as analysis_utils
import digital_rf as drf

target = "/home/danielk/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf_sim/"
config = "/home/danielk/git/hard_target/examples/cfg/sim_test.ini"
chnl = "sim"
start_time = 5.0
end_time = 7.0
relative_time = True
range_sample = 3665
acc_ind = 0


# target = "/home/danielk/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf/"
# config = "/home/danielk/git/hard_target/examples/cfg/test.ini"
# chnl = "uhf"
# start_time = "2021-04-12T12:15:57.5"
# end_time = "2021-04-12T12:15:57.9"
# relative_time = False
# range_sample = 410
# acc_ind = 76

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

print("Range: ", params["DER"]["ranges"][range_sample]*1e-3, " km")
print("Acceleration: ", params["DER"]["accelerations"][acc_ind]*1e-3, " km/s^2")

phasors = params["DER"]["acceleration_phasors"][acc_ind]
rg = params["DER"]["rgs"][range_sample]
n_ipp = params["PRO"]["n_ipp"]

start_sample = bounds[0]
delta_samples = params["EXP"]["ipp_samp"] * n_ipp

z_rx = reader.read_vector_1d(start_sample, delta_samples, chnl)

samps = np.arange(delta_samples)
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

xcorr = sel_rxs * txs
echo = np.sum(xcorr.copy().reshape(-1, params["PRO"]["frequency_decimation"]), axis=-1)
c_echo = echo * phasors

dec_txlen = params["EXP"]["tx_pulse_samps"] // params["PRO"]["frequency_decimation"]
dec_sig_samps = np.arange(params["DER"]["dec_signal_length"])
dec_signal_vec = np.zeros((params["DER"]["dec_signal_length"], ), dtype=np.complex64)
dec_rx_window_indices = params["DER"]["dec_rx_window_indices"]

fvec = params["DER"]["fvec"]

dec_signal_vec[dec_rx_window_indices] = echo
spec = fft.fft(dec_signal_vec)

c_dec_signal_vec = dec_signal_vec.copy()
c_dec_signal_vec[dec_rx_window_indices] = c_echo
c_spec = fft.fft(c_dec_signal_vec, len(fvec))


fig, axes = plt.subplots(3, 1)

axes[0].semilogy(t[not_used_sig], np.abs(z_rx[not_used_sig]), ".k")
axes[0].semilogy(
    t[params["DER"]["rx_stencil"]][rx_window],
    np.abs(z_rx[params["DER"]["rx_stencil"]][rx_window]),
    ".g"
)
axes[0].semilogy(
    t[params["DER"]["rx_stencil"]][not_rx_window],
    np.abs(z_rx[params["DER"]["rx_stencil"]][not_rx_window]),
    ".b",
)
axes[0].semilogy(t[params["DER"]["tx_stencil"]], np.abs(z_rx[params["DER"]["tx_stencil"]]), ".r")
axes[0].set_title("Raw signal power")

axes[1].plot(comp_samps, np.real(sel_rxs) / np.sum(np.abs(sel_rxs)), "-g", label="RX")
axes[1].plot(comp_samps, np.real(txs) / np.sum(np.abs(txs)), "-r", alpha=0.5, label="TX")
for ind in range(n_ipp):
    axes[1].axvline(params["EXP"]["tx_pulse_samps"] * (ind + 1), ls="--", c="c")
axes[1].set_title("Stenciled signals")
axes[1].legend()

axes[2].plot(comp_samps, np.real(xcorr), "-k")
axes[2].plot(comp_samps, np.abs(xcorr), "--b")
for ind in range(n_ipp):
    axes[2].axvline(params["EXP"]["tx_pulse_samps"] * (ind + 1), ls="--", c="c")
axes[2].set_title("Correlated echo")

fig, axes = plt.subplots(3, 2)

axes[0, 0].plot(comp_samps, np.real(xcorr), "-k")
axes[0, 0].plot(comp_samps, np.abs(xcorr), "--b")
for ind in range(n_ipp):
    axes[0, 0].axvline(params["EXP"]["tx_pulse_samps"] * (ind + 1), ls="--", c="c")
axes[0, 0].set_title("Correlated echo")

axes[0, 1].plot(nfft, np.real(phasors))
for ind in range(n_ipp):
    axes[0, 1].axvline(dec_txlen * (ind + 1), ls="--", c="c")
axes[0, 1].set_title("Acceleration phasors")

axes[1, 0].plot(nfft, np.real(echo), "-k")
for ind in range(n_ipp):
    axes[1, 0].axvline(dec_txlen * (ind + 1), ls="--", c="c")
axes[1, 0].set_title("Correlated & decimated echo")

axes[1, 1].plot(nfft, np.real(c_echo), "-b")
for ind in range(n_ipp):
    axes[1, 1].axvline(dec_txlen * (ind + 1), ls="--", c="c")
axes[1, 1].set_title("Correlated & decimated & acceleration corrected echo")

axes[2, 0].plot(fvec, np.abs(spec), "-k")
axes[2, 0].set_title("Correlated & decimated echo spectrum")

axes[2, 1].plot(fvec, np.abs(c_spec), "-b")
axes[2, 1].set_title("Correlated & decimated & acceleration corrected echo spectrum")


plt.show()
