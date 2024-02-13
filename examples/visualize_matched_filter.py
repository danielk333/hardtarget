import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import hardtarget
import pathlib

config_path = pathlib.Path("./examples/cfg/sim_test.ini").resolve()

range0 = 2000e3
vel0 = 0.15e3
acel0 = -0.20e3

t_start = 2.5
echo_len = 5.0
t_abs = np.arange(t_start, t_start + echo_len, 0.1)
t = t_abs - t_start
peak_SNR = 10**(25.0/10.0)

look_time_ind = np.argmin(np.abs(t - echo_len/2))
look_time = t[look_time_ind]

simulation_params = {
    "epoch": "2021-04-12T12:15:40",
    "start_time": t_start,
    "end_time": t_start + echo_len,
    "noise_sigma": 0,
    "tx_amp": np.sqrt(peak_SNR),
}

simulation_data = {
    "ranges": range0 + vel0*t + acel0*0.5*t**2,
    "velocities": vel0 + t*acel0,
    "accelerations": np.ones_like(t)*acel0,
    "snr": peak_SNR*np.exp(-(t - 5.0)**2/2.0**2),
    "times": t_abs,
}

range_true = simulation_data["ranges"][look_time_ind]

params_exp = {
    "sample_rate": 1000000,
    "ipp": 20000,
    "tx_pulse_length": 1920.0,
    "tx_start": 82.0,
    "tx_end": 2002.0,
    "rx_start": 0,
    "rx_end": 20000,
    "cal_on": 19900.0,
    "cal_off": 19997.0,
    "frequency": 929.6,
    "baud_length": 30.0,
    "code": hardtarget.load_radar_code("leo_bpark"),
}
params_exp["radar_frequency"] = params_exp["frequency"]
params_pro = hardtarget.gmf_in_utils.load_gmf_processing_params(config_path)
params_exp, params_pro, params_der = hardtarget.gmf_in_utils.compute_derived_gmf_params(
    params_exp, params_pro
)
params = {
    "EXP": params_exp,
    "PRO": params_pro,
    "DER": params_der
}

range_ind = np.argmin(np.abs(params["DER"]["ranges"] - range_true))
acc_ind = np.argmin(np.abs(params["DER"]["accelerations"] - acel0))

print("Best range gate: ", params["DER"]["abs_rgs"][range_ind])
print("Frequency decimation: ", params["PRO"]["frequency_decimation"])

simulated_signal = hardtarget.simulation.drf(
    None,
    simulation_data,
    simulation_params,
    params_exp,
)

print("Range: ", params["DER"]["ranges"][range_ind]*1e-3, " km")
print("Acceleration: ", params["DER"]["accelerations"][acc_ind]*1e-3, " km/s^2")

phasors = params["DER"]["acceleration_phasors"][acc_ind]
rg = params["DER"]["rgs"][range_ind]
n_ipp = params["PRO"]["n_ipp"]

start_sample = np.round(look_time * params["EXP"]["sample_rate"]).astype(np.int64)
start_sample = (start_sample // params["EXP"]["ipp_samp"]) * params["EXP"]["ipp_samp"]

delta_samples = params["EXP"]["ipp_samp"] * n_ipp

z_rx = simulated_signal[start_sample:(start_sample + delta_samples)]

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
dec_rx_window_indices = params["DER"]["dec_rx_window_indices"] + params["DER"]["dec_rgs"][range_ind]

fvec = params["DER"]["fvec"]

print("Decimated signal length: ", params["DER"]["dec_signal_length"])
print("FFT len: ", len(fvec))

dec_signal_vec[dec_rx_window_indices] = echo
spec = fft.fft(dec_signal_vec)

c_dec_signal_vec = dec_signal_vec.copy()
c_dec_signal_vec[dec_rx_window_indices] = c_echo
c_spec = fft.fft(c_dec_signal_vec)


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
axes[2].plot(comp_samps, np.abs(xcorr), "--k")
for ind in range(n_ipp):
    axes[2].axvline(params["EXP"]["tx_pulse_samps"] * (ind + 1), ls="--", c="c")
axes[2].set_title("Correlated echo")

fig, axes = plt.subplots(3, 2)

axes[0, 0].plot(comp_samps, np.real(xcorr), "-k")
axes[0, 0].plot(comp_samps, np.abs(xcorr), "--k")
for ind in range(n_ipp):
    axes[0, 0].axvline(params["EXP"]["tx_pulse_samps"] * (ind + 1), ls="--", c="c")
axes[0, 0].set_title("Correlated echo")

axes[0, 1].plot(nfft, np.real(phasors))
for ind in range(n_ipp):
    axes[0, 1].axvline(dec_txlen * (ind + 1), ls="--", c="c")
axes[0, 1].set_title("Acceleration phasors")

axes[1, 0].plot(nfft, np.real(echo), "-k")
axes[1, 0].plot(nfft, np.abs(echo), "--k")
for ind in range(n_ipp):
    axes[1, 0].axvline(dec_txlen * (ind + 1), ls="--", c="c")
axes[1, 0].set_title("Correlated & decimated echo")

axes[1, 1].plot(nfft, np.real(c_echo), "-b")
axes[1, 1].plot(nfft, np.abs(c_echo), "--b")
for ind in range(n_ipp):
    axes[1, 1].axvline(dec_txlen * (ind + 1), ls="--", c="c")
axes[1, 1].set_title("Correlated & decimated & acceleration corrected echo")

# inds = np.abs(fvec) < 3e3
# axes[2, 0].semilogy(fvec[inds], np.abs(spec[inds]), "-k")
axes[2, 0].semilogy(fvec, np.abs(spec), "-k")
axes[2, 0].set_title("Correlated & decimated echo spectrum")

# axes[2, 1].semilogy(fvec[inds], np.abs(c_spec[inds]), "-b")
axes[2, 1].semilogy(fvec, np.abs(c_spec), "-b")
axes[2, 1].set_title("Correlated & decimated & acceleration corrected echo spectrum")


plt.show()
