# # Matched filter visualization
# ---
# A visualization of the matched filter process. In this example zero noise is set to be able to better
# visualize the filtering process.


import os

# Workaround to make jupyter notebook find utils
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft
from radardef.radar_stations.eiscat import load_radar_code
from radardef.types import BoundParams, ExpParams
from scipy import constants

import hardtarget
from hardtarget.types.constants import AnalysisMethod

sys.path.insert(1, str(Path(os.path.abspath("")) / "docs" / "examples" / "extras"))
import utils

try:
    config = Path(__file__).parent.parent.absolute() / "cfg" / "sim_test.ini"
except NameError:
    config = Path(os.path.abspath("")) / "docs" / "examples" / "cfg" / "sim_test.ini"

# ## Simulate DRF data
# ---

# Exp
exp = ExpParams(
    name="simulation",
    radar_frequency=929.6,
    t_ipp_usec=20000,
    ipp_samps=20000,
    sample_rate=1000000,
    t_samp_usec=1,
    rx_channels=["sim"],
    t_tx_start_usec=82.0,
    t_tx_end_usec=2002.0,
    t_rx_start_usec=0,
    t_rx_end_usec=20000,
    tx_channel="sim",
    tx_pulse_length=1920,
    t_cal_on_usec=19900.0,
    t_cal_off_usec=19997.0,
    wavelength=constants.c / (929.6 * 1e6),
    code=load_radar_code("leo_bpark"),
)
# Bounds
bounds_params = BoundParams(
    ts_start_usec=1445511612.8,
    ts_end_usec=1445551228.8,
)
# Generate simulated data
tmp_sim_path = tempfile.TemporaryDirectory(suffix="_drf")
range0, vel0, acel0, t_rel, t_abs, SNR, echo_len, _ = utils.sim_data(
    output_path=Path(tmp_sim_path.name),
    exp_params=exp,
    bounds_params=bounds_params,
    config=config,
    zero_noise=True,
)

# ## Setup process and access the data
# ---
# Read the measurement and then start a process to get the correct process parameters

measurement = hardtarget.Measurement(Path(tmp_sim_path.name), config, AnalysisMethod.target_estimation)
analyse_process = hardtarget.process.GMFProcess(
    config,
    measurement.exp_params,
    measurement.cfg_params,
    measurement.pro_params,
    None,
    None,
    None,
    None,
)
pro_params = analyse_process.pro_params
cfg_params = analyse_process.cfg_params

# ## Matched filter calculation
# ---
# Calculate further parameters

look_time_ind = np.argmin(np.abs(t_rel - echo_len / 2))
look_time = t_rel[look_time_ind]
range_true = (range0 + vel0 * t_rel + acel0 * 0.5 * t_rel**2)[look_time_ind]
range_ind = np.argmin(np.abs(pro_params.ranges - range_true))
acc_ind = np.argmin(np.abs(pro_params.accelerations - acel0))
phasors = pro_params.acceleration_phasors[acc_ind]
rg = pro_params.rel_rgs[range_ind]
print("Best range gate: ", rg)
print("Frequency decimation: ", cfg_params.frequency_decimation)
print("Range: ", pro_params.ranges[range_ind] * 1e-3, " km")
print("Acceleration: ", pro_params.accelerations[acc_ind] * 1e-3, " km/s^2")

# Start sample and amount of samples

n_ipp = measurement.cfg_params.n_ipp
start_sample = np.round(look_time * exp.sample_rate).astype(np.int64)
start_sample = (start_sample // exp.ipp_samps) * exp.ipp_samps
start_sample += measurement.rx_sample_bounds.start
delta_samples = exp.ipp_samps * n_ipp
print(f"read amount of samples: {delta_samples}, size tx_stencil = {pro_params.tx_stencil.shape}")

# Extract rx and tx data
z = measurement.extract_signals(start_sample, delta_samples)


# Calculate the time vector, filter for used and unused signals and extract il1 rx windows from the rx samples

samps = np.arange(delta_samples)
t = samps / exp.sample_rate
not_used_sig = np.full(t.shape, True, dtype=bool)
not_used_sig[pro_params.rx_stencil] = False
not_used_sig[pro_params.tx_stencil] = False
sel_rxs = z.rx[pro_params.il1_rx_window_indices + rg]
comp_samps = np.arange(z.tx.size)
nfft = np.arange(phasors.size)

# Extract the samples from the rx windows

rx_window = np.full(z.rx.shape, False, dtype=bool)
rx_window[pro_params.il1_rx_window_indices + rg] = True
not_rx_window = np.logical_not(rx_window)

# Calculate the cross correlation and echo

assert cfg_params.range_gate_sub_resolution <= 1, "Example only runs without subresolution for now"
xcorr = sel_rxs * z.tx[:, 0]
xcorr = xcorr.copy().reshape(-1, cfg_params.frequency_decimation)
echo = np.sum(xcorr, axis=-1)
c_echo = echo * phasors

# Calculate the decimated parameters

dec_rgs = np.floor(pro_params.rel_rgs / cfg_params.frequency_decimation).astype(np.int32)
dec_txlen = (exp.tx_pulse_length / exp.t_samp_usec) // cfg_params.frequency_decimation
dec_sig_samps = np.arange(pro_params.decimated_read_length)
dec_signal_vec = np.zeros((pro_params.decimated_read_length,), dtype=np.complex64)
dec_rx_window_indices = pro_params.il0_dec_rx_window_indices  # + dec_rgs
dec_signal_vec[dec_rx_window_indices] = echo
print("Decimated signal length: ", pro_params.decimated_read_length)
spec = fft.fft(dec_signal_vec)
c_dec_signal_vec = dec_signal_vec.copy()
c_dec_signal_vec[dec_rx_window_indices] = c_echo
c_spec = fft.fft(c_dec_signal_vec)

# FFT frequencay can be extracted from the range_rates

fft_freq = pro_params.range_rates / exp.wavelength
print("FFT len: ", len(fft_freq))

# Plot the matched filter

fig, axes = plt.subplots(3, 1)
axes[0].semilogy(t[not_used_sig], np.abs(z.ipp[not_used_sig]), ".k")
axes[0].semilogy(t[pro_params.rx_stencil][rx_window], np.abs(z.rx[rx_window]), ".g")
axes[0].semilogy(
    t[pro_params.rx_stencil][not_rx_window],
    np.abs(z.rx[not_rx_window]),
    ".b",
)
axes[0].semilogy(t[pro_params.tx_stencil], np.abs(z.tx), ".r")
axes[0].set_title("Raw signal power")
#
axes[1].plot(comp_samps, np.real(sel_rxs) / np.sum(np.abs(sel_rxs)), "-g", label="RX")
axes[1].plot(comp_samps, np.real(z.tx) / np.sum(np.abs(z.tx)), "-r", alpha=0.5, label="TX")
for ind in range(n_ipp):
    axes[1].axvline((exp.tx_pulse_length / exp.t_samp_usec) * (ind + 1), ls="--", c="c")
axes[1].set_title("Stenciled signals")
axes[1].legend()
#
axes[2].plot(comp_samps, np.real(xcorr), "-k")
axes[2].plot(comp_samps, np.abs(xcorr), "--k")
for ind in range(n_ipp):
    axes[2].axvline((exp.tx_pulse_length / exp.t_samp_usec) * (ind + 1), ls="--", c="c")
axes[2].set_title("Correlated echo")
fig.set_size_inches(10, 10)

fig, axes = plt.subplots(3, 2)
axes[0, 0].plot(comp_samps, np.real(xcorr), "-k")
axes[0, 0].plot(comp_samps, np.abs(xcorr), "--k")
for ind in range(n_ipp):
    axes[0, 0].axvline((exp.tx_pulse_length / exp.t_samp_usec) * (ind + 1), ls="--", c="c")
axes[0, 0].set_title("Correlated echo")
#
axes[0, 1].plot(nfft, np.real(phasors))
for ind in range(n_ipp):
    axes[0, 1].axvline(dec_txlen * (ind + 1), ls="--", c="c")
axes[0, 1].set_title("Acceleration phasors")
#
axes[1, 0].plot(nfft, np.real(echo), "-k")
axes[1, 0].plot(nfft, np.abs(echo), "--k")
for ind in range(n_ipp):
    axes[1, 0].axvline(dec_txlen * (ind + 1), ls="--", c="c")
axes[1, 0].set_title("Correlated & decimated echo")
#
axes[1, 1].plot(nfft, np.real(c_echo), "-b")
axes[1, 1].plot(nfft, np.abs(c_echo), "--b")
for ind in range(n_ipp):
    axes[1, 1].axvline(dec_txlen * (ind + 1), ls="--", c="c")
axes[1, 1].set_title("Correlated & decimated & acceleration corrected echo")
#
inds = np.abs(fft_freq) < 3e3
axes[2, 0].semilogy(fft_freq[inds], np.abs(spec[inds]), "-k")
axes[2, 0].semilogy(fft_freq, np.abs(spec), "-k")
axes[2, 0].set_title("Correlated & decimated echo spectrum")
#
axes[2, 1].semilogy(fft_freq[inds], np.abs(c_spec[inds]), "-b")
axes[2, 1].semilogy(fft_freq, np.abs(c_spec), "-b")
axes[2, 1].set_title("Correlated & decimated & acceleration corrected echo spectrum")
fig.set_size_inches(15, 10)
plt.show()
tmp_sim_path.cleanup()
