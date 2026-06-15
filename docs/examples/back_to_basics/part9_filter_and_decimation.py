# #Back to basics - Part 9: Filter and decimation
# ---

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parameters
fs = 40e6  # original sampling rate: 40 MHz
fs_dec = 1e6  # target sampling rate: 1 MHz
decim = int(fs / fs_dec)

pulse_time = 80e-6
baud_time = 4e-6
offset_samples = 50
# doppler = -150e3
doppler = 0

pulse_samples = int(pulse_time * fs)
total_samples = offset_samples + pulse_samples

t = np.arange(total_samples) / fs
t_pulse = np.arange(pulse_samples) / fs

# Initialize signal with zeros
x = np.zeros(total_samples, dtype=complex)

# BPSK chip sequence
num_bauds = int(pulse_time / baud_time)
bits = np.random.choice([-1, 1], size=num_bauds)

baud_idx = np.floor(t_pulse / baud_time).astype(int)
baud_idx = np.clip(baud_idx, 0, num_bauds - 1)

bpsk_phase = bits[baud_idx]

# Analytical complex baseband signal with negative Doppler shift
pulse = bpsk_phase * np.exp(1j * 2 * np.pi * doppler * t_pulse)

x[offset_samples:offset_samples + pulse_samples] = pulse

# Low-pass FIR filter
# Cutoff must be below new Nyquist rate = 500 kHz
cutoff = 400e3
numtaps = decim + 1  # filter order approximately same scale as decimation
fir = signal.firwin(numtaps, cutoff, fs=fs)

# Filter before decimation
x_filt = signal.lfilter(fir, 1.0, x)
x_filt_dec = x_filt[::decim]

# Decimate without filtering, intentionally aliased
x_dec_no_filter = x[::decim]

t_dec = t[::decim]


plt.figure(figsize=(12, 7))
plt.plot(t * 1e6, np.abs(x))
plt.plot(t_dec * 1e6, np.abs(x_filt_dec))
plt.plot(t_dec * 1e6, np.abs(x_dec_no_filter))

# Plot real parts
plt.figure(figsize=(12, 7))

plt.plot(t * 1e6, np.real(x), label="Original BPSK signal, 40 MHz", alpha=0.6)
plt.plot(t_dec * 1e6, np.real(x_filt_dec), "o-", label="Filtered + decimated to 1 MHz")
plt.plot(t_dec * 1e6, np.real(x_dec_no_filter), "x--", label="Decimated without filtering")

plt.xlabel("Time [µs]")
plt.ylabel("Amplitude, real part")
plt.title("BPSK pulse: original, filtered-decimated, and naïvely decimated")
plt.legend()
plt.tight_layout()
plt.show()

