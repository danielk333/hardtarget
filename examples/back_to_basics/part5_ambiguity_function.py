"""

Back to basics - Part 5: Ambiguity function
============================================

For completeness we here show the spectrum of the ambiguity function
(or the numerical differentiation of the phase) of a pulse train of an
accelerating signal.

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import scipy.constants as constants


sample_rate = 1000000
signal_samps = 200000
wavelength = 0.3
frequency = constants.c / wavelength

vel0 = 0.4e3
dop0 = vel0 / wavelength
omega0 = dop0 * 2 * np.pi
acel0 = -0.20e3
ddop0 = (acel0 * 0.5) / wavelength
domega0 = ddop0 * 2 * np.pi

s = np.arange(signal_samps)
t = s / sample_rate

r = vel0*t + acel0*0.5*t**2
phase = np.mod(r / wavelength, 1) * np.pi * 2
signal = np.exp(1j * phase)

box_size = 6000
cycle_size = 20000
boxes = signal_samps // cycle_size
tau0 = cycle_size * (boxes // 2)

stensil = np.full((signal_samps, ), True, dtype=bool)
for ind in range(boxes):
    stensil[(cycle_size*ind):(cycle_size*ind + box_size)] = False

signal[stensil] = 0

fvec = fft.fftfreq(signal_samps, d=1.0/sample_rate)
spec = np.abs(fft.fft(signal))

dsignal = signal[tau0:] * np.conj(signal[:-tau0])
# TODO: i dont remember why its 2*offset, can be seen in DAF-definition
step = 2*tau0/sample_rate
dsignal_samps = signal_samps - tau0

# Divide this with the step size to get to Hz/s
dfvec = fft.fftfreq(dsignal_samps, d=1.0/sample_rate) / step
dspec = np.abs(fft.fft(dsignal))
dspec_peak = np.argmax(dspec)
dfreq_est = dfvec[dspec_peak]
accel_est = (dfreq_est * wavelength) * 2

print(f"Frequency drift estimation: {dfreq_est} Hz/s (real = {ddop0})")
print(f"Acceleration estimation: {accel_est} m/s^2 (real = {acel0})")

# correct signal
signal = signal * np.exp(-1j*np.pi*2*dfreq_est*t**2)
cspec = np.abs(fft.fft(signal))

df = 0.6e3
ddf = 0.6e4

fig, axes = plt.subplots(4, 1)

axes[0].plot(t, np.real(signal), label="Signal real component")
axes[0].set_xlabel("Time [s]")
axes[0].legend()

axes[1].plot(fvec, spec, ".-", label="Signal spectrum")
axes[1].set_xlim([dop0 - df, dop0 + df])
axes[1].axvline(dop0, c="r", ls="--", label="True doppler")
axes[1].set_xlabel("Frequency [Hz/s]")
axes[1].legend()

axes[2].plot(dfvec, dspec, ".-", label=f"Lagged spectrum @ tau={tau0}")
axes[2].axvline(ddop0, c="r", ls="--", label="True doppler drift")
axes[2].set_xlim([ddop0 - ddf, ddop0 + ddf])
axes[2].set_xlabel("Frequency drift [Hz/s]")
axes[2].legend()

axes[3].plot(fvec, cspec, ".-", label="Acceleration corrected signal spectrum")
axes[3].axvline(dop0, c="r", ls="--", label="True doppler")
axes[3].set_xlim([dop0 - df, dop0 + df])
axes[3].set_xlabel("Frequency [Hz/s]")
axes[3].legend()

plt.show()
