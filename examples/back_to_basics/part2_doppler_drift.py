"""

Back to basics - Part 2: Doppler drift
======================================

When the target is accelerating, not only a doppler shift is applied to the
signal but also a doppler drift. This drift broadens and shifts the frequency
spectrum of the signal across the frequencies that the acceleration changes
the doppler frequency across.

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import scipy.constants as constants

sample_rate = 1000000
signal_samps = 100000
wavelength = 0.3
frequency = constants.c / wavelength

vel0 = 0.4e3
dop0 = vel0 / wavelength
acel0 = -0.30e3

s = np.arange(signal_samps)
t = s / sample_rate

r0 = vel0*t
r = vel0*t + acel0*0.5*t**2

phase0 = np.mod(r0 / wavelength, 1) * np.pi * 2
phase = np.mod(r / wavelength, 1) * np.pi * 2

signal0 = np.exp(1j * phase0)
signal = np.exp(1j * phase)

fvec = fft.fftfreq(signal_samps, d=1.0/sample_rate)

spec0 = np.abs(fft.fft(signal0))
spec = np.abs(fft.fft(signal))

sp0_max = np.argmax(spec0)
sp_max = np.argmax(spec)

df = 1e3

fig, ax = plt.subplots()
ax.plot(fvec, spec, ls="-", c="b", label="Accelerating target spectrum")
ax.plot(fvec, spec0, ls="-", c="k", label="Non-accelerating target spectrum")
ax.set_xlim([
    fvec[sp_max] + df,
    fvec[sp_max] - df,
])
ax.axvline(dop0, ls="--", c="r", label="Initial doppler shift")
ax.set_xlabel("Frequency [Hz]")
ax.legend()

plt.show()
