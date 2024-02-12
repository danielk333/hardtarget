"""

Back to basics - Part 1: Boxcar
===============================

First, to understand the effect of running an FFT on several consecutive
received pulses that have been placed into an isochronal array with zeros where
no signal was received we can look at the FT of a boxcar function
(commonly called `rect`),

$$
\mathcal{F}\left [rect \left ( \frac{t - D}{W} \right ) \right ] (w)
    \propto e^{i T w}( \theta(w + D) - \theta(w - D) ) \frac{\sin(0.5 D w)}{w}
$$

where $D$ is the size of the box and $T$ the temporal offset.

We can here see that several boxes offset in time would introduce high frequency
modes in a beat between the offset value and the boxcar size in the output FT.
Since the FT of a multiplication is a convolution, the boxcars multiplied by
the signal will cause the dirac delta signal FT to be convolved with these high
frequency sinc-function modes, as we will see in the below plot. However, as
this $sinc$ function is convolved with a dirac delta the sinc-function peak will
simply be shifted to the dirac delta location,

$$
\mathcal{F}\left [rect \left ( \frac{t - D}{W} \right ) e^{-i \omega t} \right ] (w)
    \propto ( \theta(w + D) - \theta(w - D) ) e^{-i T (w - \omega)}
            \frac{\sin(0.5 D (w - \omega))}{w - \omega}
$$

and a correct peak location can still be determined.

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft

signal_len = 100000
sample_rate = 1000000
box_size = 6000
cycle_size = 20000
boxes = signal_len // cycle_size
frequency = 1e3

signal = np.zeros((signal_len, ), dtype=np.complex128)
stensil = np.full((signal_len, ), True, dtype=bool)
t = np.arange(signal_len) / sample_rate

for ind in range(boxes):
    stensil[(cycle_size*ind):(cycle_size*ind + box_size)] = False

signal = np.exp(2j * np.pi * frequency * t)
boxcar_stenciled_signal = signal.copy()
boxcar_stenciled_signal[stensil] = 0

fvec = fft.fftfreq(signal_len, d=1.0/sample_rate)
spec = np.abs(fft.fft(signal))
boxcar_stenciled_spec = np.abs(fft.fft(boxcar_stenciled_signal))

peak_i = np.argmax(spec)
bpeak_i = np.argmax(boxcar_stenciled_spec)


df = 5e3
fig, axes = plt.subplots(2, 2)

axes[0, 0].plot(t, np.real(signal), "-k")
axes[0, 0].set_title("Signal")

axes[0, 1].plot(t, np.real(boxcar_stenciled_signal), "-k")
axes[0, 1].set_title("Boxcar stenciled signal")

axes[1, 0].semilogy(fvec, spec, "-k")
axes[1, 0].plot(fvec[peak_i], spec[peak_i], "or")
axes[1, 0].set_title(f"Spectrum | peak = {fvec[peak_i]:.2e} Hz")
axes[1, 0].set_xlim([frequency - df, frequency + df])

axes[1, 1].semilogy(fvec, boxcar_stenciled_spec, "-k")
axes[1, 1].plot(fvec[bpeak_i], boxcar_stenciled_spec[bpeak_i], "or")
axes[1, 1].set_title(f"Boxcar stenciled spectrum | peak = {fvec[bpeak_i]:.2e} Hz")
axes[1, 1].set_xlim([frequency - df, frequency + df])

plt.show()
