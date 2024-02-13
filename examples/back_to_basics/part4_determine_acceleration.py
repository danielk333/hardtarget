"""

Back to basics - Part 4: Determine Acceleration
================================================

We get the FFT bin-width by dividing with the total signal time, this is
convenient if the sampling rate is constant, in the resolution can also be
descibed by is `freq_res = sample_rate / signal_samps` which we can see is
equal to `1 / signal_times`. Then the frequencies examined are this frequency
[-freq_res*signal_samps/2, freq_res*signal_samps/2]. Or understund from the
point of view of the Nyquist sampling theorem, [-sample_rate/2, sample_rate/2].

In a sense it is intuitive that the acceleration resolution we can reach is
`freq_drift_res = 1.0 / signal_times**2` since the phase change is dependant on
the square of the time. To actually prove this we can use what is called the
discrete polynomial transform or other similar forms such as the discrete
ambiguity function. They all rely on the concept of multiplying the signal with
a conjugated version of itself at an delay, this delay and multiplication
essentially creates a numerical differentiation of the phase. Consider e.g.

$$
e^{i\theta(t_j)} e^{-i\theta(t_{j - \tau})}
    = e^{i(\theta(t_j) - \theta(t_{j - \tau}))},
$$

This is the numerical differentiation of the phase at a lag of $\tau$ samples.
A single frequency signal, such as a doppler shift, can be seen as a constant
change of phase which gives a single peak in its frequency spectrum, if we have
a constant change of the numerical differentiation, i.e. the doppler drift, and
fourier transform this lagged signal will give a single peak in its spectrum at
the frequency of this drift. Of course this can be done recursively taking lags
of the lag to find higher order numerical differentiations (the basic concept of
the discrete polynomial transform), here we will assume we will only ever need
to go to doppler drift.

In essence, the discrete fourier transform is a numerical differentiation of the
phase using the step size equal to the sampling frequency. Considering the
formula for a numerical differentiation

$$
\dot{f}(x) = \lim_{h \mapsto 0} \frac{f(x + h) - f(x)}{h},
$$

we can see that the second order numerical differentiation is

$$
\ddot{f}(x) = \lim_{h \mapsto 0} \frac{f(x + 2h) - 2f(x + h) + f(x)}{h^2}.
$$

Thinking of $h$ as the resolution, or `1 / signal_times`, it is clear that the
resolution of the second order spectrum of doppler drift will be
`1 / signal_times**2`. However, this is if we use the differentiation on the
entire signal. Forming the lagged product at $\tau$ of a finite signal will
reduce the number of available samples, and hence also the resolution and signal
strength of the estimation if we try to estimate the doppler drift purely trough
this measure. Instead it can be used as a rough estimate for a later estimation
using the full Maximum Likelihood Estimator function to achieve optimal SNR in
the parameter estimations.

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import scipy.constants as constants
from tqdm import tqdm

sig = 0

sample_rate = 100000
n_sample_times = 100
signal_times = np.linspace(0.02, 0.1, n_sample_times)
signal_samps = (signal_times*sample_rate).astype(np.int64)
wavelength = 0.3
frequency = constants.c / wavelength

max_acel = 1.0e3
vel0 = 0.3e3
dop0 = vel0 / wavelength

specs = np.empty((n_sample_times, ), dtype=np.float64)
specs_f = np.empty((n_sample_times, ), dtype=np.float64)
specs_a = np.empty((n_sample_times, ), dtype=np.float64)
acel_res = np.empty((n_sample_times, ), dtype=np.float64)

freq_res = 1.0 / signal_times
freq_drift_res = 1.0 / signal_times**2

# Motion model: r0 + v0*t + 0.5*a0*t^2
vel_res = wavelength * freq_res
accel_res = 2 * wavelength * freq_drift_res

for si in tqdm(range(n_sample_times), total=n_sample_times):
    signal_time = signal_times[si]
    n_samps = signal_samps[si]

    s = np.arange(n_samps)
    t = s / sample_rate

    fvec = fft.fftshift(fft.fftfreq(n_samps, d=1.0/sample_rate))

    accel_res_rad = np.pi/4
    max_phase_change = np.pi / wavelength * (2*max_acel) * t[-1]**2
    n_accel = np.ceil(max_phase_change/accel_res_rad).astype(np.int64)
    acel_res[si] = 2*max_acel/n_accel
    daccs = np.linspace(-max_acel, max_acel, n_accel)

    r = vel0*t[:, None] + daccs[None, :]*0.5*t[:, None]**2
    phase = np.mod(r / wavelength, 1) * np.pi * 2

    xi = (np.random.randn(n_samps) + 1j*np.random.randn(n_samps))*sig
    signal = np.exp(1j * phase) + xi[:, None]
    spec = fft.fftshift(np.abs(fft.fft(signal, axis=0)), axes=0)
    ainds = np.arange(n_accel)

    max_spec_ind = np.argmax(spec, axis=0)
    max_spec = spec[max_spec_ind, ainds]
    max_spec_f = fvec[max_spec_ind]

    a_max = np.argmax(max_spec)

    specs[si] = max_spec[a_max] / signal_time
    specs_f[si] = max_spec_f[a_max] - dop0
    specs_a[si] = daccs[a_max]


fig, axes = plt.subplots(3, 1)
ax = axes[0]
ax.plot(signal_times, specs)
ax.set_xlabel("Signal size [s]")
ax.set_title("Peak intensity")

# We use half the acceleration as the upper bound since without noise
# we will only every be half a bin off

ax = axes[1]
ax.plot(signal_times, specs_f, label="Peak frequency error")
ax.plot(signal_times, 0.5*freq_res, c="r", ls="--", label="Frequency resolution")
ax.set_xlabel("Signal size [s]")
ax.set_ylabel("Frequency [Hz]")
ax.set_title("Peak frequency error")
ax.legend()

ax = axes[2]
ax.plot(signal_times, specs_a, label="Peak acceleration error")
ax.plot(signal_times, 0.5*accel_res, c="r", ls="--", label="Acceleration resolution")
ax.legend()
ax.set_xlabel("Signal size [s]")
ax.set_ylabel("Acceleration [m/s^2]")
ax.set_title("Peak acceleration error")

# TODO: add peak width height things

plt.show()
