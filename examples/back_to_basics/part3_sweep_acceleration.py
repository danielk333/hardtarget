"""

Back to basics - Part 3: Sweep acceleration
============================================

Compensating a signal $S$ for a guessed acceleration $a$ simplifies to

$$
S(t) = e^{i\frac{2\pi}{\lambda}\left ( r_0 + v_0 t + 0.5 a_0 t^2 \right )} \\
S_{comp}(t) = S(t) e^{-i\frac{2\pi}{\lambda}\left ( 0.5 a_0 t^2 \right )} = \\
    = e^{i\frac{2\pi}{\lambda}\left ( r_0 + v_0 t + 0.5 (a_0 - a) t^2 \right )}.
$$

Here we can introduce a new variable $\delta a$ which is 0 when we have found
the correct acceleration and only a regular doppler shifted signal remains.

Below we investigate if sweeping this $\delta a$ and extracting the FFT peak can
be used to determined the acceleration. If there is a defined peak at
$\delta a = 0$, then such a sweep will determine the correct acceleration,
otherwise we will find an erroneous acceleration using such a sweep.

Here we see that the peak for a short signal length is quite far displaced from
the true acceleration while the peak for a longer signal length is much closer
to the true value. This is basically because the resolution by which we can
determine the acceleration is not the same as the frequency resolution.
More on that in the next part.

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import scipy.constants as constants

fig, axes = plt.subplots(2, 1)

sample_rate = 1000000
wavelength = 0.3
frequency = constants.c / wavelength

vel0 = 0.4e3
dop0 = vel0 / wavelength
max_acel = 0.3e4
n_accel = 300

df = 1e3

daccs = np.arange(-max_acel, max_acel, 2*max_acel/n_accel)
daccs0_ind = n_accel//2


for signal_time in [0.02, 0.2]:
    signal_samps = int(signal_time*sample_rate)
    fvec = fft.fftshift(fft.fftfreq(signal_samps, d=1.0/sample_rate))

    s = np.arange(signal_samps)
    t = s / sample_rate
    r = vel0*t[:, None] + daccs[None, :]*0.5*t[:, None]**2
    phase = np.mod(r / wavelength, 1) * np.pi * 2

    # Do the FFT for all accelerations
    signal = np.exp(1j * phase)
    spec = fft.fftshift(np.abs(fft.fft(signal, axis=0)), axes=0)

    # Now pick the maximum for each for each acceleration
    mspec_ind = np.argmax(spec, axis=0)
    mspec = spec[mspec_ind, np.arange(n_accel)]
    mspec = mspec / signal_time  # Normalize by signal length
    mspec_f = fvec[mspec_ind]
    mspec_a_ind = np.argmax(mspec)

    fig_ex, axes_ex = plt.subplots(2, 1)
    axes_ex[0].plot(
        fvec, spec[:, daccs0_ind], c="k",
        label=f"true delta acceleration {daccs[daccs0_ind]} m/s^2"
    )
    axes_ex[0].plot(
        fvec, spec[:, mspec_a_ind], c="b",
        label=f"peak delta acceleration {daccs[mspec_a_ind]} m/s^2"
    )
    axes_ex[0].set_xlabel("Frequency [Hz]")
    axes_ex[0].set_xlim([
        dop0 + df,
        dop0 - df,
    ])
    axes_ex[0].set_title(f"Signals with signal time {signal_time} s")
    axes_ex[0].legend()
    axes_ex[1].plot(t, np.real(signal[:, daccs0_ind]), c="k")
    axes_ex[1].plot(t, np.real(signal[:, mspec_a_ind]), c="b")
    axes_ex[1].set_xlabel("Time [s]")

    axes[0].plot(daccs, mspec, label=f"Signal length = {signal_time} s")
    axes[1].plot(daccs, mspec_f)


axes[0].axvline(0, ls="--", c="r", label="True value")
axes[0].legend()
axes[0].set_ylabel("FFT Peak intensity")
axes[0].set_xlabel("$\delta a$ Acceleration difference [m/s^2]")

axes[1].axhline(dop0, ls="--", c="r", label="True doppler")
axes[1].set_ylabel("FFT Peak frequency [Hz]")
axes[1].set_xlabel("$\delta a$ Acceleration difference [m/s^2]")


plt.show()
