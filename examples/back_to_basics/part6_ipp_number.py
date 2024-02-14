"""

# TODO: finish writing description here

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import scipy.constants as constants

sample_rate = 1000000


ipp_samps = 20000
baud_length = 40*16/sample_rate
code_len = 8
tx_len = np.round(baud_length * sample_rate * code_len).astype(np.int64)

code = np.random.randint(0, high=2, size=(code_len,))
code[code == 0] = -1
wavelength = 0.3
frequency = constants.c / wavelength


def waveform_generator(t, baud_length, frequency, code):
    t_ind = (t // baud_length).astype(np.int64)
    carrier = np.zeros(t.shape, dtype=np.complex128)
    inds = np.logical_and(t >= 0, t <= baud_length * len(code))
    carrier[inds] = code[t_ind[inds]]
    return carrier


# Determining acceleration
range0 = 2000e3
vel0 = 0.4e3
dop0 = vel0 / wavelength
acel0 = -0.20e3


def search_accel(decimation, n_ipp, accels, noise_sigma=None):
    dec_sample_rate = sample_rate / decimation
    peak_dop = np.empty_like(accels)
    peak = np.empty_like(accels)

    for ai, acc in enumerate(accels):

        dec_z_rx = np.zeros((ipp_samps*n_ipp//decimation, ), dtype=np.complex128)
        dec_ajs = np.zeros((ipp_samps*n_ipp//decimation, ), dtype=np.complex128)

        for ind in range(n_ipp):
            samps0 = np.arange(tx_len)
            samps = samps0 + ind*ipp_samps
            dec_samps = np.arange(tx_len//decimation) + ind*ipp_samps//decimation
            t0 = samps0 / sample_rate
            t = samps / sample_rate

            dec_t = t[::decimation]
            dec_t2 = dec_t**2

            r = range0 + vel0*t + acel0*0.5*t**2
            phase = np.mod(r / wavelength, 1) * np.pi * 2
            tx_code = waveform_generator(t0, baud_length, frequency, code)
            signal = np.exp(1j * phase) * tx_code

            dec_aj = np.exp(-1j * np.pi / wavelength * acc * dec_t2)
            dec_ajs[dec_samps] = dec_aj

            if noise_sigma is not None:
                signal += noise_sigma * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))

            echo = signal * np.conj(tx_code)

            dec_echo = np.sum(echo.reshape(-1, decimation), axis=-1)
            dec_z = dec_echo.copy()
            dec_z *= dec_aj

            dec_z_rx[dec_samps] = dec_z

        fvec = fft.fftfreq(len(dec_z_rx), d=1.0/dec_sample_rate)
        spec = np.abs(fft.fft(dec_z_rx))
        si = np.argmax(spec)
        peak_dop[ai] = fvec[si]
        peak[ai] = spec[si]

    return peak_dop, peak


signal_ipps = np.arange(2, 20)
signal_times = signal_ipps * ipp_samps / sample_rate

freq_res = 1.0 / signal_times
freq_drift_res = 1.0 / signal_times**2

vel_res = wavelength * freq_res
accel_res = 2 * wavelength * freq_drift_res

# TODO axis labels and clean up
fig, axes = plt.subplots(2, 1)
axes[0].semilogy(signal_ipps, 0.5*vel_res, label="Frequency resolution")
axes[1].semilogy(signal_ipps, 0.5*accel_res, label="Acceleration resolution")
for ax in axes:
    ax.set_xticks(signal_ipps)
    ax.grid(True)
plt.show()

accels = np.linspace(-0.3e3, 0.3e3, 300)

for ns in [None, 1.0e1]:

    fig, axes = plt.subplots(2, 1)

    for n_ipp in [1, 2, 5, 10]:
        peak_dop, peak = search_accel(5, n_ipp, accels, noise_sigma=ns)

        axes[0].plot(accels, peak_dop, label=f"N_ipp={n_ipp}")
        axes[1].plot(accels, peak)

    axes[0].axhline(dop0, ls="--", c="k")
    axes[1].axvline(acel0, ls="--", c="k")
    axes[0].legend()


plt.show()