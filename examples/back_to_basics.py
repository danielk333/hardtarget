"""
The following is a simple script to demonstrate the effects of zeroing missing
data in FFTs, a concept vital to the GMF FFT approach, as well as the effect of
correcting the signal for acceleration.

# TODO: finish writing description here
# TODO: make example simpler and clean up code a bit,
#       maybe put some of the functions into hardtarget as helpers

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import scipy.constants as constants
from tqdm import tqdm


signal_len = 1000000
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


range0 = 2000e3
vel0 = 0.4e3
dop0 = vel0 / wavelength
acel0 = -0.30e3


def get_spectrum(decimation, n_ipp, plot=False, batch=False, noise_sigma=None, acc_correction=True):
    dec_sample_rate = sample_rate / decimation
    dec_tx_len = tx_len // decimation

    z_rx = np.zeros((ipp_samps*n_ipp, ), dtype=np.complex128)
    dec_z_rx = np.zeros((ipp_samps*n_ipp//decimation, ), dtype=np.complex128)
    signals = np.zeros((ipp_samps*n_ipp, ), dtype=np.complex128)
    z_tx = np.zeros((ipp_samps*n_ipp, ), dtype=np.complex128)
    ajs = np.zeros((ipp_samps*n_ipp, ), dtype=np.complex128)
    dec_ajs = np.zeros((ipp_samps*n_ipp//decimation, ), dtype=np.complex128)
    t_all = np.arange(ipp_samps*n_ipp) / sample_rate
    dec_t_all = t_all[::decimation]

    bspec = np.zeros((dec_tx_len, ), dtype=np.float64)
    bfvec = fft.fftfreq(dec_tx_len, d=1.0/dec_sample_rate)

    for ind in range(n_ipp):
        samps0 = np.arange(tx_len)
        samps = samps0 + ind*ipp_samps
        dec_samps = np.arange(tx_len//decimation) + ind*ipp_samps//decimation
        t0 = samps0 / sample_rate
        t = samps / sample_rate
        t2 = t**2
        dec_t = t[::decimation]
        dec_t2 = dec_t**2

        r = range0 + vel0*t + acel0*0.5*t**2
        phase = np.mod(r / wavelength, 1) * np.pi * 2
        tx_code = waveform_generator(t0, baud_length, frequency, code)
        signal = np.exp(1j * phase) * tx_code

        aj = np.exp(-1j * np.pi / wavelength * acel0 * t2)

        dec_aj = np.exp(-1j * np.pi / wavelength * acel0 * dec_t2)
        ajs[samps] = aj
        dec_ajs[dec_samps] = dec_aj

        echo = signal * np.conj(tx_code)

        dec_echo = np.sum(echo.reshape(-1, decimation), axis=-1)
        dec_z = dec_echo.copy()
        if acc_correction:
            dec_z *= dec_aj

        if noise_sigma is not None:
            dec_z += noise_sigma * (np.random.randn(*dec_z.shape) + 1j * np.random.randn(*dec_z.shape))
        dec_z_rx[dec_samps] = dec_z

        z = echo.copy()
        if acc_correction:
            z *= aj

        z_rx[samps] += z
        signals[samps] = signal
        z_tx[samps] = tx_code

        bspec += np.abs(fft.fft(dec_z))

    fvec = fft.fftfreq(len(dec_z_rx), d=1.0/dec_sample_rate)
    spec = np.abs(fft.fft(dec_z_rx))

    if plot:
        max_f_ind = np.argmax(spec)
        df = 5e3
        fig, axes = plt.subplots(2, 2)

        axes[0, 0].plot(t_all, np.real(z_rx), "-k")
        axes[0, 0].plot(dec_t_all, np.real(dec_z_rx), "--g")
        axes[0, 0].set_title("Doppler shifted echo")

        axes[0, 1].plot(t_all, np.real(signals), "-k")
        axes[0, 1].plot(t_all, np.real(z_tx), "--b")
        axes[0, 1].set_title("Signal & tx_code")

        axes[1, 0].semilogy(fvec, spec, "-k")
        axes[1, 0].set_title("spectrum")
        axes[1, 0].set_xlim([
            fvec[max_f_ind] - df,
            fvec[max_f_ind] + df,
        ])

        axes[1, 1].plot(t_all, np.real(ajs), "-k")
        axes[1, 1].set_title("aj")

    if batch:
        return fvec, spec, bfvec, bspec
    else:
        return fvec, spec


df = 6e3
nsig = 10.0

get_spectrum(decimation=1, n_ipp=1, plot=True)
get_spectrum(decimation=1, n_ipp=1, plot=True, noise_sigma=nsig)

get_spectrum(decimation=2, n_ipp=5, plot=True)
get_spectrum(decimation=2, n_ipp=5, plot=True, noise_sigma=nsig)


fig, axes = plt.subplots(2, 2)
decimation = 5
n_ipp = 5

# FFT bin size is sampling rate / fft size, hence the decimation cancels out
fft_bin_size = sample_rate/(ipp_samps*n_ipp)
print(f"spectral granularity={fft_bin_size} Hz ({fft_bin_size * wavelength} m/s)")
fvec, spec, bfvec, bspec = get_spectrum(decimation=decimation, n_ipp=n_ipp, batch=True)
max_f_ind = np.argmax(spec)
axes[0, 0].semilogy(fvec, spec, "-k")
axes[0, 0].axvline(dop0, c="r", ls="--")
axes[0, 0].set_title(f"dec={decimation} n_ipp={n_ipp} granularity={fft_bin_size} Hz")
axes[0, 0].set_xlim([
    fvec[max_f_ind] - df,
    fvec[max_f_ind] + df,
])
axes[0, 1].semilogy(fvec, spec, "-k")
axes[0, 1].axvline(dop0, c="r", ls="--")
axes[0, 1].set_title(f"dec={decimation} n_ipp={n_ipp}")
axes[0, 1].axvline(fvec[max_f_ind] - df, c="g", ls="--")
axes[0, 1].axvline(fvec[max_f_ind] + df, c="g", ls="--")

# Since the batches do not have zero regions
# this is true spectral resolution rather than granularity
fft_bin_size = sample_rate/tx_len
print(f"batch spectral resolution={fft_bin_size} Hz ({fft_bin_size * wavelength} m/s)")
max_f_ind = np.argmax(bspec)
axes[1, 0].semilogy(bfvec, bspec, "-k")
axes[1, 0].axvline(dop0, c="r", ls="--")
axes[1, 0].set_title(f"dec={decimation} n_ipp={n_ipp} (batch) resolution={fft_bin_size} Hz")
axes[1, 0].set_xlim([
    bfvec[max_f_ind] - df,
    bfvec[max_f_ind] + df,
])
axes[1, 1].semilogy(bfvec, bspec, "-k")
axes[1, 1].axvline(dop0, c="r", ls="--")
axes[1, 1].set_title(f"dec={decimation} n_ipp={n_ipp} batch")
axes[1, 1].axvline(bfvec[max_f_ind] - df, c="g", ls="--")
axes[1, 1].axvline(bfvec[max_f_ind] + df, c="g", ls="--")

df = 2e3

fig, axes = plt.subplots(2, 2)

fvec, spec, bfvec, bspec = get_spectrum(decimation=decimation, n_ipp=n_ipp, acc_correction=True, batch=True)
max_f_ind = np.argmax(spec)
axes[0, 0].semilogy(fvec, spec, "-k")
axes[0, 0].semilogy(fvec[max_f_ind], np.abs(spec[max_f_ind]), "og")
axes[0, 0].axvline(dop0, c="r", ls="--")
axes[0, 0].set_title(f"dec={decimation} n_ipp={n_ipp} (Acceleration corrected)")
axes[0, 0].set_xlim([
    fvec[max_f_ind] - df,
    fvec[max_f_ind] + df,
])
max_f_ind = np.argmax(bspec)
axes[0, 1].semilogy(bfvec, bspec, "-k")
axes[0, 1].semilogy(bfvec[max_f_ind], np.abs(bspec[max_f_ind]), "og")
axes[0, 1].axvline(dop0, c="r", ls="--")
axes[0, 1].set_title(f"dec={decimation} n_ipp={n_ipp} (Acceleration corrected batch)")
axes[0, 1].set_xlim([
    bfvec[max_f_ind] - df,
    bfvec[max_f_ind] + df,
])

fvec, spec, bfvec, bspec = get_spectrum(decimation=decimation, n_ipp=n_ipp, acc_correction=False, batch=True)
max_f_ind = np.argmax(spec)
axes[1, 0].semilogy(fvec, spec, "-k")
axes[1, 0].semilogy(fvec[max_f_ind], np.abs(spec[max_f_ind]), "og")
axes[1, 0].axvline(dop0, c="r", ls="--")
axes[1, 0].set_title(f"dec={decimation} n_ipp={n_ipp} (No acceleration correction)")
axes[1, 0].set_xlim([
    fvec[max_f_ind] - df,
    fvec[max_f_ind] + df,
])
max_f_ind = np.argmax(bspec)
axes[1, 1].semilogy(bfvec, bspec, "-k")
axes[1, 1].semilogy(bfvec[max_f_ind], np.abs(bspec[max_f_ind]), "og")
axes[1, 1].axvline(dop0, c="r", ls="--")
axes[1, 1].set_title(f"dec={decimation} n_ipp={n_ipp} (No acceleration correction batch)")
axes[1, 1].set_xlim([
    bfvec[max_f_ind] - df,
    bfvec[max_f_ind] + df,
])


# Estimate batch vs regular error in prepense of noise
mc = 400
dpeaks = np.zeros((mc, 4), dtype=np.float64)
for mci in tqdm(range(mc), total=mc):
    vel0 = np.random.randn(1)*0.1e3 + 0.2e3
    dop0 = vel0 / wavelength

    fvec, spec, bfvec, bspec = get_spectrum(
        decimation=10, n_ipp=5, noise_sigma=nsig, acc_correction=True, batch=True,
    )
    dpeaks[mci, 0] = fvec[np.argmax(spec)]
    dpeaks[mci, 1] = bfvec[np.argmax(bspec)]
    fvec, spec, bfvec, bspec = get_spectrum(
        decimation=10, n_ipp=5, noise_sigma=nsig, acc_correction=False, batch=True,
    )
    dpeaks[mci, 2] = fvec[np.argmax(spec)]
    dpeaks[mci, 3] = bfvec[np.argmax(bspec)]

    dpeaks[mci, :] -= dop0


print(f"Peak error (accel. corr.): mu={np.mean(dpeaks[:, 0])} Hz sigma={np.std(dpeaks[:, 0])} Hz")
print(f"Batch peak error (accel. corr.): mu={np.mean(dpeaks[:, 1])} Hz sigma={np.std(dpeaks[:, 1])} Hz")

print(f"Peak error (no accel. corr.): mu={np.mean(dpeaks[:, 2])} Hz sigma={np.std(dpeaks[:, 2])} Hz")
print(f"Batch peak error (no accel. corr.): mu={np.mean(dpeaks[:, 3])} Hz sigma={np.std(dpeaks[:, 3])} Hz")


df = 3e3

fig, axes = plt.subplots(3, 4)

for yi, ipps in enumerate([1, 2, 5]):
    for xi, dec in enumerate([1, 5, 10, 20]):
        fvec, spec = get_spectrum(decimation=dec, n_ipp=ipps)
        max_f_ind = np.argmax(spec)

        axes[yi, xi].semilogy(fvec, spec, "-k")
        axes[yi, xi].set_title(f"dec={dec} n_ipp={ipps}")
        axes[yi, xi].set_xlim([
            fvec[max_f_ind] - df,
            fvec[max_f_ind] + df,
        ])


# Determining acceleration
range0 = 2000e3
vel0 = 0.4e3
dop0 = vel0 / wavelength
acel0 = -0.30e3


def search_accel(decimation, n_ipp, accels, noise_sigma=None):
    dec_sample_rate = sample_rate / decimation
    peak_dop = np.empty_like(accels)
    peak = np.empty_like(accels)
    bpeak_dop = np.empty_like(accels)
    bpeak = np.empty_like(accels)

    for ai, acc in enumerate(accels):

        dec_z_rx = np.zeros((ipp_samps*n_ipp//decimation, ), dtype=np.complex128)
        dec_ajs = np.zeros((ipp_samps*n_ipp//decimation, ), dtype=np.complex128)

        dec_tx_len = tx_len // decimation
        bspec = np.zeros((dec_tx_len, ), dtype=np.float64)
        bfvec = fft.fftfreq(dec_tx_len, d=1.0/dec_sample_rate)

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

            echo = signal * np.conj(tx_code)

            dec_echo = np.sum(echo.reshape(-1, decimation), axis=-1)
            dec_z = dec_echo.copy()

            if noise_sigma is not None:
                dec_z += noise_sigma * (np.random.randn(*dec_z.shape) + 1j * np.random.randn(*dec_z.shape))

            dec_z *= dec_aj

            dec_z_rx[dec_samps] = dec_z
            bspec += np.abs(fft.fft(dec_z))

        fvec = fft.fftfreq(len(dec_z_rx), d=1.0/dec_sample_rate)
        spec = np.abs(fft.fft(dec_z_rx))
        si = np.argmax(spec)
        peak_dop[ai] = fvec[si]
        peak[ai] = spec[si]

        bsi = np.argmax(bspec)
        bpeak_dop[ai] = bfvec[bsi]
        bpeak[ai] = bspec[bsi]

    return peak_dop, peak, bpeak_dop, bpeak


accels = np.linspace(-0.4e3, 0.4e3, 300)

fig, axes = plt.subplots(2, 2)

for n_ipp in [1, 2, 5, 10]:
    peak_dop, peak, bpeak_dop, bpeak = search_accel(5, n_ipp, accels, noise_sigma=None)

    axes[0, 0].plot(accels, peak_dop, label=f"N_ipp={n_ipp}")
    axes[0, 1].plot(accels, peak)

    axes[1, 0].plot(accels, bpeak_dop)
    axes[1, 1].plot(accels, bpeak)

axes[0, 0].axhline(dop0, ls="--", c="k")
axes[1, 0].axhline(dop0, ls="--", c="k")
axes[0, 1].axvline(acel0, ls="--", c="k")
axes[1, 1].axvline(acel0, ls="--", c="k")

axes[0, 0].legend()


peak_dop, peak, bpeak_dop, bpeak = search_accel(5, 5, accels, noise_sigma=nsig)

fig, axes = plt.subplots(2, 2)
axes[0, 0].plot(accels, peak_dop)
axes[0, 1].plot(accels, peak)

axes[1, 0].plot(accels, bpeak_dop)
axes[1, 1].plot(accels, bpeak)

# Estimate acceleration error as a function of N-ipp
mc = 20
dpeaks = np.zeros((mc, 4), dtype=np.float64)
for mci in tqdm(range(mc), total=mc):
    acel0 = np.random.randn(1)*0.2e3

    for ii, n_ipp in enumerate([1, 2, 5, 10]):
        peak_dop, peak, bpeak_dop, bpeak = search_accel(5, n_ipp, accels, noise_sigma=nsig)
        dpeaks[mci, ii] = accels[np.argmax(peak)]
    dpeaks[mci, :] -= acel0

for ii, n_ipp in enumerate([1, 2, 5, 10]):
    print(f"Acceleration error @ {n_ipp} IPPs: mu={np.mean(dpeaks[:, ii])} m/s^2 sigma={np.std(dpeaks[:, ii])} m/s^2")


plt.show()
