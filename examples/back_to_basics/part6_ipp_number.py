"""

# TODO: finish writing description here

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import scipy.constants as constants
from tqdm import tqdm
sample_rate = 1000000


ipp_samps = 20000
baud_length = 15*16/sample_rate
code_len = 8
T_tx_samps = np.round(baud_length * sample_rate * code_len).astype(np.int64)

wavelength = 0.3
frequency = constants.c / wavelength


def waveform_generator(t, baud_length, code):
    t_ind = (t // baud_length).astype(np.int64)
    carrier = np.zeros(t.shape, dtype=np.complex128)
    inds = np.logical_and(t >= 0, t <= baud_length * len(code))
    carrier[inds] = code[t_ind[inds]]
    return carrier


# Determining acceleration
range0 = 2000e3
range_gate0 = np.round((range0 / constants.c) * sample_rate).astype(np.int64)


def search_accel(decimation, n_ipp, accels, acel0, vel0, noise_sigma=None):
    dec_sample_rate = sample_rate / decimation
    peak_dop = np.empty_like(accels)
    peak = np.empty_like(accels)

    z_rx = np.zeros((ipp_samps*n_ipp, ), dtype=np.complex64)
    dec_z_rx = np.zeros((ipp_samps*n_ipp//decimation, ), dtype=np.complex64)

    window_inds = np.concatenate([
        np.arange(T_tx_samps) + ind*ipp_samps
        for ind in range(n_ipp)
    ])
    dec_window_inds = np.concatenate([
        np.arange(T_tx_samps//decimation) + ind*ipp_samps//decimation
        for ind in range(n_ipp)
    ])

    t_tx = np.arange(T_tx_samps) / sample_rate

    for ind in range(n_ipp):
        signal = np.zeros((ipp_samps,), dtype=np.complex64)
        samp0 = ind * ipp_samps
        t0 = samp0 / sample_rate

        code = np.random.randint(0, high=2, size=(code_len,))
        code[code == 0] = -1
        tx_wave = waveform_generator(t_tx, baud_length, code)

        signal[0:T_tx_samps] += tx_wave

        r0 = range0 + vel0*t0 + 0.5*acel0*t0**2
        v0 = vel0 + acel0*t0
        a0 = acel0

        rg0 = np.round((r0 / constants.c) * sample_rate).astype(np.int64)
        rg_samp0 = rg0

        ranges = r0 + v0*t_tx + 0.5*a0*t_tx**2
        phase = np.mod(ranges / wavelength, 1) * np.pi * 2

        rx_wave = tx_wave * np.exp(1j * phase)

        if noise_sigma is not None:
            rx_wave += noise_sigma * (np.random.randn(*rx_wave.shape) + 1j * np.random.randn(*rx_wave.shape))

        signal[rg_samp0:(rg_samp0 + T_tx_samps)] += rx_wave
        z_rx[(ind * ipp_samps):((ind + 1) * ipp_samps)] = signal

    for ai, acc in enumerate(accels):
        ztx = z_rx[window_inds]
        zrx = z_rx[window_inds + range_gate0]

        tx_pwr = np.sum(np.abs(ztx) ** 2.0)
        tx_amp = np.sqrt(tx_pwr)
        ztx = np.conj(ztx) / tx_amp

        t = window_inds/sample_rate
        dec_t = t[::decimation]
        dec_t2 = dec_t**2

        echo = zrx * ztx
        dec_echo = np.sum(echo.reshape(-1, decimation), axis=-1)

        dec_aj = np.exp(-1j * np.pi / wavelength * acc * dec_t2)
        dec_echo *= dec_aj

        dec_z_rx[dec_window_inds] = dec_echo

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

accels = np.linspace(-0.3e3, 0.3e3, 300)
acel = -0.20e3
vel = 0.4e3
dop0 = vel / wavelength

for ns in [None, 3.0]:

    fig, axes = plt.subplots(2, 1)

    for n_ipp in [1, 2, 5, 10]:
        peak_dop, peak = search_accel(16, n_ipp, accels, acel, vel, noise_sigma=ns)

        axes[0].plot(accels, peak_dop, label=f"N_ipp={n_ipp}")
        axes[1].plot(accels, peak)

    axes[0].axhline(dop0, ls="--", c="k")
    axes[1].axvline(acel, ls="--", c="k")
    axes[0].legend()

cs = ["b", "c", "g"]
fig, ax = plt.subplots()

accs = np.linspace(0, 0.2e3, 20)
err_vecs_acc = np.empty((3, len(accs)))
pbar = tqdm(total=err_vecs_acc.size)
for ni, n_ipp in enumerate([5, 10, 20]):
    for ai in range(len(accs)):
        peak_dop, peak = search_accel(16, n_ipp, accels, accs[ai], vel, noise_sigma=None)
        err_vecs_acc[ni, ai] = accels[np.argmax(peak)] - accs[ai]
        pbar.update(1)
    signal_times = n_ipp * ipp_samps / sample_rate
    freq_drift_lim = 1.0 / signal_times**2
    ax.axhline(-freq_drift_lim*0.5, ls="--", c=cs[ni])
    ax.axhline(freq_drift_lim*0.5, ls="--", c=cs[ni])
    ax.plot(accs, err_vecs_acc[ni, :], c=cs[ni], label=f"N_ipp={n_ipp}")
pbar.close()

ax.axhline(0, ls="--", c="r")
ax.set_xlabel("True acceleration [m/s^2]")
ax.set_ylabel("Found acceleration error [m/s^2]")
ax.legend()


fig, ax = plt.subplots()

vels = np.linspace(0, 1e3, 20)
err_vecs_vel = np.empty((3, len(vels)))
pbar = tqdm(total=err_vecs_vel.size)
for ni, n_ipp in enumerate([5, 10, 20]):
    for vi in range(len(vels)):
        peak_dop, peak = search_accel(16, n_ipp, accels, acel, vels[vi], noise_sigma=None)
        err_vecs_vel[ni, vi] = accels[np.argmax(peak)] - acel
        pbar.update(1)
    signal_times = n_ipp * ipp_samps / sample_rate
    freq_drift_lim = 1.0 / signal_times**2
    ax.axhline(-freq_drift_lim*0.5, ls="--", c=cs[ni])
    ax.axhline(freq_drift_lim*0.5, ls="--", c=cs[ni])
    ax.plot(vels, err_vecs_vel[ni, :], c=cs[ni], label=f"N_ipp={n_ipp}")
pbar.close()

ax.axhline(0, ls="--", c="r")
ax.set_xlabel("True velocity [m/s^2]")
ax.set_ylabel("Found acceleration error [m/s^2]")
ax.legend()

plt.show()
