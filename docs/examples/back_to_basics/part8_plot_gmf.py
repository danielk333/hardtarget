# #Back to basics - Part 8: GMF plots
# ---

# TODO: more text

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as constants
import scipy.fft as fft
from tqdm import tqdm

enable_progress = False  # Running locally a progress bar is appreciated

box_size = 6000
cycle_size = 20000
sample_rate = 1000000
signal_samps = [75000, 150000]
# signal_samps = [100000]
wavelength = 0.3
frequency = constants.c / wavelength

vlim = [0.35e3, 0.45e3]
alim = [-0.1e3, 0.0e3]

vel0 = 0.4e3
dop0 = vel0 / wavelength
acel0 = -0.05e3

da = 0.20e3 * 2 / 300
v_res = 300
a_res = int((alim[1] - alim[0]) / da)
res = [v_res, a_res]


def gen_signal(size):
    s = np.arange(size)
    t = s / sample_rate

    r = vel0 * t + acel0 * 0.5 * t**2
    phase = np.mod(r / wavelength, 1) * np.pi * 2

    stensil = np.full((size,), True, dtype=bool)
    boxes = size // cycle_size

    for ind in range(boxes):
        stensil[(cycle_size * ind) : (cycle_size * ind + box_size)] = False

    s = np.exp(1j * phase)
    bss = s.copy()
    bss[stensil] = 0
    return t, bss


def gmf(t, signal, v, a, progress):
    t2 = t**2
    ret = np.zeros_like(v)
    total = range(a_res)
    if progress:
        pbar = tqdm(total=len(total))
    for ai in range(a_res):
        for vi in range(v_res):
            r = v[ai, vi] * t + a[ai, vi] * 0.5 * t2
            phase = np.mod(r / wavelength, 1) * np.pi * 2
            model = np.exp(1j * phase)
            ret[ai, vi] = np.abs(np.sum(signal * np.conj(model), axis=0))
        if progress:
            pbar.update(1)
    if progress:
        pbar.close()
    return ret


def fft_gmf(t, signal, a, progress):
    t2 = t**2
    fvec = fft.fftshift(fft.fftfreq(len(t), d=1.0 / sample_rate))
    ret = np.zeros((len(a), len(fvec)), dtype=np.float64)
    total = len(a)
    if progress:
        pbar = tqdm(total=total)
    for ind in range(total):
        r = a[ind] * 0.5 * t2
        phase = np.mod(r / wavelength, 1) * np.pi * 2
        model = np.exp(1j * phase)
        z = signal * np.conj(model)
        spec = fft.fftshift(np.abs(fft.fft(z, axis=0)), axes=0)
        ret[ind, :] = np.abs(spec)
        if progress:
            pbar.update(1)
    if progress:
        pbar.close()
    return fvec, ret


fig, axes = plt.subplots(2, 2)
#
for si, sign_size in enumerate(signal_samps):
    t, signal = gen_signal(sign_size)
    #
    vvec = np.linspace(vlim[0], vlim[1], res[0])
    avec = np.linspace(alim[0], alim[1], res[1])
    vels, accs = np.meshgrid(
        vvec,
        avec,
    )
    #
    G = gmf(t, signal, vels, accs, enable_progress)
    fvec, G2 = fft_gmf(t, signal, avec, enable_progress)
    #
    vels2, accs2 = np.meshgrid(
        fvec * wavelength,
        avec,
    )
    inds = np.logical_and(
        fvec * wavelength < np.max(vvec),
        fvec * wavelength > np.min(vvec),
    )
    #
    ax = axes[si, 0]
    ax.pcolormesh(vels, accs, G)
    ax.axhline(acel0, c="black")
    ax.axvline(vel0, c="black")
    ax.set_xlabel("Velocity [m/s]")
    ax.set_ylabel("Acceleration [m/s^2]")
    ax.set_title(f"Full GMF - {sign_size}")
    #
    ax = axes[si, 1]
    ax.pcolormesh(vels2[:, inds], accs2[:, inds], G2[:, inds])
    ax.axhline(acel0, c="black")
    ax.axvline(vel0, c="black")
    ax.set_xlabel("Velocity [m/s]")
    ax.set_ylabel("Acceleration [m/s^2]")
    ax.set_title(f"Fast GMF - {sign_size}")
fig.set_size_inches(10, 10)

plt.show()
