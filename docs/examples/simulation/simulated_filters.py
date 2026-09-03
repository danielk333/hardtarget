import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from tqdm import tqdm

from hardtarget.data_simulation import receiver_chain

parser = argparse.ArgumentParser(
    description="Calculate an impulse response from an EISCAT FIR definition file."
)
parser.add_argument("firpar_file", type=Path, help="path to the .fir file")
parser.add_argument(
    "p_dtau",
    type=float,
    help="time step of the interpolated response, in microseconds",
)
args = parser.parse_args()

colors = ["tab:blue", "tab:orange", "tab:green"]

impresp, t0, taps, decimation = receiver_chain.get_impresp(
    args.firpar_file,
    args.p_dtau,
    do_plot=True,
)


dc_gain = np.sum(taps)
y = -dc_gain + 2 * np.cumsum(taps)
n = np.arange(len(y))

f = interp1d(n, y, kind="linear", fill_value=(-1, 1), bounds_error=False)


t = np.linspace(0, len(y) - 1, 1000)
fig, ax = plt.subplots()
ax.plot(t, f(t))
ax.set(xlabel="Sample", ylabel="Amplitude")
ax.grid()

num = 100
offsets = np.linspace(-1, 1, num)
# given the filter group delay of 1, we can sample the transitions around the step
# and the offset is just a offset in this function?


np.random.seed(124)
sig_n = 0.1
true_offset = 0.4
samps = np.arange(3)
xi = sig_n * (np.random.randn(len(samps)) + 1j * np.random.randn(len(samps)))
f0 = f((samps - true_offset) * decimation)
syn_data = f0 + xi


def fit_transition(offset, points):
    s = f((samps - offset) * decimation)
    return np.sum((s - points) ** 2)


# we rotate it back to real plane by ~mask samples
# and y/np.mean(y[~mask]) in real data

s = np.real(syn_data)

fig, ax = plt.subplots()
ax.plot(offsets, np.array([fit_transition(x, s) for x in offsets]))

# one can use interp as an inverter! but for now we minimze
res = minimize_scalar(fit_transition, bounds=(0, 1), args=(s,))
print(res)


fig, ax = plt.subplots()
ax.plot(offsets, f((0 - offsets) * decimation), label="sample -1", c=colors[0])
ax.plot(offsets, f((1 - offsets) * decimation), label="sample 0", c=colors[1])
ax.plot(offsets, f((2 - offsets) * decimation), label="sample +1", c=colors[2])
for ind in range(len(samps)):
    ax.plot(true_offset, f0[ind], "o", c=colors[ind])
    ax.plot(true_offset, syn_data[ind], "x", c=colors[ind])
ax.axvline(res.x, c="g")
ax.legend()

lims = (0, 1)

# Monte-Carlo that stuff!
def run_est(mc_num, sig_num, true_n):
    sigs = np.linspace(0.1, 0.5, sig_num)
    true_offsets = np.linspace(lims[0], lims[1], true_n)
    pbar = tqdm(total=sig_num*true_n*mc_num)

    res_mat = np.full((len(sigs), true_n), np.nan, dtype=np.float64)
    for si, sig_n in enumerate(sigs):
        est_offset = np.empty((true_n, mc_num), dtype=np.float64)
        for ind, true_offset in enumerate(true_offsets):
            for mci in range(mc_num):
                xi = sig_n * (np.random.randn(len(samps)) + 1j * np.random.randn(len(samps)))
                f0 = f((samps - true_offset) * decimation)
                syn_data = f0 + xi
                s = np.real(syn_data)
                res = minimize_scalar(fit_transition, bounds=lims, args=(s,))
                est_offset[ind, mci] = res.x
                pbar.update()

        res_mat[si, :] = np.mean(np.abs(est_offset - true_offsets[:, None]), axis=1)
    pbar.close()
    return sigs, true_offsets, res_mat.T


sig_num = 10
true_n = 40
mcn = 500
sigs, true_offsets, res_mat = run_est(mcn, sig_num, true_n)
X, Y = np.meshgrid(sigs, true_offsets)

fig, ax = plt.subplots()
pm = ax.pcolormesh(X, Y, res_mat)
fig.colorbar(pm, ax=ax)


plt.show()
print(f"impresp = {np.array2string(np.asarray(impresp), separator=', ')}")
print(f"t0 = {t0}")
print(f"taps = {np.array2string(taps, separator=', ')}")
print(f"decimation = {decimation}")
