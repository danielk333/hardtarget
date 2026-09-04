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
offset_limits = (0, 1)

impresp, t0, taps, decimation = receiver_chain.get_impresp(
    args.firpar_file,
    args.p_dtau,
    do_plot=True,
)


dc_gain = np.sum(taps)
y = np.real(-dc_gain + 2 * np.cumsum(taps))
n = np.arange(len(y))

f = interp1d(n, y, kind="linear", fill_value=(-1, 1), bounds_error=False)


t = np.linspace(0, len(y) - 1, 1000)
fig, ax = plt.subplots()
ax.plot(t, f(t), "-", c=colors[0], label="Interpolation")
ax.plot(n, y, "o", c=colors[0], label="Filter output")
ax.set(
    xlabel="Sample (@ input sample rate)",
    ylabel="Amplitude",
    title="Interpolated filter step response",
)
ax.legend()
ax.grid()
plt.show()

num = 100
offsets = np.linspace(-1, 1, num)

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
# but here we assume that is already done

s = np.real(syn_data)

fig, ax = plt.subplots()
ax.plot(offsets, np.array([fit_transition(x, s) for x in offsets]))
ax.axvline(true_offset, c="r", label="True offset")
ax.set(
    xlabel="Trial offset [samples]",
    ylabel="Sum of squared errors",
    title="Transition-fit objective",
)
ax.legend()
ax.grid()

# One can use interpolation as an inverter, but for now we minimize.
res = minimize_scalar(fit_transition, bounds=offset_limits, args=(s,))
print(res)


fig, ax = plt.subplots()
ax.plot(offsets, f((0 - offsets) * decimation), label="Sample -1", c=colors[0])
ax.plot(offsets, f((1 - offsets) * decimation), label="Sample 0", c=colors[1])
ax.plot(offsets, f((2 - offsets) * decimation), label="Sample +1", c=colors[2])
for ind in range(len(samps)):
    ax.plot(true_offset, f0[ind], "o", c=colors[ind], label="True offset/signal")
    ax.plot(true_offset, syn_data[ind], "x", c=colors[ind], label="Noisy signal")
ax.axvline(res.x, c="g", label="Estimated offset")
ax.set(
    xlabel="Transition offset [samples]",
    ylabel="Amplitude",
    title="Transition samples and fitted offset",
)
ax.grid()
ax.legend()


def run_est(mc_num, sig_num, true_n, all_samps=True):
    """Estimate mean absolute offset errors with a Monte Carlo simulation."""
    sigs = np.linspace(0.1, 0.5, sig_num)
    true_offsets = np.linspace(*offset_limits, true_n)
    pbar = tqdm(total=sig_num * true_n * mc_num)
    if all_samps:
        samps = np.arange(3)
    else:
        samps = np.array([1])

    def fit_transition(offset, points):
        s = f((samps - offset) * decimation)
        return np.sum((s - points) ** 2)

    res_mat = np.full((len(sigs), true_n), np.nan, dtype=np.float64)
    for si, sig_n in enumerate(sigs):
        est_offset = np.empty((true_n, mc_num), dtype=np.float64)
        for ind, true_offset in enumerate(true_offsets):
            for mci in range(mc_num):
                xi = sig_n * (
                    np.random.randn(len(samps))
                    + 1j * np.random.randn(len(samps))
                )
                f0 = f((samps - true_offset) * decimation)
                syn_data = f0 + xi
                s = np.real(syn_data)
                res = minimize_scalar(
                    fit_transition,
                    bounds=offset_limits,
                    args=(s,),
                )
                est_offset[ind, mci] = res.x
                pbar.update()

        res_mat[si, :] = np.mean(np.abs(est_offset - true_offsets[:, None]), axis=1)
    pbar.close()
    return sigs, true_offsets, res_mat.T


sig_num = 10
true_n = 40
mcn = 100
sigs_3, true_offsets_3, res_mat_3 = run_est(mcn, sig_num, true_n)
sigs_1, true_offsets_1, res_mat_1 = run_est(mcn, sig_num, true_n, all_samps=False)

color_limits = {
    "vmin": min(res_mat_3.min(), res_mat_1.min()),
    "vmax": max(res_mat_3.max(), res_mat_1.max()),
}
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, constrained_layout=True)
for ax, sigs, true_offsets, errors, title in zip(
    axes,
    (sigs_3, sigs_1),
    (true_offsets_3, true_offsets_1),
    (res_mat_3, res_mat_1),
    ("Three transition samples", "One transition sample"),
):
    mesh = ax.pcolormesh(
        sigs,
        true_offsets,
        errors,
        shading="auto",
        **color_limits,
    )
    ax.set(
        xlabel="Noise standard deviation",
        ylabel="True offset [samples]",
        title=title,
    )

fig.suptitle("Monte Carlo offset-estimation error")
fig.colorbar(mesh, ax=axes, label="Mean absolute error [samples]")


plt.show()
print(f"impresp = {np.array2string(np.asarray(impresp), separator=', ')}")
print(f"t0 = {t0}")
print(f"taps = {np.array2string(taps, separator=', ')}")
print(f"decimation = {decimation}")
