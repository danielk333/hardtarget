# # Simulated Tx signal
# ---
# In the cases when there is no tx signal available the tx signal can be modeled. From the metadata of the
# measurement the tx signal code is available, using this we can model the signal.

import numpy as np
from matplotlib import pyplot as plt
from radardef.radar_stations.eiscat.experiments import load_radar_code

from hardtarget.constants import ReceiverChainModel
from hardtarget.data_simulation.tx_model import tx_signal_model

# First we define a simple code
# ```
#   ‾‾‾‾‾|  |‾‾| |‾| |‾
#        |__|  |_| |_|
# ```

code = load_radar_code("leo_bpark")[0, :]
fir_filter = ReceiverChainModel.b414d15_gaus
t_samp_usec = 1
baud_length_usec = 30
decimation = 15

# Lets say we have a ipp with 56 samples, we want to read all of them and we start at 0. We should then get
# a signal similar to the one we inserted, but decimated. The source of the signal has a baud length of 12 us,
# but we want to sample the signal at 6 us, thus we will oversample it.

tx_samples = len(code) * int(baud_length_usec / t_samp_usec)
tx = tx_signal_model(
    code=code,
    baud_length_usec=baud_length_usec,
    t_samp_usec=t_samp_usec,
    start_samp=0,
    ipp_samps=tx_samples,
    read_length=tx_samples,
    filt=fir_filter,
    bandwidth=1e6,
)

fig, ax = plt.subplots()
(ls,) = ax.plot(np.real(tx))
ax.plot(np.imag(tx), c=ls.get_color(), ls="--")

# Now in some cases we want to increase the resolution of the signal, to do this *sub_resolution* is
# introduced. With sub resolution alternative tx signals are simulated for the samples between the "real"
# samples. In the example below we set the subresolution to 4, meaning we will get additional data from each
# 0.25 decimal inbetween each sample:

example_sub_resolution = np.linspace(0, 1, num=4)
tx = tx_signal_model(
    code=code,
    baud_length_usec=baud_length_usec,
    t_samp_usec=t_samp_usec,
    start_samp=tx_samples * 0.5,
    ipp_samps=tx_samples * 10,
    read_length=tx_samples * 2,
    sub_resolution=example_sub_resolution,
    filt=fir_filter,
    bandwidth=1e6,
)
fig, axes = plt.subplots(2, 1)
for ind in range(tx.shape[1]):
    axes[0].plot(np.real(tx[:, ind]), label=f"Sub step: {ind}")
    axes[1].plot(np.real(tx[:, ind]))

axes[0].legend()
axes[1].set_xlim((tx_samples * 0.5, tx_samples * 0.5 + baud_length_usec / t_samp_usec * 1.5))

"""
# We can also model the phase flip behaviour directly
values, offsets = phase_flip_model(filt=fir_filter, sample_offset=2)

fig, ax = plt.subplots()
ax.plot(offsets, np.real(values[:, -1]), label="Index -1")
ax.plot(offsets, np.real(values[:, 0]), label="Index 0")
ax.plot(offsets, np.real(values[:, 1]), label="Index +1")
ax.set_xlabel("Decimated sample offset [1]")
ax.set_ylabel("Transition sample real-amplitude [1]")
ax.grid()
ax.legend()


# We can also solve for the offset

base_resolution = constants.c * t_samp_usec * 1e-6 / decimation
decimated_resolution = constants.c * t_samp_usec * 1e-6
true_offset = example_sub_resolution[2]
samps = np.arange(tx.shape[0])
sub_resolution = np.linspace(0, 1, num=decimation)


def simulate_and_match(noise_sigma, offset):
    tx = tx_signal_model(
        code=code,
        baud_length_usec=baud_length_usec,
        t_samp_usec=t_samp_usec,
        start_samp=tx_samples * 0.5,
        ipp_samps=tx_samples * 10,
        read_length=tx_samples * 2,
        sub_resolution=np.array([offset]),
        filt=fir_filter,
        bandwidth=None,
    )
    xi = noise_sigma * (np.random.randn(tx.shape[0]) + 1j * np.random.randn(tx.shape[0]))
    noisy_signal = np.exp(1j * 0.23) * tx[:, 0] + xi

    tx_match = tx_signal_model(
        code=code,
        baud_length_usec=baud_length_usec,
        t_samp_usec=t_samp_usec,
        start_samp=tx_samples * 0.5,
        ipp_samps=tx_samples * 10,
        read_length=tx_samples * 2,
        sub_resolution=sub_resolution,
        filt=fir_filter,
        bandwidth=None,
        normalize=True,
    )
    mu = np.mean(noisy_signal)
    sig = np.std(noisy_signal)
    noisy_signal = (noisy_signal - mu) / sig

    match = np.abs(np.sum(noisy_signal[:, None] * np.conj(tx_match), axis=0))
    best_match = np.argmax(match)
    return match, best_match, tx_match, noisy_signal


match, best_match, tx_match, noisy_signal = simulate_and_match(0, true_offset)
proper_match = np.argmin(np.abs(sub_resolution - true_offset))

fig, axes = plt.subplots(3, 1)
axes[0].plot(samps, np.real(noisy_signal))
axes[1].plot(sub_resolution, match)
axes[1].axvline(true_offset, c="r")
axes[1].axvline(sub_resolution[best_match], c="g")
axes[2].plot(samps, np.real(noisy_signal), "-b")
axes[2].plot(samps, np.real(tx_match[:, best_match]), "-r")
axes[2].plot(samps, np.real(tx_match[:, proper_match]), "--k")


# And we can estimate the accuracy of such an inversion

noise_sigmas = np.linspace(0, 1, 20)
# noise_sigmas = np.linspace(0, 1, 1)
errors = np.zeros((len(noise_sigmas), 2))
mc_samples = 30
pbar = tqdm(desc="MC err", total=len(noise_sigmas) * mc_samples)
for ni, ns in enumerate(noise_sigmas):
    offsets = np.random.rand(mc_samples)
    mc_errs = np.zeros((mc_samples,), dtype=np.float64)
    for mci in range(mc_samples):
        _, best_match, _, _ = simulate_and_match(ns, offsets[mci])
        mc_errs[mci] = offsets[mci] - sub_resolution[best_match]
        pbar.update(1)
    errors[ni, 0] = np.mean(np.abs(mc_errs))
    errors[ni, 1] = np.std(np.abs(mc_errs))
pbar.close()

errors *= t_samp_usec * constants.c * 1e-6
fig, ax = plt.subplots()
ax.semilogy(noise_sigmas, errors[:, 0])
ax.semilogy(noise_sigmas, errors[:, 0] + errors[:, 1], ls="--")
ax.axhline(base_resolution, c="g")
ax.axhline(decimated_resolution, c="r")

"""
plt.show()

# This is used during the analysis part for unknown tx signals, the subresolution is configurable in the .ini
# file, more details can be found at [Config parameters](../stuff/config_parameters.md).

# def obj_func(offset):
#     sub_resolution = np.array([offset])
#     tx_match = tx_signal_model(
#         code=code,
#         baud_length_usec=baud_length_usec,
#         t_samp_usec=t_samp_usec,
#         start_samp=tx_samples * 0.5,
#         ipp_samps=tx_samples * 10,
#         read_length=tx_samples * 2,
#         sub_resolution=sub_resolution,
#         filt=fir_filter,
#         bandwidth=1e6,
#     )
#     match = np.abs(np.sum(noisy_signal * np.conj(tx_match[:, 0])))
#     return -match
#
#
# res = minimize_scalar(obj_func, bounds=(0, 1))
