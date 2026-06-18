# #Back to basics - Part 10: Monochromatic pulse
# ---
# There are many methods to determine the frequency of a monochromatic pulse.
#
# The Fourier transform of a monochromatic pulse,
# $$
#   z(t) = exp(i (2 \pi f_0 t + \phi)) \forall 0 < t < T,
# $$
#
# is
#
# $$
#   Z(f) = T exp(-i (\pi (f - f_0) T - \phi)) \frac{\sin(\pi(f - f_0)T)}{\pi(f - f_0)T},
# $$
#
# Using this we can create an equation for the true frequency f_0 based on what the FFT
# output would be of the function above. TODO: put in ref to paper here

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.fft as fft

# TODO: we might need to put this function in a different location
from hardtarget.target_estimation.gmf.gmf_numpy import dft_taylor, dtft_solve

np.random.seed(233442)

# Parameters
T = 0.1
sample_rate = 100_000
n0 = int(T * sample_rate)
f0 = 1.046e3
phi0 = np.pi / 4

t = np.arange(n0) / sample_rate

noise_sigmas = 10 ** np.linspace(-3, 1, 20)
monte_carlo_samples = 500

errs = np.zeros((3, len(noise_sigmas), monte_carlo_samples))
for ind, sigma in tqdm(enumerate(noise_sigmas), total=len(noise_sigmas)):
    for mci in range(monte_carlo_samples):
        xi = 1j * np.random.randn(n0) + np.random.randn(n0)
        signal = np.exp(1j * (2 * np.pi * f0 * t + phi0)) + xi * sigma

        fft_len = 2 ** (int(np.log2(n0)) + 2)
        spec = fft.fftshift(fft.fft(signal, n=fft_len))
        freqs = fft.fftshift(fft.fftfreq(fft_len, d=1.0 / sample_rate))
        spec_max = np.argmax(np.abs(spec))
        d_freq = freqs[spec_max] - freqs[spec_max - 1]

        f_est_dtft = dtft_solve(
            dec_signal=signal,
            sample_rate=sample_rate,
            freq_bracket=(
                freqs[spec_max] - d_freq,
                freqs[spec_max] + d_freq,
            ),
        )

        try:
            f_est, phi_est = dft_taylor(
                spectrum=spec,
                signal_len=n0,
                sample_rate=sample_rate,
            )
        except RuntimeError:
            f_est, phi_est = np.nan, np.nan
        errs[0, ind, mci] = f_est - f0
        errs[1, ind, mci] = phi_est - phi0
        errs[2, ind, mci] = f_est_dtft - f0


fig, axes = plt.subplots(2, 1)

mu_f_dtft_err = np.nanmean(np.abs(errs[2, :, :]), axis=1)
std_f_dtft_err = np.nanstd(np.abs(errs[2, :, :]), axis=1)
mu_f_err = np.nanmean(np.abs(errs[0, :, :]), axis=1)
std_f_err = np.nanstd(np.abs(errs[0, :, :]), axis=1)
mu_phi_err = np.nanmean(np.abs(errs[1, :, :]), axis=1)
std_phi_err = np.nanstd(np.abs(errs[1, :, :]), axis=1)

axes[0].loglog(noise_sigmas, mu_f_err, c="k")
axes[0].loglog(noise_sigmas, mu_f_err + std_f_err * 3, ls="--", alpha=0.5, c="k")
axes[0].loglog(noise_sigmas, mu_f_dtft_err, c="b")
axes[0].loglog(noise_sigmas, mu_f_dtft_err + std_f_err * 3, ls="--", alpha=0.5, c="b")
axes[0].set_ylabel("Frequency error [Hz]")
axes[1].loglog(noise_sigmas, mu_phi_err)
axes[1].loglog(noise_sigmas, mu_phi_err + std_phi_err * 3, ls="--", alpha=0.5, c="k")
axes[1].set_xlabel("Noise standard deviation [1]")
axes[1].set_ylabel("Phase error [rad]")

plt.show()
