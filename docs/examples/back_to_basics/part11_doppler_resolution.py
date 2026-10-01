# #Back to basics - Part 11: Doppler resolution
# ---
# The FFT bin is not the end-all-be-all of resolution, and the Nyquist sampling theorem gives the upper
# limit on frequency, but does not dictate the accuracy of frequencies lower than this.

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft

# Parameters
signal_len = 100000
sample_rate = 1000000
box_size = 6000
cycle_size = 20000
sub_resolution = 1000
boxes = signal_len // cycle_size
doppler = 1.046e3

np.random.seed(233442)


def sim_signal(noise: bool = False) -> None:

    signal = np.zeros((signal_len,), dtype=np.complex128)
    stensil = np.full((signal_len,), True, dtype=bool)
    t = np.arange(signal_len) / sample_rate

    for ind in range(boxes):
        stensil[(cycle_size * ind) : (cycle_size * ind + box_size)] = False

    signal = np.exp(2j * np.pi * doppler * t)

    boxcar_stenciled_signal = signal.copy()
    boxcar_stenciled_signal[stensil] = 0

    if noise:
        sigma = 1
        xi = (np.random.randn(len(t)) + 1j * np.random.randn(len(t))) * sigma
        boxcar_stenciled_signal += xi

    fvec = fft.fftshift(fft.fftfreq(signal_len, d=1.0 / sample_rate))
    boxcar_stenciled_spec = np.abs(fft.fftshift(fft.fft(boxcar_stenciled_signal)))

    peak_i = np.argmax(boxcar_stenciled_spec)
    width = 2

    # DTFT method
    nums = np.arange(len(boxcar_stenciled_signal))
    fvec_sub = np.linspace(fvec[peak_i - width], fvec[peak_i + width], sub_resolution)
    dt = 1.0 / sample_rate
    dtft_fractors = np.exp(-1j * 2 * np.pi * fvec_sub[:, None] * dt * nums[None, :])
    dtft = np.abs(np.sum(dtft_fractors * boxcar_stenciled_signal[None, :], axis=1))
    sub_peak_i = np.argmax(dtft)

    err_dtft = np.abs(fvec_sub[sub_peak_i] - doppler)
    err_fft = np.abs(fvec[peak_i] - doppler)

    df = 100.0
    df_w = 5e3
    fig, axes = plt.subplots(2, 2, layout="tight")
    #
    axes[0, 0].plot(t, np.real(boxcar_stenciled_signal), "-k")
    axes[0, 0].set_title("Boxcar signal train")
    #
    axes[1, 0].semilogy(fvec, boxcar_stenciled_spec, "-k", label="FFT")
    axes[1, 0].plot(fvec[peak_i], boxcar_stenciled_spec[peak_i], "or", label=f"FFT Max {fvec[peak_i]:.1f} Hz")
    axes[1, 0].axvline(doppler, ls="--", c="m", label=f"True Doppler {doppler:.1f} Hz")
    axes[1, 0].set_title("FFT Signal spectrum | Zoomed")
    axes[1, 0].set_xlim([doppler - df, doppler + df])
    axes[1, 0].legend()
    #
    axes[0, 1].semilogy(fvec_sub, dtft, "-k", label="DTFT")
    axes[0, 1].plot(
        fvec_sub[sub_peak_i],
        dtft[sub_peak_i],
        "or",
        label=f"DTFT Max {fvec_sub[sub_peak_i]:.1f} Hz",
    )
    axes[0, 1].plot(
        fvec[(peak_i - width) : (peak_i + width + 1)],
        boxcar_stenciled_spec[(peak_i - width) : (peak_i + width + 1)],
        "xb",
        ls="-.",
        label="FFT",
    )
    axes[0, 1].axvline(doppler, ls="--", c="m", label=f"True Doppler {doppler:.1f} Hz")
    axes[0, 1].set_title(f"DTFT Signal spectrum | error {err_dtft:.2e} Hz")
    axes[0, 1].legend()
    #
    axes[1, 1].semilogy(fvec, boxcar_stenciled_spec, "-k", label="FFT")
    axes[1, 1].plot(fvec[peak_i], boxcar_stenciled_spec[peak_i], "or", label=f"FFT Max {fvec[peak_i]:.1f} Hz")
    axes[1, 1].axvline(doppler, ls="--", c="m", label=f"True Doppler {doppler:.1f} Hz")
    axes[1, 1].set_title(f"FFT Signal spectrum | error {err_fft:.2e} Hz")
    axes[1, 1].set_xlim([-df_w, df_w])
    #
    fig.set_size_inches(15, 10)


sim_signal(noise=False)
sim_signal(noise=True)
plt.show()
