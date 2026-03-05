import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import scipy.signal


def sample_analysis(noise_sigma, sample_rate, pulse_length, frequency, range0, vel0, padding=0.5):
    wavelength = const.c / frequency

    tx_samps = int(pulse_length * sample_rate)
    size = int(tx_samps * (1 + padding))
    signal = noise_sigma * (np.random.randn(size) + 1j * np.random.randn(size))
    t_tx = np.arange(tx_samps) / sample_rate

    rg = (range0 / scipy.constants.c) * sample_rate
    rg0 = np.ceil(rg).astype(np.int64)

    code = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.float64)
    baud_length = pulse_length / len(code)
    t_ind = (t_tx // baud_length).astype(np.int64)
    tx_wave = np.zeros(t_tx.shape, dtype=signal.dtype)
    inds = np.logical_and(t_tx >= 0, t_tx <= baud_length * len(code))
    tx_wave[inds] = code[t_ind[inds]]

    ranges = 2 * (range0 + vel0 * t_tx)
    dop0 = vel0 * 2 / wavelength
    phase = np.mod(ranges / wavelength, 1) * np.pi * 2

    rx_wave = tx_wave * np.exp(1j * phase)
    s0 = (size - tx_samps) // 2
    s_true = s0 + rg - rg0
    t_sig = (np.arange(size) - s0) / sample_rate
    signal[s0 : (s0 + tx_samps)] += rx_wave

    delta_f = sample_rate / len(rx_wave)
    delta_vel = delta_f * const.c / (2 * frequency)
    print(delta_vel)
    # delta_f = 50.0
    delta_f /= 1000
    dop_min = vel0 * 0.9 * 2 * frequency / const.c
    dop_max = vel0 * 1.1 * 2 * frequency / const.c
    check_freqs = np.arange(dop_min, dop_max, delta_f)
    max_amp = np.zeros((3, len(check_freqs)), dtype=np.float64)

    for ind in range(len(check_freqs)):
        sim_signal = tx_wave * np.exp(2j * np.pi * t_tx * check_freqs[ind])
        corr = scipy.signal.correlate(signal, sim_signal, mode="same")
        abs_corr = np.abs(corr)
        max_corr = np.argmax(abs_corr)
        lags = scipy.signal.correlation_lags(signal.size, tx_wave.size, mode="same")
        lag = lags[max_corr]
        max_amp[0, ind] = lag
        max_amp[1, ind] = np.abs(corr[max_corr])
        max_amp[2, ind] = np.angle(corr[max_corr])

        # fig, axes = plt.subplots(3, 1)
        # axes[0].plot(t_sig, np.real(signal))
        # axes[1].plot(t_tx, np.real(sim_signal))
        # axes[2].plot(lags, np.abs(corr))
        # plt.show()

    # fig, ax = plt.subplots()
    # ax.plot(check_freqs, np.abs(max_amp[1, :]))
    # ax.axvline(dop0, c="r")
    # plt.show()

    best_dop = np.argmax(max_amp[1, :])
    dop = check_freqs[best_dop]
    vel = dop * const.c / (2 * frequency)
    phase0 = max_amp[2, best_dop]
    lag = max_amp[0, best_dop]

    return lag, s_true, vel, vel0, phase0, max_amp[1, best_dop]


if __name__ == "__main__":
    samples = 1_00
    # samples = 1
    np.random.seed(2387)

    pulse_lengths = [0.1e-3, 1e-3, 10e-3]
    mean_errs = np.zeros((2, len(pulse_lengths)), dtype=np.float64)
    for mind, pl in enumerate(pulse_lengths):
        res = np.zeros((3, samples), dtype=np.float64)
        sr = 100_000
        for ind in range(samples):
            lag, lag0, vel, vel0, phase0, amp = sample_analysis(
                noise_sigma=0.001,
                sample_rate=sr,
                pulse_length=pl,
                frequency=50e6,
                range0=np.random.rand() * (120e3 - 70e3) + 70e3,
                vel0=np.random.rand() * (60e3 - 20e3) + 20e3,
            )
            # print(lag, lag0, vel, vel0, phase0, amp)
            res[0, ind] = (lag - lag0) / sr * const.c
            res[1, ind] = vel - vel0
            res[2, ind] = phase0
        mean_err = np.mean(np.abs(res[:2, :]), axis=1)
        print(mean_err)

        bins = int(np.sqrt(samples))
        fig, axes = plt.subplots(3, 1)
        axes[0].hist(res[0, :], bins=bins)
        axes[1].hist(res[1, :], bins=bins)
        axes[2].hist(res[2, :], bins=bins)
        plt.show()

    sample_rates = [100_000, 500_000, 1_000_000]
    mean_errs = np.zeros((2, len(sample_rates)), dtype=np.float64)
    for mind, sr in enumerate(sample_rates):
        res = np.zeros((3, samples), dtype=np.float64)
        for ind in range(samples):
            lag, lag0, vel, vel0, phase0, amp = sample_analysis(
                noise_sigma=0.001,
                sample_rate=sr,
                pulse_length=2e-3,
                frequency=50e6,
                range0=np.random.rand() * (120e3 - 70e3) + 70e3,
                vel0=np.random.rand() * (60e3 - 20e3) + 20e3,
            )
            # print(lag, lag0, vel, vel0, phase0, amp)
            res[0, ind] = (lag - lag0) / sr * const.c
            res[1, ind] = vel - vel0
            res[2, ind] = phase0
        mean_errs[:, mind] = np.mean(np.abs(res[:2, :]), axis=1)
        print(mean_errs[:, mind])

        bins = int(np.sqrt(samples))
        fig, axes = plt.subplots(3, 1)
        axes[0].hist(res[0, :], bins=bins)
        axes[1].hist(res[1, :], bins=bins)
        axes[2].hist(res[2, :], bins=bins)
        plt.show()
