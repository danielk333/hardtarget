"""

Back to basics - Part 5: Ambiguity function
============================================

For completeness we here show the spectrum of the ambiguity function
(or the numerical differentiation of the phase) of a pulse train of an
accelerating signal.

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft

box_size = 6000
cycle_size = 20000
sample_rate = 1000000

wavelength = 0.3
vel0 = 0.4e3
dop0 = vel0 / wavelength


def analyse_cycle_daf(cycles, acel0, plot=False):
    ddop0 = (acel0 * 0.5) / wavelength
    signal_samps = cycles*cycle_size

    s = np.arange(signal_samps)
    t = s / sample_rate

    r = vel0*t + acel0*0.5*t**2
    phase = np.mod(r / wavelength, 1) * np.pi * 2
    signal = np.exp(1j * phase)

    boxes = signal_samps // cycle_size
    tau0 = cycle_size * (boxes // 2)

    stensil = np.full((signal_samps, ), True, dtype=bool)
    for ind in range(boxes):
        stensil[(cycle_size*ind):(cycle_size*ind + box_size)] = False

    signal[stensil] = 0

    fvec = fft.fftfreq(signal_samps, d=1.0/sample_rate)
    spec = np.abs(fft.fft(signal))

    dsignal = signal[tau0:] * np.conj(signal[:-tau0])
    # TODO: i dont remember why its 2*offset, can be seen in DAF-definition
    step = 2*tau0/sample_rate
    dsignal_samps = signal_samps - tau0

    # Divide this with the step size to get to Hz/s
    dfvec = fft.fftfreq(dsignal_samps, d=1.0/sample_rate) / step
    dspec = np.abs(fft.fft(dsignal))
    dspec_peak = np.argmax(dspec)
    dfreq_est = dfvec[dspec_peak]
    accel_est = (dfreq_est * wavelength) * 2

    # correct signal
    signal = signal * np.exp(-1j*np.pi*2*dfreq_est*t**2)
    cspec = np.abs(fft.fft(signal))
    vel_est = fvec[np.argmax(cspec)] * wavelength

    if plot:
        df = 0.6e3
        ddf = 0.6e4
        print(f"Frequency drift estimation: {dfreq_est} Hz/s (real = {ddop0})")
        print(f"Acceleration estimation: {accel_est} m/s^2 (real = {acel0})")
        fig, axes = plt.subplots(4, 1)

        axes[0].plot(t, np.real(signal), label="Signal real component")
        axes[0].set_xlabel("Time [s]")
        axes[0].legend()

        axes[1].plot(fvec, spec, ".-", label="Signal spectrum")
        axes[1].set_xlim([dop0 - df, dop0 + df])
        axes[1].axvline(dop0, c="r", ls="--", label="True doppler")
        axes[1].set_xlabel("Frequency [Hz/s]")
        axes[1].legend()

        axes[2].plot(dfvec, dspec, ".-", label=f"Lagged spectrum @ tau={tau0}")
        axes[2].axvline(ddop0, c="r", ls="--", label="True doppler drift")
        axes[2].set_xlim([ddop0 - ddf, ddop0 + ddf])
        axes[2].set_xlabel("Frequency drift [Hz/s]")
        axes[2].legend()

        axes[3].plot(fvec, cspec, ".-", label="Acceleration corrected signal spectrum")
        axes[3].axvline(dop0, c="r", ls="--", label="True doppler")
        axes[3].set_xlim([dop0 - df, dop0 + df])
        axes[3].set_xlabel("Frequency [Hz/s]")
        axes[3].legend()

    return fvec, cspec, dfvec, dspec, vel_est, accel_est


acel = -0.20e3
analyse_cycle_daf(10, acel, plot=True)


fig, axes = plt.subplots(2, 1)

n_ipps = np.arange(4, 20)
signal_times = n_ipps*cycle_size/sample_rate
res_vecs = np.empty((len(n_ipps), 2))

for ni, n_ipp in enumerate(n_ipps):
    fvec, cspec, dfvec, dspec, vel_est, accel_est = analyse_cycle_daf(n_ipp, acel)
    res_vecs[ni, 0] = vel_est
    res_vecs[ni, 1] = accel_est

    if n_ipp in [5, 10, 20]:
        axes[0].plot((dfvec * wavelength) * 2, dspec, label=f"N_ipp={n_ipp}")
        axes[1].plot(fvec * wavelength, cspec)

axes[0].axvline(acel, ls="--", c="k")
axes[0].legend()
axes[0].set_xlim([-1e3, 1e3])


axes[1].axhline(vel0, ls="--", c="k")
axes[1].set_xlim([-300 + vel0, 300 + vel0])

fig, axes = plt.subplots(2, 1)
axes[0].plot(n_ipps, res_vecs[:, 0])
axes[0].axhline(vel0, ls="--", c="r")
axes[1].plot(n_ipps, res_vecs[:, 1])
axes[1].axhline(acel, ls="--", c="r")

fig, ax = plt.subplots()

accs = np.linspace(0, 0.4e3, 100)
err_vecs = np.empty((3, len(accs)))

for ni, n_ipp in enumerate([5, 10, 20]):
    for ai in range(len(accs)):
        fvec, cspec, dfvec, dspec, vel_est, accel_est = analyse_cycle_daf(n_ipp, accs[ai])
        err_vecs[ni, ai] = accel_est - accs[ai]

    ax.plot(accs, err_vecs[ni, :], label=f"N_ipp={n_ipp}")
ax.axhline(0, ls="--", c="r")
ax.legend()

plt.show()
