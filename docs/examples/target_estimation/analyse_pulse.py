"""Analyse the range and range rate of one radar pulse.

The measured echo is matched against sub-sample transmit-pulse templates. A
precision orbit file is used to get the expected range and range rate for comparison.
"""

from __future__ import annotations

import argparse
import datetime as dt
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import radardef
from matplotlib import pyplot as plt
from radardef.types import BeamType, EiscatUHFLocation
from radardef.types.types import ExpDef
from scipy import constants
from scipy.fft import fft, fftfreq, fftshift
from spacecoords import interpolation
from tqdm import tqdm

from hardtarget import plotting
from hardtarget.constants import ReceiverChainModel
from hardtarget.data_simulation.tx_model import match_pulse_code, tx_modulation_model, tx_signal_model
from hardtarget.plotting.raw_data_plots import extract_requested_range_gates
from hardtarget.process.utils import sample_interval_to_closest_ipp
from hardtarget.target_estimation.dtft_solvers import dtft_solve
from hardtarget.utils.time_conversion import time_interval_to_sample_bound, ts_from_str

DEFAULT_START_TIME = "2024-07-04T10:21:17.500"
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"


@dataclass(frozen=True)
class OrbitReference:
    """Expected two-way range and range rate at one IPP."""

    r: float
    v: float


@dataclass(frozen=True)
class PulseEstimate:
    """Best matched-filter result for one pulse."""

    r: float
    v: float
    frequency: float
    power: npt.NDArray[np.float64]
    frequencies: npt.NDArray[np.float64]
    range_gates: npt.NDArray[np.float64]
    template: npt.NDArray[np.complex128]
    echo: npt.NDArray[np.complex128]
    template_index: int
    range_index: int


@dataclass(frozen=True)
class PulseData:
    """Signals and indices retained for diagnostic plots."""

    signal: npt.NDArray[np.complex128]
    rx: npt.NDArray[np.complex128]
    tx: npt.NDArray[np.complex128]
    templates: npt.NDArray[np.complex128]
    measured_templates: npt.NDArray[np.complex128]
    range_start: int
    tx_start: int
    tx_end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_file", type=Path, help="Radar data file")
    parser.add_argument("orbit_file", type=Path, help="Sentinel precision orbit EOF file")
    parser.add_argument("--start-time", default=DEFAULT_START_TIME, help="UTC ISO timestamp")
    parser.add_argument("--duration", type=float, default=2.5, help="Analysis interval in seconds")
    parser.add_argument("--ipp-index", type=int, default=14, help="IPP relative to the analysis interval")
    parser.add_argument("--offset", type=int, default=0, help="Raw-data sample alignment offset")
    parser.add_argument("--min-range-gate", type=int, default=5500)
    parser.add_argument("--max-range-gate", type=int, default=7700)
    parser.add_argument("--sub-resolution", type=int, default=15, help="Templates per range sample")
    parser.add_argument("--radar-frequency-mhz", type=float, default=927.2)
    parser.add_argument(
        "--template",
        choices=("model", "measured"),
        default="model",
        help="Use an analytic or interpolated measured transmit pulse",
    )
    # TODO: for some reason my "predicted" code does not match??? look at the tlan files or
    # something to figure out how to predict the code
    parser.add_argument(
        "--code-source",
        choices=("detected", "sequence"),
        default="detected",
        help="Detect the pulse code or take it from the experiment sequence",
    )
    parser.add_argument("--rti", action="store_true", help="Include a range-time-intensity plot")
    parser.add_argument(
        "--diagnostics",
        choices=("summary", "full"),
        default="full",
        help="Plot the summary only or all signal/template diagnostics",
    )
    parser.add_argument("--output-dir", type=Path, help="Save figures to this directory")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive figures")
    return parser.parse_args()


def parse_datetime(value: str) -> dt.datetime:
    return dt.datetime.strptime(value, DATETIME_FORMAT).replace(tzinfo=dt.timezone.utc)


def load_orbit(path: Path, start: dt.datetime, end: dt.datetime) -> interpolation.Legendre8:
    """Load orbit states surrounding the requested observation."""
    epochs: list[float] = []
    states: list[list[float]] = []
    for osv in ET.parse(path).getroot().findall(".//OSV"):
        utc = osv.findtext("UTC")
        if utc is None:
            continue
        epoch = parse_datetime(utc.removeprefix("UTC="))
        if start <= epoch <= end:
            epochs.append(epoch.timestamp())
            states.append([float(osv.findtext(key, "nan")) for key in ("X", "Y", "Z", "VX", "VY", "VZ")])

    if len(epochs) < 9:
        raise ValueError(f"Orbit file contains only {len(epochs)} usable states around the observation")
    return interpolation.Legendre8(states=np.asarray(states, dtype=np.float64).T, t=np.asarray(epochs))


def orbit_references(
    radar: radardef.EiscatUHF,
    orbit: interpolation.Legendre8,
    epochs: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """TODO: this does a really janky "light-time" correction to account for the actuall
    scattering time - it might play a role but still not sure. Investigate more.
    """

    def measurements(enu: npt.NDArray[np.float64]) -> tuple[npt.NDArray, npt.NDArray]:
        one_way_range = np.linalg.norm(enu, axis=0)
        one_way_rate = -np.sum(enu[:3] * enu[3:], axis=0) / one_way_range
        return 2 * one_way_range, 2 * one_way_rate

    state = orbit.get_state(epochs)
    enu = radar.enu(state)
    ranges, _ = measurements(enu)
    state = orbit.get_state(epochs + 0.5 * ranges / constants.c)
    return measurements(radar.enu(state))


def select_code(
    tx: npt.NDArray[np.complex128],
    exp_def: ExpDef,
    sequence_index: int,
    source: str,
) -> int:
    """Select the code used to construct the matching template."""
    sequence_index %= exp_def.code.shape[0]
    if source == "sequence":
        return sequence_index
    matches = match_pulse_code(
        tx_signal=tx,
        codes=exp_def.code,
        baud_length_usec=exp_def.baud_length_usec,
        t_samp_usec=exp_def.t_samp_usec,
        ipp_t_usec=exp_def.ipp_samps * exp_def.t_samp_usec,
    )
    detected = int(np.argmax(matches))
    print(f"Code index: {detected} detected, {sequence_index} expected from sequence")
    return detected


def make_templates(
    tx: npt.NDArray[np.complex128],
    exp_def: ExpDef,
    code_index: int,
    count: int,
    source: str,
) -> npt.NDArray[np.complex128]:

    if source == "measured":
        return tx_modulation_model(
            tx,
            np.ones(tx.size, dtype=bool),
            sub_resolution=count,
            filt=ReceiverChainModel.b414d15_gaus,
        )
    return tx_signal_model(
        code=exp_def.code[code_index],
        baud_length_usec=exp_def.baud_length_usec,
        t_samp_usec=exp_def.t_samp_usec,
        ipp_samps=exp_def.ipp_samps,
        read_length=tx.size,
        sub_resolution=count,
        bandwidth=None,
        filt=ReceiverChainModel.b414d15_gaus,
        # normalize=True,
    )


def estimate_pulse_phase_flips(
    rx: npt.NDArray[np.complex128],
    impulse_response,
    detected_range_gate: int,
):
    """Search range offsets and estimate Doppler at every offset."""
    # The steps to do this is:
    # 1. estimate range gate
    # 2. estimate doppler by decoding
    # 3. remove doppler, estimate phase flip locations `mask`
    # 4. remove constant phase by doing signal/mean(signal[~mask])
    # 5. invert phase flip signals to step-function offset
    # 6. fit linear function to phase flip ranges
    raise NotImplementedError()


def estimate_pulse(
    rx: npt.NDArray[np.complex128],
    templates: npt.NDArray[np.complex128],
    min_range_gate: int,
    max_range_gate: int,
    sample_rate: float,
    carrier: float,
) -> PulseEstimate:
    """Search range offsets and estimate Doppler at every offset."""
    count = templates.shape[1]
    base_gates = np.arange(min_range_gate, max_range_gate - templates.shape[0])
    range_gates = np.arange(min_range_gate, max_range_gate - templates.shape[0], 1.0 / count)
    fft_length = 2 ** (int(np.log2(templates.shape[0])) + 2)
    fft_frequencies = fftshift(fftfreq(fft_length, d=1.0 / sample_rate))
    bin_width = fft_frequencies[1] - fft_frequencies[0]
    power = np.empty(range_gates.size, dtype=np.float64)
    frequencies = np.empty_like(power)

    with tqdm(total=range_gates.size, desc="Matching pulse", unit="template") as progress:
        for range_index, _ in enumerate(base_gates):
            echo = rx[range_index : range_index + templates.shape[0]]
            for offset_index in range(count):
                index = range_index * count + offset_index
                decoded = echo * np.conj(templates[:, offset_index])
                spectrum = fftshift(fft(decoded, n=fft_length))
                peak = int(np.argmax(np.abs(spectrum)))
                power[index], frequencies[index], _ = dtft_solve(
                    decoded_signal=decoded,
                    sample_rate=sample_rate,
                    freq_bracket=(
                        fft_frequencies[peak] - bin_width,
                        fft_frequencies[peak] + bin_width,
                    ),
                )
                progress.update()

    best = int(np.argmax(power))
    base_index, offset_index = divmod(best, count)
    echo = rx[base_index : base_index + templates.shape[0]]
    return PulseEstimate(
        r=float(range_gates[best] * constants.c / sample_rate),
        v=float(frequencies[best] * constants.c / carrier),
        frequency=float(frequencies[best]),
        power=power,
        frequencies=frequencies,
        range_gates=range_gates,
        template=np.asarray(templates[:, offset_index], dtype=np.complex128),
        echo=np.asarray(echo, dtype=np.complex128),
        template_index=offset_index,
        range_index=base_index,
    )


def normalized(signal: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    scale = np.max(np.abs(signal))
    return signal / scale if scale else signal


def plot_diagnostics(
    estimate: PulseEstimate,
    reference: OrbitReference,
    sample_rate: float,
    carrier: float,
    code: npt.NDArray[np.floating],
    samples_per_baud: int,
) -> list[plt.Figure]:
    """Create summary plots for the range search and best match."""
    ranges_km = estimate.range_gates * constants.c * 1e-3 / sample_rate
    best = int(np.argmax(estimate.power))

    fig_summary, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    axes[0].plot(ranges_km, 10 * np.log10(np.maximum(estimate.power, np.finfo(float).tiny)))
    axes[0].axvline(reference.r * 1e-3, color="tab:red", linestyle="--", label="Orbit reference")
    axes[0].scatter(
        ranges_km[best], 10 * np.log10(estimate.power[best]), color="tab:orange", zorder=3, label="Estimate"
    )
    axes[0].set(xlabel="Two-way range (km)", ylabel="Matched power (dB)", title="Sub-sample range search")
    axes[0].legend()
    rates = estimate.frequencies * constants.c / carrier
    axes[1].plot(ranges_km, rates, linewidth=1)
    axes[1].axhline(reference.v, color="tab:red", linestyle="--", label="Orbit reference")
    axes[1].scatter(ranges_km[best], rates[best], color="tab:orange", zorder=3, label="Estimate")
    axes[1].set(xlabel="Two-way range (km)", ylabel="Range rate (m/s)", title="Doppler estimate")
    axes[1].legend()

    fft_length = 2 ** (int(np.log2(estimate.echo.size)) + 2)
    spectrum = fftshift(fft(estimate.echo * np.conj(estimate.template), n=fft_length))
    spectrum_frequencies = fftshift(fftfreq(fft_length, d=1.0 / sample_rate))
    axes[2].plot(spectrum_frequencies * 1e-3, np.abs(spectrum), color="tab:purple")
    axes[2].axvline(estimate.frequency * 1e-3, color="tab:orange", linestyle="--")
    axes[2].set(xlabel="Doppler frequency (kHz)", ylabel="Magnitude", title="Decoded echo spectrum")

    phase = np.exp(-2j * np.pi * estimate.frequency * np.arange(estimate.echo.size) / sample_rate)
    compensated = normalized(estimate.echo * phase)
    template = normalized(estimate.template)
    mask = transition_mask(code, samples_per_baud, template.size)
    masked_samples = np.flatnonzero(~mask)

    fig_match, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6), constrained_layout=True)
    axes[0].plot(template.real, label="Template I", color="tab:blue")
    axes[0].plot(compensated.real, label="Echo I", color="tab:orange", alpha=0.8)
    axes[0].plot(
        masked_samples,
        template.real[~mask],
        ".",
        color="tab:red",
        label="Masked transitions",
    )
    axes[1].plot(template.imag, label="Template Q", color="tab:blue")
    axes[1].plot(compensated.imag, label="Echo Q", color="tab:orange", alpha=0.8)
    axes[1].plot(
        masked_samples,
        template.imag[~mask],
        ".",
        color="tab:red",
        label="Masked transitions",
    )

    axes[0].set_title("Best template and Doppler-compensated echo")
    axes[1].set_xlabel("Sample")
    for axis in axes:
        axis.set_ylabel("Normalized amplitude")
        axis.legend(loc="upper right")
    return [fig_summary, fig_match]


def plot_signal_chain(
    template: npt.NDArray[np.complex128],
    echo: npt.NDArray[np.complex128],
    doppler: float,
    sample_rate: float,
    title: str,
) -> plt.Figure:
    """Show the template multiplication and Doppler compensation step by step."""
    doppler_phasor = np.exp(-2j * np.pi * doppler * np.arange(echo.size) / sample_rate)
    compensated = echo * doppler_phasor
    decoded = echo * np.conj(template)
    compensated_decoded = compensated * np.conj(template)
    template_norm = normalized(template)

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(11, 9), constrained_layout=True)
    fig.suptitle(title)

    axes[0].plot(template_norm.real, "--", color="tab:blue", label="Template I")
    axes[0].plot(template_norm.imag, "--", color="tab:red", label="Template Q")
    axes[0].plot(normalized(echo).real, color="tab:blue", alpha=0.65, label="Echo I")
    axes[0].plot(normalized(echo).imag, color="tab:red", alpha=0.65, label="Echo Q")
    axes[0].set_title("Template and raw echo")

    axes[1].plot(template_norm.real, "--", color="tab:blue", label="Template I")
    axes[1].plot(template_norm.imag, "--", color="tab:red", label="Template Q")
    axes[1].plot(normalized(compensated).real, color="tab:blue", alpha=0.65, label="Echo I")
    axes[1].plot(normalized(compensated).imag, color="tab:red", alpha=0.65, label="Echo Q")
    axes[1].set_title("Template and Doppler-compensated echo")

    axes[2].plot(normalized(decoded).real, color="tab:blue", label="I")
    axes[2].plot(normalized(decoded).imag, color="tab:red", label="Q")
    axes[2].set_title("Echo × conjugate(template)")

    axes[3].plot(normalized(compensated_decoded).real, color="tab:blue", label="I")
    axes[3].plot(normalized(compensated_decoded).imag, color="tab:red", label="Q")
    axes[3].set_title("Doppler-compensated echo × conjugate(template)")
    axes[3].set_xlabel("Sample")
    for axis in axes:
        axis.set_ylabel("Normalized amplitude")
        axis.legend(loc="upper right", ncols=2)
    return fig


def plot_full_diagnostics(
    data: PulseData,
    estimate: PulseEstimate,
    reference: OrbitReference,
    code: npt.NDArray[np.float64],
    exp_def: object,
    min_range_gate: int,
    sub_resolution: int,
    sample_rate: float,
    carrier: float,
) -> list[plt.Figure]:
    """Template, phase, raw-signal, and true-match plots."""
    figures: list[plt.Figure] = []

    # Compare every analytic sub-sample template with its measured counterpart.
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 7), constrained_layout=True)
    colors = plt.colormaps["viridis"](np.linspace(0, 1, sub_resolution))
    for index, color in enumerate(colors):
        axes[0].plot(data.templates[:, index].real, color=color, alpha=0.8)
        axes[1].plot(data.measured_templates[:, index].real, color=color, alpha=0.8)
    axes[0].set_title("Analytic templates across sub-sample offsets")
    axes[1].set_title("Measured templates across sub-sample offsets")
    axes[1].set_xlabel("Sample")
    for axis in axes:
        axis.set_ylabel("In-phase amplitude")
    figures.append(fig)

    phase = np.unwrap(2 * np.angle(data.tx)) / 2
    window = max(1, 5 * int(exp_def.baud_length_usec * 1e-6 * sample_rate))
    padded = np.pad(phase, (window // 2, window - 1 - window // 2), mode="edge")
    smooth_phase = np.convolve(padded, np.ones(window) / window, mode="valid")
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(phase, alpha=0.55, label="Measured phase")
    ax.plot(smooth_phase, "--", linewidth=2, label="Smoothed phase")
    ax.set(title="Transmit-pulse phase", xlabel="Sample", ylabel="Unwrapped phase (rad)")
    ax.legend()
    figures.append(fig)

    figures.append(
        plot_signal_chain(
            estimate.template,
            estimate.echo,
            estimate.frequency,
            sample_rate,
            "Signal chain at the estimated range and Doppler",
        )
    )

    true_gate = reference.r / constants.c * sample_rate
    true_range_index = int(true_gate) - min_range_gate - 1
    true_template_index = int((true_gate - min_range_gate - 1 - true_range_index) * sub_resolution)
    true_offset = true_template_index / sub_resolution
    true_template = tx_signal_model(
        code=code,
        baud_length_usec=exp_def.baud_length_usec,
        t_samp_usec=exp_def.t_samp_usec,
        ipp_samps=exp_def.ipp_samps,
        read_length=data.tx.size,
        sub_resolution=np.array([true_offset]),
        bandwidth=None,
        filt=ReceiverChainModel.b414d15_gaus,
    )[:, 0]
    true_echo = data.rx[true_range_index : true_range_index + data.tx.size]
    true_doppler = reference.v * carrier / constants.c
    figures.append(
        plot_signal_chain(
            true_template,
            true_echo,
            true_doppler,
            sample_rate,
            "Signal chain at the orbit-derived range and Doppler",
        )
    )

    echo_sample = sample_rate * reference.r / constants.c + data.tx_start
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    cropped = data.signal[data.range_start : data.range_start + data.rx.size]
    axes[0].plot(cropped.real, label="I")
    axes[0].plot(cropped.imag, label="Q", alpha=0.75)
    axes[0].axvline(echo_sample - data.range_start, color="tab:red", linestyle="--", label="Orbit echo")
    axes[0].set_title("Selected receive range")
    axes[1].plot(data.signal.real, label="I")
    axes[1].plot(data.signal.imag, label="Q", alpha=0.75)
    axes[1].axvline(echo_sample, color="tab:red", linestyle="--", label="Orbit echo")
    axes[1].axvspan(data.tx_start, data.tx_end, color="tab:green", alpha=0.15, label="Transmit pulse")
    axes[1].set_title("Full IPP")
    axes[1].set_xlabel("Sample")
    for axis in axes:
        axis.set_ylabel("Amplitude")
        axis.legend(loc="upper right")
    figures.append(fig)
    return figures


def transition_mask(
    code: npt.NDArray[np.floating],
    samples_per_baud: int,
    length: int,
    margin: int = 1,
) -> npt.NDArray[np.bool_]:
    """Mask code transitions at the selected template's fractional offset."""
    mask = np.ones(length, dtype=bool)
    mask[:margin] = False
    mask[-margin:] = False
    flips = np.flatnonzero(np.diff(code) != 0) + 1
    for flip in flips:
        centre = flip * samples_per_baud
        mask[max(0, centre - margin) : min(length, centre + margin + 1)] = False
    return mask


def main() -> None:
    args = parse_args()
    if args.duration <= 0 or args.sub_resolution < 1 or args.min_range_gate >= args.max_range_gate:
        raise ValueError(
            "Duration and sub-resolution must be positive, and the range-gate interval non-empty"
        )

    start_dt = parse_datetime(args.start_time)
    end_dt = start_dt + dt.timedelta(seconds=args.duration)
    orbit = load_orbit(args.orbit_file, start_dt - dt.timedelta(hours=1), end_dt + dt.timedelta(hours=1))
    radar = radardef.EiscatUHF(location=EiscatUHFLocation.TROMSO, beam_type=BeamType.CASSEGRAIN)
    reader = radar.load_data(args.data_file)
    if reader is None:
        raise RuntimeError(f"Could not load radar data from {args.data_file}")

    request = time_interval_to_sample_bound(
        time_bounds=(reader.epoch_bounds.ts_start_usec * 1e-6, reader.epoch_bounds.ts_end_usec * 1e-6),
        start_time=ts_from_str(args.start_time),
        end_time=ts_from_str(end_dt.strftime(DATETIME_FORMAT)[:-3]),
        sample_rate=reader.exp_def.sample_rate,
        relative_time=False,
    )
    bounds = sample_interval_to_closest_ipp(request, reader.exp_def.ipp_samps)
    raw = reader.read(
        channel=reader.exp_def.rx_channels,
        start_sample=bounds.start + args.offset,
        vector_length=bounds.end - bounds.start,
    )
    if raw.ndim > 1:
        raw = np.sum(raw, axis=0)
    pulses = raw.reshape((-1, reader.exp_def.ipp_samps))
    if not 0 <= args.ipp_index < pulses.shape[0]:
        raise IndexError(
            f"IPP index {args.ipp_index} is outside the available range [0, {pulses.shape[0] - 1}]"
        )
    signal = np.asarray(pulses[args.ipp_index], dtype=np.complex128)

    tx_start = int(reader.exp_def.t_tx_start_usec / reader.exp_def.t_samp_usec)
    tx_end = int(reader.exp_def.t_tx_end_usec / reader.exp_def.t_samp_usec)
    tx = signal[tx_start:tx_end]
    range_start, range_end = extract_requested_range_gates(
        args.min_range_gate, args.max_range_gate, "sample", reader.exp_def
    )
    rx = signal[range_start:range_end]
    first_ipp = request.start // reader.exp_def.ipp_samps
    code_index = select_code(tx, reader.exp_def, first_ipp + args.ipp_index, args.code_source)
    templates = make_templates(tx, reader.exp_def, code_index, args.sub_resolution, args.template)
    measured_templates = tx_modulation_model(
        tx,
        np.ones(tx.size, dtype=bool),
        sub_resolution=args.sub_resolution,
    )

    analysed_epochs = np.arange(
        start=start_dt.timestamp(),
        stop=end_dt.timestamp(),
        step=reader.exp_def.t_ipp_usec * 1e-6,
    )
    reference_ranges, reference_rates = orbit_references(radar, orbit, analysed_epochs)
    reference = OrbitReference(
        r=reference_ranges[args.ipp_index],
        v=reference_rates[args.ipp_index],
    )
    sample_rate = float(reader.exp_def.sample_rate)
    carrier = args.radar_frequency_mhz * 1e6
    estimate = estimate_pulse(rx, templates, args.min_range_gate, args.max_range_gate, sample_rate, carrier)
    print(f"Range:      {estimate.r / 1e3:.3f} km ({estimate.r - reference.r:+.1f} m)")
    print(f"Range rate: {estimate.v:.3f} m/s ({estimate.v - reference.v:+.3f} m/s)")

    samples_per_baud = int(round(reader.exp_def.baud_length_usec / reader.exp_def.t_samp_usec))
    template_offsets = np.linspace(0.0, 1.0, num=args.sub_resolution)
    figures = plot_diagnostics(
        estimate,
        reference,
        sample_rate,
        carrier,
        reader.exp_def.code[code_index],
        samples_per_baud,
    )
    if args.diagnostics == "full":
        figures.extend(
            plot_full_diagnostics(
                PulseData(
                    signal=signal,
                    rx=rx,
                    tx=tx,
                    templates=make_templates(tx, reader.exp_def, code_index, args.sub_resolution, "model"),
                    measured_templates=measured_templates,
                    range_start=range_start,
                    tx_start=tx_start,
                    tx_end=tx_end,
                ),
                estimate,
                reference,
                reader.exp_def.code[code_index],
                reader.exp_def,
                args.min_range_gate,
                args.sub_resolution,
                sample_rate,
                carrier,
            )
        )
    if args.rti:
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        plotting.rti(
            ax,
            reader,
            start_time=args.start_time,
            end_time=end_dt.strftime(DATETIME_FORMAT)[:-3],
            start_range_gate=args.min_range_gate,
            end_range_gate=args.max_range_gate,
            range_gate_unit="sample",
            axis_units=True,
            log=True,
        )
        ax.set_title("Range-time intensity")
        figures.append(fig)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for index, figure in enumerate(figures, start=1):
            name = f"pulse_analysis_{index:02d}"
            figure.savefig(args.output_dir / f"{name}.png", dpi=180, bbox_inches="tight")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
