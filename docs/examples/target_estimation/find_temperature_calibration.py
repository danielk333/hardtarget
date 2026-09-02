"""Estimate the ADC-to-noise-temperature calibration from injected noise.

The EISCAT calibration source alternates between IPPs. Its known temperature
can be used to determine the conversion between signal power and units of energy (or temperature).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import radardef
from matplotlib import pyplot as plt
from radardef.types import BeamType, EiscatUHFLocation

DEFAULT_CALIBRATION_TEMPERATURE_K = 177.8


@dataclass(frozen=True)
class CalibrationResult:
    """Noise statistics and the resulting conversion factors."""

    per_ipp_variance: npt.NDArray[np.float64]
    kelvin_per_complex_power: npt.NDArray[np.float64]
    receiver_temperature: npt.NDArray[np.float64]
    cal_variance: npt.NDArray[np.float64]
    sky_variance: npt.NDArray[np.float64]
    on_parity: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_file", type=Path, help="Radar data file containing calibration IPPs")
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_CALIBRATION_TEMPERATURE_K,
        help="Injected calibration noise temperature in kelvin",
    )
    parser.add_argument("--num-ipps", type=int, default=200, help="Number of IPPs to use")
    parser.add_argument("--start-ipp", type=int, default=0, help="First IPP relative to the data bounds")
    parser.add_argument(
        "--cal-on-parity",
        choices=("auto", "even", "odd"),
        default="auto",
        help="Calibration-on IPPs; auto selects the parity with greater variance",
    )
    parser.add_argument(
        "--edge-samples",
        type=int,
        default=0,
        help="Discard this many samples from each edge of the calibration gate",
    )
    parser.add_argument("--channel", type=int, default=0, help="Index into the experiment RX channels")
    parser.add_argument("--output", type=Path, help="Save the diagnostic plot to this path")
    parser.add_argument("--no-show", action="store_true", help="Do not open the diagnostic plot")
    return parser.parse_args()


def estimate_calibration(
    calibration_samples: npt.NDArray[np.complex128],
    temperature: float,
    parity: str,
):
    """Compare alternating IPPs and estimate temperature conversion factors."""
    per_ipp_variance = np.var(
        np.concatenate(
            [
                calibration_samples.real,
                calibration_samples.imag,
            ],
            axis=1,
        ),
        axis=1,
    )
    variances = [per_ipp_variance[::2], per_ipp_variance[1::2]]
    variances_mean = [np.mean(x) for x in variances]

    if parity == "auto":
        on_parity = int(np.argmax(variances_mean))
    else:
        on_parity = 0 if parity == "even" else 1
    off_parity = 1 - on_parity
    cal_variance = variances[on_parity]
    sky_variance = variances[off_parity]

    # For z = complex noise with component sigma -> E[|z|^2] = 2 sigma^2 = Npwr.
    # Assume coefficient converts c * E[|z|^2] -> Temperature
    # c = T / Npwr
    kelvin_per_complex_power = temperature / (2 * cal_variance)
    receiver_temperature = 2 * sky_variance * kelvin_per_complex_power

    result = CalibrationResult(
        per_ipp_variance=per_ipp_variance,
        kelvin_per_complex_power=kelvin_per_complex_power,
        receiver_temperature=receiver_temperature,
        cal_variance=cal_variance,
        sky_variance=sky_variance,
        on_parity=on_parity,
    )
    return result


def plot_diagnostics(
    samples: npt.NDArray[np.complex128],
    result: CalibrationResult,
) -> plt.Figure:
    """Plot the calibration gate and separation of on/off IPPs."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    image = axes[0, 0].imshow(np.abs(samples), aspect="auto", interpolation="nearest", origin="lower")
    axes[0, 0].set(title="Calibration-gate magnitude", xlabel="Sample in gate", ylabel="IPP")
    fig.colorbar(image, ax=axes[0, 0], label="ADC magnitude")

    indices = np.arange(result.per_ipp_variance.size)
    for parity, label, color in ((0, "Even IPPs", "tab:blue"), (1, "Odd IPPs", "tab:orange")):
        selected = indices % 2 == parity
        suffix = " (cal on)" if parity == result.on_parity else " (cal off)"
        axes[0, 1].plot(indices[selected], result.per_ipp_variance[selected], ".", color=color, label=label + suffix)
        axes[1, 0].hist(
            np.sqrt(result.per_ipp_variance[selected]),
            bins="auto",
            alpha=0.6,
            color=color,
            label=label + suffix,
        )
    axes[0, 1].set(title="Component variance per IPP", xlabel="IPP", ylabel=r"Variance $\sigma^2$ (ADC$^2$)")
    axes[0, 1].legend()
    axes[1, 0].set(title="Noise-sigma distributions", xlabel=r"Component $\sigma$ (ADC)", ylabel="Count")
    axes[1, 0].legend()

    max_points = min(samples.size, 30_000)
    flattened = samples.ravel()[:: max(1, samples.size // max_points)]
    axes[1, 1].hexbin(flattened.real, flattened.imag, gridsize=55, bins="log", mincnt=1, cmap="viridis")
    axes[1, 1].set(
        title="Complex calibration samples",
        xlabel="In-phase component (ADC)",
        ylabel="Quadrature component (ADC)",
        aspect="equal",
    )
    return fig


def main() -> None:
    args = parse_args()
    if args.temperature <= 0 or args.num_ipps < 4 or args.start_ipp < 0 or args.edge_samples < 0:
        raise ValueError(
            "Temperature must be positive, at least four IPPs are required, and indices must be non-negative"
        )

    radar = radardef.EiscatUHF(location=EiscatUHFLocation.TROMSO, beam_type=BeamType.CASSEGRAIN)
    reader = radar.load_data(args.data_file)
    if reader is None:
        raise RuntimeError(f"Could not load radar data from {args.data_file}")
    if not 0 <= args.channel < len(reader.exp_def.rx_channels):
        raise IndexError(f"RX channel index {args.channel} is unavailable")
    if reader.exp_def.t_cal_on_usec is None or reader.exp_def.t_cal_off_usec is None:
        raise ValueError("The experiment definition does not contain a calibration interval")

    channel = reader.exp_def.rx_channels[args.channel]
    channel_start, channel_end = reader.bounds(channel)
    ipp_samples = reader.exp_def.ipp_samps
    start_sample = channel_start + args.start_ipp * ipp_samples
    available_ipps = (channel_end - start_sample) // ipp_samples
    num_ipps = min(args.num_ipps, available_ipps)
    if num_ipps < 4:
        raise ValueError(f"Only {num_ipps} complete IPPs are available from --start-ipp")
    if num_ipps % 2:
        num_ipps -= 1

    raw = reader.read(channel=channel, start_sample=start_sample, vector_length=num_ipps * ipp_samples)
    if raw.ndim != 1:
        raw = np.asarray(raw).squeeze()
    pulses = raw.reshape((num_ipps, ipp_samples))
    cal_start = int(reader.exp_def.t_cal_on_usec / reader.exp_def.t_samp_usec) + args.edge_samples
    cal_end = int(reader.exp_def.t_cal_off_usec / reader.exp_def.t_samp_usec) - args.edge_samples
    if cal_end - cal_start < 2:
        raise ValueError("Calibration gate has fewer than two samples after removing its edges")
    calibration_samples = pulses[:, cal_start:cal_end]

    result = estimate_calibration(
        calibration_samples,
        args.temperature,
        args.cal_on_parity,
    )
    parity_name = "even" if result.on_parity == 0 else "odd"
    print(f"Calibration-on parity:                {parity_name}")
    print(f"Mean Calibration-off variance:        {result.sky_variance.mean():.6g} ADC units")
    print(f"Mean Calibration-on variance:         {result.cal_variance.mean():.6g} ADC units")
    print(f"Mean temperature / complex power:     {result.kelvin_per_complex_power.mean():.6g} K/ADC^2")
    print(f"Mean equivalent receiver temperature: {result.receiver_temperature.mean():.3f} K")

    figure = plot_diagnostics(calibration_samples, result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=180, bbox_inches="tight")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
