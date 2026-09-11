"""Receiver-chain impulse responses.

Currently implements reading from from EISCAT `.fir` files based on `get_impresp.m` written by Jussi
Markkanen [(c) EISCAT Scientific Association 1998-]

"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import scipy.signal as sc_signal
from matplotlib.figure import Figure
from scipy.interpolate import PchipInterpolator

from hardtarget.constants import ReceiverChainModel


def boxcar(n: int, normalize: bool = True) -> npt.NDArray:
    h = np.ones(n, dtype=float)
    return h / h.sum() if normalize else h


def b414d15_gaus(
    x: npt.NDArray[np.complexfloating],
    h_stages: int,
    h_drate: int,
    f_taps: int,
    f_esym: int,
    f_drate: int,
    taps: list[int],
) -> npt.NDArray[np.complexfloating]:
    """
    https://www.renesas.com/en/document/dst/hsp43220-datasheet?r=528226

    """

    y = x.copy()

    # --- High Order Decimation Filter ---
    hodf_decimation = h_drate + 1
    h_hdf = boxcar(hodf_decimation)
    for _ in range(h_stages):
        y = sc_signal.lfilter(h_hdf, [1.0], y)  # type: ignore[attr-defined]

    # Downsample
    y = y[::hodf_decimation]

    # --- FIR Decimation filter ---

    norm_taps = np.array(taps[:f_taps]) / 0x7FFFF

    if f_esym:
        h_fir = np.concatenate((norm_taps, norm_taps[::-1]))
    else:
        h_fir = np.concatenate((norm_taps, -norm_taps[::-1]))

    h_fir = h_fir / h_fir.sum()

    y = sc_signal.lfilter(h_fir, [1.0], y)  # type: ignore[attr-defined]
    # Final decimation downsample step
    y = y[:: f_drate + 1]

    return np.array(y)


def cic_decimate(
    x: npt.NDArray[np.complexfloating], decimation: int, combs: int, delay: int = 1
) -> npt.NDArray[np.complexfloating]:
    """
    CIC decimator: N integrators at input rate that is decimated,
    then N comb stages at output rate with a delay.
    """
    y = np.asarray(x, dtype=np.complex128)

    # integrators
    for _ in range(combs):
        y = np.cumsum(y)

    # decimate
    y = y[::decimation]

    # combs
    for _ in range(combs):
        y = y - np.concatenate([np.zeros(delay, dtype=y.dtype), y[:-delay]])

    # normalize CIC DC gain
    y /= (decimation * delay) ** combs
    return y


def mu_radar_filter_post_2004(
    x: npt.NDArray[np.complexfloating],
) -> npt.NDArray[np.complexfloating]:
    """
    Model MUR chain accoring to [^1]
    IF samples at complex baseband -> CIC decimation -> 16-tap FIR compensation.

    [^1]: Hassenpflug, G., Yamamoto, M., Luce, H., Fukao, S., 2008.
        Description and demonstration of the new Middle and Upper atmosphere Radar imaging system: 1-D, 2-D, and 3-D imaging of troposphere and stratosphere.
        Radio Sci. 43, RS2013. https://doi.org/10.1029/2006RS003603

    """
    # CIC matched-filter / decimator
    # TODO: guessing the cic decimation rate of 8, it kinda makes sense beacuse with 16 taps
    # 8*15=120 which is the total decimation rate... but double check needed!
    y_cic = cic_decimate(x, decimation=8, combs=6)

    # 16-tap FIR amplitude/frequency compensator
    # TODO: gussing the compensating FIR, no coefficients were available in the paper?
    fir_taps = sc_signal.firwin(  # type: ignore[attr-defined]
        numtaps=16,
        cutoff=0.8,
        window="hamming",
    )
    y_out = sc_signal.lfilter(fir_taps, [1.0], y_cic)  # type: ignore[attr-defined]
    y_out = y_out[::15]

    return np.array(y_out)


class PulseFilter(Protocol):
    def __call__(self, x: npt.NDArray[np.complexfloating]) -> npt.NDArray[np.complexfloating]: ...


@dataclass
class DigitalReceiverChain:
    """Describes the digital receiver-chain of a radar system,
    i.e. all the steps that occur after the initial ADC such as filtering and decimation
    """

    model: PulseFilter = lambda x: x
    delay: float = 0
    decimation: int = 1


B414d15Filter = DigitalReceiverChain(
    model=lambda x: b414d15_gaus(
        x, h_stages=5, h_drate=4, f_taps=2, f_esym=1, f_drate=2, taps=[0x29F17, 0x53E2D]
    ),  # TODO: Hardcoded for now, should be read from FIR file.
    delay=1,
    decimation=15,  # (h_drate+1)*(f_drate+1)
)

MuPost2004Filter = DigitalReceiverChain(model=mu_radar_filter_post_2004, delay=0, decimation=120)


def get_reciver_chain(model: ReceiverChainModel | str) -> DigitalReceiverChain:
    if model == ReceiverChainModel.b414d15_gaus:
        filt = B414d15Filter
    elif model == ReceiverChainModel.mu2004:
        filt = MuPost2004Filter
    elif model == ReceiverChainModel.none:
        filt = DigitalReceiverChain()
    else:
        raise ValueError(
            f"No reciver chain available with name: {model}, available models are: {'/'.join(ReceiverChainModel)} "
        )
    return filt


def _parse_integer(value: str, line_number: int) -> int:
    """Conveniant debugging when parsing files"""
    try:
        return int(value, 0)
    except ValueError as exc:
        raise ValueError(f"line {line_number}: invalid integer {value!r}") from exc


def read_fir(firpar_file: str | Path) -> tuple[npt.NDArray[np.float64], int, int, int]:
    """Read an EISCAT FIRPAR 0.1 filter definition.

    Returns
    -------
    fir
        The complete FIR tap sequence, with symmetry expanded and signed
        20-bit values decoded.
    f_dec
        FIR decimation factor.
    h_dec
        HDF/CIC decimation factor.
    h_order
        Number of HDF/CIC stages.
    """
    path = Path(firpar_file)
    entries: list[tuple[int, list[str]]] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if line and not line.startswith("%"):
            # Comments may be after a value (as they do in the EISCAT files).
            entries.append((line_number, line.split("%", maxsplit=1)[0].split()))

    if not entries:
        raise ValueError(f"{path} appears empty?")
    if not any(words == ["FIRPAR_VS", "0.1"] for _, words in entries):
        raise ValueError("Illegal format: only FIRPAR_VS 0.1")

    scalar_keys = {"H_STAGES", "H_DRATE", "F_DRATE", "F_TAPS", "F_ESYM"}
    values: dict[str, int] = {}
    tap_values: dict[int, int] = {}
    for line_number, words in entries:
        key = words[0]
        if key == "TAP":
            if len(words) < 3:
                raise ValueError(f"line {line_number}: TAP requires an index and value")
            index = _parse_integer(words[1], line_number)
            if index < 0:
                raise ValueError(f"line {line_number}: TAP index cannot be negative")
            tap_values[index] = _parse_integer(words[2], line_number)
        elif key in scalar_keys:
            if len(words) < 2:
                raise ValueError(f"line {line_number}: {key} requires a value")
            values[key] = _parse_integer(words[1], line_number)

    missing = scalar_keys.difference(values)
    if missing:
        raise ValueError(f"Required parameter(s) not defined: {', '.join(sorted(missing))}")
    if not tap_values:
        raise ValueError("TAPs not defined")

    # MATLAB grows a numeric array and fills skipped indices with zero.
    # The rest is mostly a direct copy from the .m file
    taps = np.zeros(max(tap_values) + 1, dtype=np.int64)
    for index, value in tap_values.items():
        taps[index] = value - 2**20 if value >= 2**19 else value

    f_taps = values["F_TAPS"] + 1
    if values["F_ESYM"] == 1:
        reflected = taps[::-1] if f_taps % 2 == 0 else taps[-2::-1]
    elif values["F_ESYM"] == 0:
        reflected = taps[-2::-1]
    else:
        raise ValueError("'F_ESYM' must be 0 or 1")
    fir = np.concatenate((taps, reflected)).astype(np.float64)

    if fir.size != f_taps:
        raise ValueError(f"Number of FIR taps ({f_taps}) does not match expanded TAPs ({fir.size})")
    return fir, values["F_DRATE"] + 1, values["H_DRATE"] + 1, values["H_STAGES"]


def _cic_impulse_response(order: int, decimation: int) -> npt.NDArray[np.float64]:
    response = np.ones(decimation, dtype=np.float64)
    for _ in range(order - 1):
        response = np.convolve(response, np.ones(decimation, dtype=np.float64))
    # This reproduces hcic's scaling. Later normalizations make the scale cancel,
    # but retaining it makes the intermediate MATLAB-compatible too.
    response /= 2.0 ** np.ceil(np.log2(response.sum()))
    return response


def _insert_zeros(values: npt.NDArray[np.float64], count: int) -> npt.NDArray[np.float64]:
    """replica of the matlab function used"""
    if count < 1:
        return values.copy()
    result = np.zeros((values.size - 1) * (count + 1) + 1, dtype=np.float64)
    result[:: count + 1] = values
    return result


def get_impresp(
    firpar_file: str | Path, p_dtau: float, do_plot: bool = False
) -> tuple[npt.NDArray[np.float64] | float, float, npt.NDArray[np.float64], int]:
    """Build the equivalent receiver-chain impulse response.

    Parameters are compatible with the original MATLAB `get_impresp` from GUISDAP(?).
    Times, including `p_dtau` and the returned `t0`, are in microseconds.
    `taps` has unit sum.
    """
    if not np.isscalar(p_dtau) or isinstance(p_dtau, (str, bytes)):
        raise TypeError("p_dtau must be a numeric scalar")
    p_dtau = float(p_dtau)

    path = Path(firpar_file)
    initial = path.stem[:1].lower()
    if initial == "w":
        adc_rate = 10.0
    else:
        adc_rate = 15.0
        if initial != "b":
            # Dont know if this can happen? better warn if it does
            warnings.warn(
                f"Cannot infer ADC rate from {path.name!r}; using 15 MHz",
                RuntimeWarning,
                stacklevel=2,
            )

    fir, f_dec, h_dec, h_order = read_fir(path)
    total_decimation = h_dec * f_dec
    fir /= fir.max()
    hdf = _cic_impulse_response(h_order, h_dec)
    hdf /= np.max(np.abs(hdf))
    zero_stuffed_fir = _insert_zeros(fir, h_dec - 1)
    zero_stuffed_fir /= np.max(np.abs(zero_stuffed_fir))
    taps = np.convolve(hdf, zero_stuffed_fir)
    taps /= taps.sum()

    # TODO: this was in the original code but i would rather split this across two functions later
    # and have this return path the default for this function
    if p_dtau <= 0:
        return np.nan, np.nan, taps, total_decimation

    ddf = taps / np.sum(taps / adc_rate)
    t_ddf = np.arange(ddf.size, dtype=np.float64) / adc_rate
    center = t_ddf[-1] / 2.0
    half_step = p_dtau / 2.0
    n_side = np.floor((center - half_step) / p_dtau)
    t0 = center - half_step - n_side * p_dtau
    t_end = center + half_step + n_side * p_dtau
    # arange with a half-step tolerance mirrors MATLAB's inclusive colon.
    t_ip = np.arange(t0, t_end + p_dtau / 2.0, p_dtau)
    impresp = PchipInterpolator(t_ddf, ddf, extrapolate=False)(t_ip)

    if do_plot:
        # TODO: this should be moved away from here
        plot_response(path, adc_rate, h_dec, f_dec, hdf, fir, ddf, t_ip, impresp)
    return impresp, t0, taps, total_decimation


def plot_response(
    path: Path,
    adc_rate: float,
    h_dec: int,
    f_dec: int,
    hdf: npt.NDArray[np.float64],
    fir: npt.NDArray[np.float64],
    ddf: npt.NDArray[np.float64],
    t_ip: npt.NDArray[np.float64],
    impresp: npt.NDArray[np.float64],
) -> tuple[Figure, npt.NDArray]:
    """Plot the HDF, FIR, and combined DDF like the MATLAB implementation.

    TODO: Move this to the plotting subpackage
    """
    import matplotlib.pyplot as plt

    t_hdf = np.arange(hdf.size) / adc_rate
    t_fir = h_dec * np.arange(fir.size) / adc_rate
    t_ddf = np.arange(ddf.size) / adc_rate
    fig, axes = plt.subplots(2, 2, layout="tight")
    axes[0, 0].plot(t_hdf, hdf, "-" if hdf.size >= 20 else "o-")
    axes[0, 0].set_title(f"HDF / {path.stem} + Decimation {h_dec}")
    axes[0, 1].plot(t_fir, fir, "-" if fir.size >= 20 else "o-")
    axes[0, 1].set_title(f"FIR / {path.stem} + Decimation {f_dec}")
    axes[1, 0].remove()
    ddf_axis = axes[1, 1]
    ddf_axis.plot(t_ddf, ddf, "b-", t_ip, impresp, "r-")
    ddf_axis.set(title="DDF", xlabel="time [us]", xlim=(0, t_ddf[-1]))
    return fig, axes
