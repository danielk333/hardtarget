# # Visualisation of analysis paramters
import os
import sys

# Workaround to make jupyter notebook find utils
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import hardtarget
from hardtarget.types import AnalysisMethod

# Workaround to make jupyter notebook find utils
sys.path.insert(1, str(Path(os.path.abspath("")) / "docs" / "examples" / "extras"))
import utils

try:
    config = Path(__file__).parent.parent.absolute() / "cfg" / "test.ini"
except NameError:
    config = Path(os.path.abspath("")) / "docs" / "examples" / "cfg" / "test.ini"

# Data to analyse

tmp_dir = tempfile.TemporaryDirectory()
raw_path = Path(tmp_dir.name) / "raw"
raw_path.mkdir(parents=True, exist_ok=True)
converted_path = Path(tmp_dir.name) / "converted"
raw_data = utils.download_test_data(raw_path)

# Convert the data to a usable format.

data = utils.convert_test_data(raw_data, converted_path)[0]


# Extract data from the measurement, the Measurement object will load experiment and config parameters from
# the user config and the measurement data, then from that compute the process parameters. Furthermore it
# extracts the meta and raw data in a simple way.

measurement = hardtarget.data_handling.Measurement(
    Path(data),
    Path(config),
    AnalysisMethod.target_estimation,
)

process = hardtarget.GMFProcess(
    config,
    measurement.exp_params,
    measurement.cfg_params,
    measurement.pro_params,
    None,
    None,
    None,
)

exp = process.exp_params
cfg = process.cfg_params
pro = process.pro_params

# Extract the range gates

t = np.arange(pro.read_length, dtype=np.float64) / exp.sample_rate
t_ipp = np.arange(exp.ipp_samps, dtype=np.float64) / exp.sample_rate
rgs_t = np.zeros_like(t)
rel_rgs = pro.il0_rgs
rgs_t[rel_rgs] = 0.5

# Extract RX samples

all_rx = np.ones_like(t_ipp) * 0.75
all_rx[np.logical_or(t_ipp < exp.t_rx_start_usec * 1e-6, t_ipp > exp.t_rx_end_usec * 1e-6)] = 0
all_rx[0] = 0
vline_st = dict(ls="--", alpha=0.5)

# Mark the tx and rx start and end for each ipp

fig, ax = plt.subplots()
for ind in range(4):
    ax.axvline((exp.t_tx_start_usec + exp.t_ipp_usec * ind) * 1e-6, c="r", **vline_st)
    ax.axvline((exp.t_tx_end_usec + exp.t_ipp_usec * ind) * 1e-6, c="r", **vline_st)

    ax.axvline((exp.t_rx_start_usec + exp.t_ipp_usec * ind) * 1e-6, c="b", **vline_st)
    ax.axvline((exp.t_rx_end_usec + exp.t_ipp_usec * ind) * 1e-6, c="b", **vline_st)

    if exp.t_cal_on_usec is not None and exp.t_cal_off_usec is not None:
        ax.axvline((exp.t_cal_on_usec + exp.t_ipp_usec * ind) * 1e-6, c="g", **vline_st)
        ax.axvline((exp.t_cal_off_usec + exp.t_ipp_usec * ind) * 1e-6, c="g", **vline_st)

    dt = exp.t_ipp_usec * ind * 1e-6
    ax.plot(t_ipp + dt, all_rx, "-c")

ax.set_title("Stencils")
ax.set_xlabel("Time [s]")

# Plot rx and tx stencil, this is which samples from the whole ipp that represents the rx and tx signal

fig1, ax1 = plt.subplots()
ax1.plot(t, pro.rx_stencil, "-b", label="RX stencil")
ax1.plot(t, pro.tx_stencil, "-r", label="TX stencil")

# Plot all rx samples

ax1.plot(t_ipp, all_rx, "-c", label="RX")

# Plot range gates

ax1.plot(t, rgs_t, ".g", label="Range-gates")
ax1.plot(t, rgs_t, "-g", alpha=0.5)

# Mark the tx and rx start and end.

ax1.axvline(exp.t_tx_start_usec * 1e-6, c="r", **vline_st)
ax1.axvline(exp.t_tx_end_usec * 1e-6, c="r", **vline_st)
ax1.axvline(exp.t_rx_start_usec * 1e-6, c="b", **vline_st)
ax1.axvline(exp.t_rx_end_usec * 1e-6, c="b", **vline_st)

# mark calibration if available

if exp.t_cal_on_usec is not None and exp.t_cal_off_usec is not None:
    ax1.axvline(exp.t_cal_on_usec * 1e-6, c="m", **vline_st)
    ax1.axvline(exp.t_cal_off_usec * 1e-6, c="m", **vline_st)
ax1.set_xlim(0, exp.t_ipp_usec * 1e-6)
ax1.set_title("Stencils - single")
ax1.legend()
ax1.set_xlabel("Time [s]")
plt.show()
tmp_dir.cleanup()
