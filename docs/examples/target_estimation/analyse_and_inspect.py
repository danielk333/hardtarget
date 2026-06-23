# # Analyse and Inspect
# ---
# How to trigger analysis and then inspect the output

import os
import sys
import tempfile
from pathlib import Path

from matplotlib import gridspec
from matplotlib import pyplot as plt

from hardtarget import load_analysed_data, target_estimation
from hardtarget.plotting import target_estimation_plots

# Workaround to make jupyter notebook find utils
sys.path.insert(1, str(Path(os.path.abspath("")) / "docs" / "examples" / "extras"))
import utils

try:
    config = Path(__file__).parent.parent.absolute() / "cfg" / "test.ini"
except NameError:
    config = Path(os.path.abspath("")) / "docs" / "examples" / "cfg" / "test.ini"

# ##Prerequisites
# ---

# Data to analyse

tmp_dir = tempfile.TemporaryDirectory()
raw_path = Path(tmp_dir.name) / "raw"
raw_path.mkdir(parents=True, exist_ok=True)
converted_path = Path(tmp_dir.name) / "converted"
raw_data = utils.download_test_data(raw_path)

# Convert the data to a usable format.

data = utils.convert_test_data(raw_data, converted_path)[0]

# ## Analyse data
# ---
# Minimal configuration for analysis, progress, start_time, end_time and relative time are optionals.
# In this case we choose to visualize the 500 first ipps of the measurement, acceleration has been ignored
# for this example.

output_path = Path(tmp_dir.name) / "analysed"
target_estimation(
    data=data,
    config=config,
    output=output_path,
    start_time=0,
    end_time=(500 * 3120) * 1e-6,  # 3120 = t_ipp_usec
    relative_time=True,
)

# ## Plot results
# ---
# First load results

data_generator = load_analysed_data(output_path)

# Then we can plot visualize it with different plots

out_data, exp_def, cfg_params, pro_params = list(data_generator)[0]

# Plot peaks, this visualizes the peaks  of the hardtargets range, velocity
# and acceleration over time (red is marking the detections). Furthermore the the signal to noise ratio is
# shown.

fig, axes = plt.subplots(2, 2)
target_estimation_plots.plot_peaks(
    axes,
    out_data,
    exp_def,
    cfg_params,
    pro_params,
    snr_dB_limit=15.0,
)
fig.set_size_inches(10, 10)

# Plot detections, this visualizes only the hardtarget detections from the measurement
# (red dots from previous plot).
# Additionaly it also adds the acceleration and range gate relative to the range for each detection.

fig, axes = plt.subplots(2, 3)
target_estimation_plots.plot_detections(
    axes,
    out_data,
    exp_def,
    cfg_params,
    pro_params,
)
fig.set_size_inches(12, 10)

# Plot map, this visualizes the hardtarget at the specific range gates,
# aswell the estimated noise relative to range.

fig = plt.figure()
gs = gridspec.GridSpec(2, 2, figure=fig)
axes = [
    fig.add_subplot(gs[0, :]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
]
target_estimation_plots.plot_map(
    axes,
    out_data,
    exp_def,
    cfg_params,
    pro_params,
)
fig.set_size_inches(10, 10)


plt.show()
tmp_dir.cleanup()
