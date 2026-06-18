# # Analyse and Inspect
# ---
# How to trigger analysis and then inspect the output

import datetime as dt
import os
import tempfile
from pathlib import Path

from matplotlib import gridspec
from matplotlib import pyplot as plt

from hardtarget import load_analysed_data, target_estimation
from hardtarget.plotting import target_estimation_plots

try:
    config = Path(__file__).parent.parent.absolute() / "cfg" / "cfg_precision_orbit.ini"
except NameError:
    config = Path(os.path.abspath("")) / "docs" / "examples" / "cfg" / "cfg_precision_orbit.ini"

# ##Prerequisites
# ---

# Data to analyse
tmp_dir = tempfile.TemporaryDirectory()

# ## Analyse data
# ---

data = "../../../Documents/Data/Eiscat/leo/EISCAT_leo_mpark_2.1u_EI@uhf_20240704_100019_278878.hdf5"
output_path = Path(tmp_dir.name) / "analysed"


start_time = dt.datetime.strptime("2024-07-04T10:21:15.000", "%Y-%m-%dT%H:%M:%S.%f").replace(
    tzinfo=dt.timezone.utc
)
end_time = dt.datetime.strptime("2024-07-04T10:21:21.000", "%Y-%m-%dT%H:%M:%S.%f").replace(
    tzinfo=dt.timezone.utc
)

target_estimation(
    data=data,
    config=config,
    output=output_path,
    start_time=start_time,
    end_time=end_time,
    progress=True,
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
