# # Analyse and Inspect
# ---
# How to trigger analysis and then inspect the output

import os
import sys
import tempfile
from pathlib import Path

from matplotlib import gridspec
from matplotlib import pyplot as plt
from radardef import RadarDef

from hardtarget import analyse, load_analysed_data
from hardtarget.plotting import mf_analysis

# Workaround to make jupyter notebook find utils
sys.path.insert(1, str(Path(os.path.abspath("")) / "docs" / "examples" / "analysis"))
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
raw_data = utils.download_test_data(Path(tmp_dir.name) / "raw")

# Convert the data to a usable format.

radars = RadarDef()
source_format = radars.get_source_format(raw_data)
target_formats = radars.available_target_formats(source_format)
converted_files = RadarDef().convert(raw_data, target_formats[0], str(converted_path))
data = converted_files[0]

# ## Analyse data
# ---
# Minimal configuration for analysis, progress, start_time, end_time and relative time are optionals.
# In this case we choose to visualize the 500 first ipps of the measurement, acceleration has been ignored
# for this example.

output_path = Path(tmp_dir.name) / "analysed"
analyse(
    path=data,
    config=config,
    output=output_path,
    start_time=0,
    end_time=500 * 3120,  # 3120 = t_ipp_usec
    relative_time=True,
)

# ## Plot results
# ---
# First load results

data_generator = load_analysed_data(output_path)

# Then we can plot visualize it with different plots

out_data, exp_params, cfg_params, pro_params = list(data_generator)[0]

# Plot peaks, this visualizes the peaks  of the hardtargets range, velocity
# and acceleration over time (red is marking the detections). Furthermore the the signal to noise ratio is
# shown.

fig, axes = plt.subplots(2, 2)
mf_analysis.plot_peaks(
    axes,
    out_data,
    exp_params,
    cfg_params,
    pro_params,
)
fig.set_size_inches(10, 10)

# Plot detections, this visualizes only the hardtarget detections from the measurement
# (red dots from previous plot).
# Additionaly it also adds the acceleration and range gate relative to the range for each detection.

fig, axes = plt.subplots(2, 3)
mf_analysis.plot_detections(
    axes,
    out_data,
    exp_params,
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
mf_analysis.plot_map(
    axes,
    out_data,
    exp_params,
    cfg_params,
    pro_params,
)
fig.set_size_inches(10, 10)


plt.show()
tmp_dir.cleanup()
