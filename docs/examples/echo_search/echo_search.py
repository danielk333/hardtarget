# # Analyse and Inspect
# ---
# How to trigger analysis and then inspect the output

import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from radardef import RadarDef

from hardtarget import echo_search, load_analysed_data, plotting

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
raw_data = utils.download_test_data(Path(tmp_dir.name) / "raw")

# Convert the data to a usable format.

data = utils.convert_test_data(raw_data, converted_path)[0]


reader = RadarDef().load_data(data)
assert reader is not None

# ## Analyse data
# ---
# Minimal configuration for analysis, progress, start_time, end_time and relative time are optionals.
# In this case we choose to visualize the 500 first ipps of the measurement

output_path = Path(tmp_dir.name) / "analysed"
start_time = int(reader.experiment.t_ipp_usec * 50)
end_time = start_time + int(reader.experiment.t_ipp_usec * 120)

echo_search(
    data=data,
    config=config,
    output=output_path,
    start_time=start_time,
    end_time=end_time,
    relative_time=True,
)

# ## Plot results
# ---
# First load results

data_generator = load_analysed_data(output_path)
out_data, exp_params, cfg_params, pro_params = list(data_generator)[0]

# Then we can visualize it with different plots

fig, ax = plt.subplots(3, 2)
plotting.plot_echo_search(ax, exp_params, out_data, pro_params, limit=0.7)
#
# plot raw data power in 2 blocks
long_plot = ax[1, 0].get_gridspec()
for axis in ax[2, 0:]:
    axis.remove()
long_plot_ax = fig.add_subplot(long_plot[2, 0:])
_, handles = plotting.rti(
    long_plot_ax,
    reader,
    start_time=start_time,
    end_time=end_time,
    axis_units=True,
    log=True,
    relative_time=True,
    colorbar=False,
)
fig.set_size_inches(12, 10)
#
plt.show()
tmp_dir.cleanup()
