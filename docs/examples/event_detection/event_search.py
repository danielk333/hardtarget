# # Analyse and Inspect
# ---
# How to trigger analysis and then inspect the output

import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from radardef import RadarDef

from hardtarget import event_detection, load_analysed_data, plotting

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

radars = RadarDef()
source_format = radars.get_source_format(raw_data)
target_formats = radars.available_target_formats(source_format)
converted_files = RadarDef().convert(raw_data, target_formats[0], str(converted_path))
data = converted_files[0]


reader = RadarDef().load_data(data)
assert reader is not None

# ## Analyse data
# ---
# Minimal configuration for analysis, progress, start_time, end_time and relative time are optionals.
# In this case we choose to visualize the 500 first ipps of the measurement

output_path = Path(tmp_dir.name) / "analysed"
start_time = int(reader.meta.experiment.t_ipp_usec * 950)
end_time = start_time + int(reader.meta.experiment.t_ipp_usec * 150)
event_detection(
    path=data,
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

# Then we can plot visualize it with different plots

fig, ax = plt.subplots(2, 2)
# plot raw data power
_, handles = plotting.rti(
    ax[0, 0],
    reader,
    start_time=start_time,
    end_time=end_time,
    axis_units=True,
    log=True,
    relative_time=True,
    colorbar=False,
)
# Max peak
ax[0, 1].plot(np.abs(out_data.max_peak))
ax[0, 1].set_ylabel("Max peak")
# Normalised power
filter = out_data.max_peak > 0.5
ax[1, 0].plot(np.abs(out_data.max_pow_norm))
ax[1, 0].set_ylabel("max_power_ind")
# Ipps power
ax[1, 1].plot(np.abs(out_data.ipps_pow))
ax[1, 1].set_ylabel("Ipp power")
#
plt.show()
tmp_dir.cleanup()
