# # Analyse and Inspect
# ---
# How to trigger analysis and then inspect the output

import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from radardef import RadarDef

import tests.utils as utils
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

data = utils.convert_test_data(raw_data, converted_path)[0]


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

# Then we can visualize it with different plots

fig, ax = plt.subplots(2, 2)
plotting.plot_event_detection(ax, out_data, pro_params)
#
# plot raw data power
_, handles = plotting.rti(
    ax[1, 1],
    reader,
    start_time=start_time,
    end_time=end_time,
    axis_units=True,
    log=True,
    relative_time=True,
    colorbar=False,
)
#
plt.show()
tmp_dir.cleanup()
