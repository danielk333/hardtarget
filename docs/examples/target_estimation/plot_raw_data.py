# # Visualization of raw data:
# ---
# Simple example on how the raw rti function can be utilized.

import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from radardef import RadarDef

from hardtarget import plotting

# Workaround to make jupyter notebook find utils
sys.path.insert(1, str(Path(os.path.abspath("")) / "docs" / "examples" / "extras"))
import utils

# ## Prerequisites
# ---

# Data to analyse

tmp_dir = tempfile.TemporaryDirectory()
raw_path = Path(tmp_dir.name) / "raw"
raw_path.mkdir(parents=True, exist_ok=True)
converted_path = Path(tmp_dir.name) / "converted"
raw_data = utils.download_test_data(raw_path)

# Convert the data to a usable format.

data = utils.convert_test_data(raw_data, converted_path)[0]

# Read data

reader = RadarDef().load_data(data, cache=False)
assert reader is not None

# ## Plot
# ---
# So what is seen here is the power for each rx sample for each ipp on the logarithmic scale, for a specific
# set of ipps

start_time = int(reader.exp_def.t_ipp_usec * 40) * 1e-6
end_time = start_time + int(reader.exp_def.t_ipp_usec * 150) * 1e-6
fig, ax = plt.subplots()
ax, handles = plotting.rti(
    ax,
    reader,
    start_time=start_time,
    end_time=end_time,
    axis_units=True,
    log=True,
    keep_tx=True,
    relative_time=True,
)

plt.show()
