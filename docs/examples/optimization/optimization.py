# # Optimize
# ---
# How to trigger analysis and then inspect the output

# ##Prerequisites
# ---

import os
import sys
import tempfile
from pathlib import Path

from matplotlib import pyplot as plt

from hardtarget import analyse, load_analysed_data
from hardtarget.constants import AnalysisMethod, OptimizationMethod, TargetEstimationMethod
from hardtarget.plotting import optimization_plots, target_estimation_plots

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
raw_data = utils.download_test_data(Path(tmp_dir.name) / "raw")

# Convert the data to a usable format.

data = utils.convert_test_data(raw_data, converted_path)[0]

# ## Analyse data
# ---
# Minimal configuration for analysis, progress, start_time, end_time and relative time are optionals.
# In this case we choose to visualize the 500 first ipps of the measurement
# Note that as we are using the general analyse function we have to manually define the method.

output_path = Path(tmp_dir.name) / "analysed"
analyse(
    data=data,
    config=config,
    method=AnalysisMethod.target_estimation,
    method_lib=TargetEstimationMethod.fgmf,
    output=output_path,
    start_time=0,
    end_time=(500 * 3120) * 1e-6,  # 3120 = t_ipp_usec
    relative_time=True,
)

# ## Plot results
# ---
# First load results

data_generator = load_analysed_data(output_path)

# Then we can visualize it with different plots

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


# ## Optimization
# ---
# If we want to increase the accuracy we can optimize the analysis, what is done then is that we load the
# previous analysis and optimize it. To specify what analysed file we want to optimize we need to configure
# the already available .ini file to point at the output of the previous analysis file.

config_str = f"""
        [processing]
            n_ipp=1
            ipp_offset=0
            min_range_gate=81
            max_range_gate=138
            range_gate_step=1
            num_cohints_per_file=500
            tx_amp_limit = 0.2
            node_gpus=1
            cache=False
        [target_estimation]
            min_acceleration=0
            max_acceleration=0
            range_gate_sub_resolution = 10
            frequency_decimation=1
            clutter_length=1500
        [optimization]
            path = {str(output_path)}
        """
#
tmp_config = tempfile.NamedTemporaryFile(mode="w+")
tmp_config.write(config_str)
tmp_config.seek(0)
tmp_config_path = tmp_config.name

# Now we can run the optimization process over the same time period as before.
# Note that the output path is different, otherwise it will add the optimized vars to available file
# This is not compatible with the optimization plot

optimization_path = Path(tmp_dir.name) / "opt"
analyse(
    data=data,
    config=tmp_config_path,
    method=AnalysisMethod.optimize,
    method_lib=OptimizationMethod.optimize_grid_gmf,
    output=optimization_path,
    start_time=0,
    end_time=(500 * 3120) * 1e-6,  # 3120 = t_ipp_usec
    relative_time=True,
)

optimized_data = load_analysed_data(optimization_path)
out_data_opt, exp_def_opt, cfg_params_opt, pro_params_opt = list(optimized_data)[0]

# ## Optimization plot
# ---
# The cross markings marks the optimized results.
fig, axes = plt.subplots(2, 2)
target_estimation_plots.plot_peaks(
    axes,
    out_data,
    exp_def,
    cfg_params,
    pro_params,
)
optimization_plots.plot_optimization_peaks(
    axes,
    out_data_opt,
    exp_def_opt,
    cfg_params_opt,
    pro_params_opt,
    snr_dB_limit=15.0,
)
fig.set_size_inches(10, 10)


plt.show()
tmp_dir.cleanup()
