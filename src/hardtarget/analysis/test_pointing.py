from pathlib import Path
from hardtarget.analysis.utils import index_from_ts, ts_from_index
from hardtarget.analysis.utils import load_pointing_data

import hardtarget.digitalrf_wrapper as drf_wrapper


ipp = 20000 # inter-pulse period in samples
n_ipp = 10 # number of inter-pulse periods 
num_cohints_per_file = 10 # number of coherent intergration periods in a file
sample_rate = 1000000 # samples per second


samples_per_task = ipp * n_ipp * num_cohints_per_file
print("samples per task", samples_per_task)

seconds_per_task = samples_per_task / sample_rate
print("seconds per task", seconds_per_task)

# rate of integration periods (per second)
integration_rate = seconds_per_task / num_cohints_per_file
print("integration rate", integration_rate)






"""
# global start data series
global_start_sample = 123456789 # just over 123 seconds after midnight 1970

# bounds corresponding to 1 hour of samples
bounds = [global_start_sample, global_start_sample + index_from_ts(3600, sample_rate)] 
print("bounds", bounds)

# bounds expressed with timestamps
ts_bounds = [ts_from_index(sample, sample_rate) for sample in bounds]
print("ts bounds", ts_bounds)

# total tasks 
total_tasks = (bounds[1] - bounds[0]) / samples_per_task 
print("number of tasks for one hour", int(total_tasks))

# one specific task
task_idx = 273
sample_idx_start = bounds[0] + (task_idx * samples_per_task) 
sample_idx_end = sample_idx_start + samples_per_task
print("task 273 samples:", [sample_idx_start, sample_idx_end])
start_ts = ts_from_index(sample_idx_start, sample_rate)
end_ts = ts_from_index(sample_idx_end, sample_rate)
print("task 273 timestamps", [start_ts, end_ts])
print("duration", end_ts - start_ts)


"""


src = "/cluster/home/inar/Data/hardtarget/leo_bpark_2.1u_NO@uhf_drf"
chnl = "pointing"

# drf reader
reader = drf_wrapper.DigitalMetadataReader(src, chnl)

# bounds
pointing_bounds = reader.get_bounds()
ts_bounds = [ts_from_index(idx, reader.sample_rate) for idx in pointing_bounds]
data_bounds = [index_from_ts(ts, sample_rate) for ts in ts_bounds]

# specific task
task_idx = 273

# time interval of specific task
idx_start = data_bounds[0] + (task_idx * samples_per_task) 
idx_end = sample_idx_start + samples_per_task

# TODO
# note - here I know the task length in integration periods which is 10
# need to read data corresponding to task, and then upsample it to 
# rate of integration periods - so that I get a vector of length 10



# n_tasks
n_tasks = (data_bounds[1] - data_bounds[0]) / samples_per_task 


origin_ts = ts_from_index(data_bounds[0], sample_rate)


print("origin_ts", origin_ts)
print("n_tasks", n_tasks)

















