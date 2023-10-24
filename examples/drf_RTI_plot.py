import matplotlib.pyplot as plt
import digital_rf as drf
import hardtarget
import numpy as np

target = "/home/danielk/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf/"
drf_reader = drf.DigitalRFReader(target)
bounds = drf_reader.get_bounds("uhf")
data_vec = drf_reader.read_vector_1d(bounds[0] + 400000, 600000, "uhf")

fig, ax = plt.subplots()
ax.plot(np.abs(data_vec))

fig, ax = plt.subplots()
ax, handles = hardtarget.plotting.rti(
    ax,
    drf_reader,
    "uhf",
    # start_time="2021-04-12T12:03:55",
    end_time="2021-04-12T12:34:18",
)

plt.show()
