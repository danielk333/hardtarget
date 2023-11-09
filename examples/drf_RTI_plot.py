import matplotlib.pyplot as plt
import hardtarget

target = "/home/danielk/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf/"
reader, params = hardtarget.load_hardtarget_drf(target)


fig, ax = plt.subplots()
ax, handles = hardtarget.plotting.rti(
    ax,
    reader,
    params,
    "uhf",
    start_time=2.0,
    end_time=8.0,
    relative_time=True,
    axis_units=True,
)

plt.show()
