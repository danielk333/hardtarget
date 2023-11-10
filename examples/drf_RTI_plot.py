import matplotlib.pyplot as plt
import hardtarget

target = "/home/danielk/data/spade/beamparks_raw/leo_bpark_2.1u_NO@uhf_drf/"
reader, params = hardtarget.load_hardtarget_drf(target)


fig, ax = plt.subplots()
ax, handles = hardtarget.plotting.rti(
    ax,
    reader,
    params,
    start_time=2.0,  # s
    end_time=8.0,  # s
    relative_time=True,
    axis_units=True,
    clutter_removal=2500.0e-6,  # s
)

plt.show()
