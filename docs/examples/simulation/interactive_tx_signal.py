# # Interactive tx simulation
# ---

import argparse

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider
from radardef.radar_stations.eiscat.experiments import load_radar_code

from hardtarget.constants import ReceiverChainModel
from hardtarget.data_simulation.tx_model import tx_signal_model

filter_options = [x.value for x in ReceiverChainModel]
parser = argparse.ArgumentParser()
parser.add_argument("model", choices=filter_options)
args = parser.parse_args()
fir_filter = ReceiverChainModel(args.model)


# First we define a simple code
# ```
#   ‾‾‾‾‾|  |‾‾| |‾| |‾
#        |__|  |_| |_|
# ```
barker13 = np.array(
    [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
    dtype=np.float64,
)

# TODO: this info should probably be easily avalible trough the choise of recevier chain model
signal_decimation = {
    ReceiverChainModel.b414d15_gaus: 15,
    ReceiverChainModel.mu2004: 120,
    ReceiverChainModel.none: 1,
}
sample_time_usec = {
    ReceiverChainModel.b414d15_gaus: 1,
    ReceiverChainModel.mu2004: 6,
    ReceiverChainModel.none: 6,
}
baud_lengths_usec = {
    ReceiverChainModel.b414d15_gaus: 30,
    ReceiverChainModel.mu2004: 12,
    ReceiverChainModel.none: 12,
}
codes = {
    ReceiverChainModel.b414d15_gaus: load_radar_code("leo_bpark")[0, :],
    ReceiverChainModel.mu2004: barker13,
    ReceiverChainModel.none: barker13,
}

tx_samples = len(codes[fir_filter]) * int(baud_lengths_usec[fir_filter] / sample_time_usec[fir_filter])
ipp_samps = tx_samples * 5

sub_resolution = np.arange(0, 2, 0.01)

tx_base = tx_signal_model(
    code=codes[fir_filter],
    baud_length_usec=baud_lengths_usec[fir_filter],
    t_samp_usec=sample_time_usec[fir_filter] / signal_decimation[fir_filter],
    start_samp=tx_samples * signal_decimation[fir_filter] * 0.5,
    ipp_samps=ipp_samps * signal_decimation[fir_filter],
    read_length=tx_samples * signal_decimation[fir_filter] * 2,
    filt=ReceiverChainModel.none,
    sub_resolution=sub_resolution * signal_decimation[fir_filter],
    bandwidth=None,
)
signals = {}


def signal_for(selected_filter):
    if selected_filter not in signals:
        signals[selected_filter] = tx_signal_model(
            code=codes[fir_filter],
            baud_length_usec=baud_lengths_usec[fir_filter],
            t_samp_usec=sample_time_usec[fir_filter],
            start_samp=tx_samples * 0.5,
            ipp_samps=ipp_samps,
            read_length=tx_samples * 2,
            filt=selected_filter,
            sub_resolution=sub_resolution,
            bandwidth=None,
        )
    return signals[selected_filter]


tx = signal_for(fir_filter)
ind = 0
zoomed = True
fig, ax = plt.subplots()

# Center the view on a phase change and show the same number of receiver
# samples for every filter.
transition_baud = np.flatnonzero(np.diff(codes[fir_filter]) != 0)[0] + 1
transition_sample = (
    tx_samples * 0.5 + transition_baud * baud_lengths_usec[fir_filter] / sample_time_usec[fir_filter]
)
zoom_half_width = 5
zoom_start = transition_sample - zoom_half_width
zoom_stop = transition_sample + zoom_half_width
sample = np.arange(tx.shape[0])
sample_orig = np.arange(tx_base.shape[0]) / signal_decimation[fir_filter]
extent = np.logical_and(sample >= zoom_start, sample <= zoom_stop)
extent_o = np.logical_and(sample_orig >= zoom_start, sample_orig <= zoom_stop)

(ls,) = ax.plot(sample[extent], np.real(tx[extent, ind]), "o-", label="filtered")
(ls_base,) = ax.plot(sample_orig[extent_o], np.real(tx_base[extent_o, ind]), "-", label="original")
ax.set_xlim(zoom_start, zoom_stop)
ax.set_xlabel("Sample")
ax.set_ylim(-1.2, 1.2)
ax.legend()


def update_title():
    sample_rate_mhz = 1 / sample_time_usec[fir_filter]
    ax.set_title(
        f"Decimation: {signal_decimation[fir_filter]} | "
        f"Sample rate: {sample_rate_mhz:g} MHz | "
        f"Baud length: {baud_lengths_usec[fir_filter]:g} us"
    )


def draw():
    if zoomed:
        ls.set_data(sample[extent], np.real(tx[extent, ind]))
        ls_base.set_data(sample_orig[extent_o], np.real(tx_base[extent_o, ind]))
        ax.set_xlim(zoom_start, zoom_stop)
    else:
        ls.set_data(sample, np.real(tx[:, ind]))
        ls_base.set_data(sample_orig, np.real(tx_base[:, ind]))
        ax.set_xlim(sample[0], sample[-1])
    ax.relim()
    ax.autoscale_view(scalex=False)
    update_title()
    fig.canvas.draw_idle()


def update_d(val):
    global ind
    ind = np.argmin(np.abs(sub_resolution - val))
    draw()


fig.tight_layout()
fig.subplots_adjust(bottom=0.14, top=0.86)
update_title()

axcolor = "lightgoldenrodyellow"
ax_d = plt.axes([0.1, 0.05, 0.2, 0.03], facecolor=axcolor)
s_d = Slider(ax_d, "Offset", 0, 2, valinit=0, valstep=0.01)
s_d.on_changed(update_d)

ax_zoom = plt.axes([0.48, 0.90, 0.20, 0.045])
zoom_button = Button(ax_zoom, "Show full signal")


def toggle_zoom(_event):
    global zoomed
    zoomed = not zoomed
    zoom_button.label.set_text("Show full signal" if zoomed else "Zoom to transition")
    draw()


zoom_button.on_clicked(toggle_zoom)

plt.show()
