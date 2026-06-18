# # Simulated Tx signal
# ---
# In the cases when there is no tx signal available the tx signal can be modeled. From the metadata of the
# measurement the tx signal code is available, using this we can model the signal.

import numpy as np
from matplotlib import pyplot as plt

from hardtarget.data_simulation.tx_model import tx_signal_model
from hardtarget.constants import FIRFilter

# First we define a simple code
# ```
#   ‾‾‾‾‾|  |‾‾| |‾| |‾
#        |__|  |_| |_|
# ```

code = np.array(
    [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
    dtype=np.float64,
)

# Lets say we have a ipp with 56 samples, we want to read all of them and we start at 0. We should then get
# a signal similar to the one we inserted, but decimated. The source of the signal has a baud length of 12 us,
# but we want to sample the signal at 6 us, thus we will oversample it.

t_samp_usec = 6
baud_length_usec = 12
ipp_samples = len(code) * int(baud_length_usec/t_samp_usec)
tx = tx_signal_model(
    code=code,
    baud_length_usec=baud_length_usec,
    t_samp_usec=t_samp_usec,
    tx_start_samp=0,
    ipp_samps=ipp_samples,
    read_length=ipp_samples,
    fir_filter=FIRFilter.mu2004,
    bandwidth=1e6,
)

fig, ax = plt.subplots()
ls, = ax.plot(np.real(tx))
ax.plot(np.imag(tx), c=ls.get_color(), ls="--")

# Now in some cases we want to increase the resolution of the signal, to do this *sub_resolution* is
# introduced. With sub resolution alternative tx signals are simulated for the samples between the "real"
# samples. In the example below we set the subresolution to 4, meaning we will get additional data from each
# 0.25 decimal inbetween each sample:

sub_resolution = 4
tx = tx_signal_model(
    code=code,
    baud_length_usec=baud_length_usec,
    t_samp_usec=t_samp_usec,
    tx_start_samp=0,
    ipp_samps=ipp_samples,
    read_length=ipp_samples,
    sub_resolution=sub_resolution,
    fir_filter=FIRFilter.mu2004,
    bandwidth=1e6,
)
fig, ax = plt.subplots()
for ind in range(tx.shape[1]):
    ls, = ax.plot(np.real(tx[:, ind]), label=f"{ind}")
    ax.plot(np.imag(tx[:, ind]), c=ls.get_color(), ls="--")
ax.legend()
plt.show()

# This is used during the analysis part for unknown tx signals, the subresolution is configurable in the .ini
# file, more details can be found at [Config parameters](../stuff/config_parameters.md).
