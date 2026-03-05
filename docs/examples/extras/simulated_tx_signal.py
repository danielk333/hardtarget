# # Simulated Tx signal
# ---
# In the cases when there is no tx signal available the tx signal can be modeled. From the metadata of the
# measurement the tx signal code is available, using this we can model the signal.

from hardtarget.data_simulation.tx_model import tx_signal_model
from matplotlib import pyplot as plt
import numpy as np

# First we define a simple code
# ```
#   ‾‾‾‾‾|  |‾‾| |‾| |‾
#        |__|  |_| |_|
# ```

code = np.kron(
    np.array(
        [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
        dtype=np.float64,
    ),
    np.ones(2),
).astype(np.float64)

# Lets say we have a ipp with 56 samples, we want to read all of them and we start at 0. We should then get
# a signal similar to the one we inserted, but decimated.

ipp_samples = 56
tx = tx_signal_model(
    code=code,
    ipp_samps=ipp_samples,
    read_length=ipp_samples,
)
plt.plot(np.real(tx))

# Now in some cases we want to increase the resolution of the signal, to do this *sub_resolution* is
# introduced. With sub resolution alternative tx signals are simulated for the samples between the "real"
# samples. In the example below we set the subresolution to 4, meaning we will get additional data from each
# 0.25 decimal inbetween each sample:

sub_resolution = 4
tx = tx_signal_model(
    code=code,
    ipp_samps=ipp_samples,
    read_length=ipp_samples,
    sub_resolution=sub_resolution,
)
fig, ax = plt.subplots()
for ind in range(tx.shape[1]):
    ax.plot(np.real(tx[:, ind]), label=f"{ind}")
ax.legend()
plt.show()

# This is used during the analysis part for unknown tx signals, the subresolution is configurable in the .ini
# file, more details can be found at [Config parameters](../stuff/config_parameters.md).
