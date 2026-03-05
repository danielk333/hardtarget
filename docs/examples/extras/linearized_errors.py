import numpy as np

from hardtarget.data_simulation.errors import linearized_mle_covariance

cov = linearized_mle_covariance(snr_db=10, range0=2000e3, vel0=0.3e3, acel0=0.1e3)
print(f"Covariance matrix: {cov}")
print(f"Standard deviations: {np.sqrt(np.diag(cov))}")
