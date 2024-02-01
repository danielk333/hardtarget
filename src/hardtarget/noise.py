import numpy as np


def estimate_noise_from_dc(dcs):
    """Estimate range dependant noise floor from DC data
    """
    mdcs = [np.nanmedian(dc, axis=0) for dc in dcs]
    dc_range_data = np.nanmedian(np.stack(mdcs, axis=0))
    return dc_range_data


def snr(gmf_values, noise_floor, dB=False):
    """Convert GMF value to SNR based on range dependant noise floor (assumes GMF has dimensions [?,range])
    """
    snr = (np.sqrt(gmf_values) - np.sqrt(noise_floor[None, :]))**2 / noise_floor[None, :]
    if dB:
        return 10 * np.log10(snr)
    else:
        return snr
