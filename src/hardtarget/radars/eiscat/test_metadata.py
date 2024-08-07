from pathlib import Path
import shutil
import digital_rf
import numpy as np
import datetime as dt
import time

def get_pointing(rows):
    azimuth = np.random.uniform(low=0.0, high=180.0, size=(rows, 1)).astype(np.float64)
    elevation = np.random.uniform(low=0.0, high=90.0, size=(rows, 1)).astype(np.float64)
    return np.hstack((azimuth, elevation))

def make_ts_from_str(datetime_str):
    """
    make utc timestamp (seconds) from human readable datetime string (local time)
    """
    specific_datetime = dt.datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S")
    # Convert to UTC (assuming the datetime is in local time)
    datetime_utc = specific_datetime.replace(tzinfo=dt.timezone.utc)
    return datetime_utc.timestamp()


TS_ORIGIN_SEC_ALIGNED = make_ts_from_str("2024-07-10T12:00:00")
TS_ORIGIN_SEC_MISALIGNED = make_ts_from_str("2024-07-10T12:03:03")
TS_ORIGIN_SEC_NOW = dt.datetime.now(dt.timezone.utc).timestamp()

# metadir
metadir = Path("/tmp/eiscatpointing/")
shutil.rmtree(metadir, ignore_errors=True)
metadir.mkdir(parents=True, exist_ok=True)

POINTING_SUBDIR_CADENCE_SECS = 3600 # 1 dir per hour
POINTING_FILE_CADENCE_SECS = 3600 # 1 hour per file


SAMPLE_RATE_NUMERATOR = 1000000 
SAMPLE_RATE_DENOMINATOR = 1
# sample batch length is 12.8 seconds worth of samples
SAMPLE_BATCH_LENGTH = 12800000

# meta writer
dmw = digital_rf.DigitalMetadataWriter(
    str(metadir),
    POINTING_SUBDIR_CADENCE_SECS,
    POINTING_FILE_CADENCE_SECS,
    SAMPLE_RATE_NUMERATOR,
    SAMPLE_RATE_DENOMINATOR * SAMPLE_BATCH_LENGTH,
    "pointing",
)


ts_origin_sec = TS_ORIGIN_SEC_ALIGNED
start_idx = int(np.uint64(ts_origin_sec * dmw.get_samples_per_second()))

# write
n_batches = 36*2
pointing = get_pointing(n_batches)
for i in range(n_batches):
    idx_arr = np.array([start_idx + i]) 
    # write azimuth, elevation meta data
    azimuth, elevation = pointing[i]
    data_dict = {}
    data_dict["azimuth"] = azimuth
    data_dict["elevation"] = elevation
    dmw.write(idx_arr, data_dict)


