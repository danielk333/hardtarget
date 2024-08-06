import pytest
from hardtarget.radars.eiscat.util import EiscatDRFWriter, EiscatDRFReader
import datetime as dt
import time
import numpy as np
import numpy.testing as npt
from pathlib import Path


####################################################################
# RANDOM DATA WRITER
####################################################################

BATCH_LEN = 100

def write(dst, ts_origin_sec):
    """
    write 2 hours of random data with vectors of 100 samples
    1 Hz sample rate and vector length 100 means 100 seconds of data per write operation.
    2 hours of data is then 2*3600 samples == 2*36 write operations, 
    written data divided into 2*36 100 second files = 72 files 
    """

    # create writer
    writer = EiscatDRFWriter(dst,
        ts_origin_sec = ts_origin_sec,
        sample_rate_numerator = 1, # 1 sample per sec
        sample_rate_denominator = 1,
        sample_batch_length = BATCH_LEN, 
        sample_file_cadence_millisecs = 100*1000 # 100 seconds per file
    )

    # make random data
    def get_data(samples):
        dtype = np.dtype([('r', '<i2'), ('i', '<i2')])
        size = (samples, 1)
        real = np.random.randint(-32768, 32767, size=size, dtype='<i2')
        imag = np.random.randint(-32768, 32767, size=size, dtype='<i2')
        arr = np.zeros(size, dtype=dtype)
        arr['r'] = real
        arr['i'] = imag
        return arr

    def get_pointing(rows):
        azimuth = np.random.uniform(low=0.0, high=180.0, size=(rows, 1)).astype(np.float64)
        elevation = np.random.uniform(low=0.0, high=90.0, size=(rows, 1)).astype(np.float64)
        return np.hstack((azimuth, elevation))

    # write
    n_batches = 36*2
    pointing = get_pointing(n_batches)
    data = get_data(n_batches * BATCH_LEN)
    for i in range(n_batches):
        # write data
        batch = data[i*BATCH_LEN: (i+1)*BATCH_LEN]
        # write azimuth, elevation meta data
        azimuth, elevation = pointing[i]
        writer.write(batch, azimuth, elevation)
    writer.close()

    return data, pointing


####################################################################
# DATETIMES
####################################################################


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


####################################################################
# TEST EISCAT DRF
####################################################################

@pytest.mark.parametrize("ts_origin_sec", [
    TS_ORIGIN_SEC_ALIGNED,
    TS_ORIGIN_SEC_MISALIGNED,
    TS_ORIGIN_SEC_NOW
])
def test_eiscat_drf(tmpdir, ts_origin_sec):
    
    assert tmpdir.exists()

    # write test data
    wr_data, wr_pointing = write(tmpdir, ts_origin_sec)        

    # reader
    reader = EiscatDRFReader(
        tmpdir, 
        ts_origin_sec=ts_origin_sec,
        sample_batch_length = BATCH_LEN
    )

    # read 2 hours from ts_origin_sec
    start_ts = ts_origin_sec
    end_ts = (dt.datetime.fromtimestamp(start_ts) + dt.timedelta(hours=2)).timestamp()
    idx_start = reader.get_index_from_ts(start_ts)
    idx_end = reader.get_index_from_ts(end_ts)
    rd_data = reader.read_data(idx_start, idx_end)
    rd_pointing, rd_indexes = reader.read_pointing(idx_start, idx_end)

    # compare written data to read data
    npt.assert_array_equal(wr_data, rd_data)
    npt.assert_array_equal(wr_pointing, rd_pointing)

    # read by bounds
    bounds = reader.get_bounds()
    rd_data = reader.read_data(*bounds)
    rd_pointing, rd_indexes = reader.read_pointing(*bounds)

    # compare written data to read data
    npt.assert_array_equal(wr_data, rd_data)
    npt.assert_array_equal(wr_pointing, rd_pointing)


####################################################################
# MAIN
####################################################################

if __name__ == "__main__":
    pytest.main()