import pytest
from hardtarget.radars.eiscat.util import EiscatDRFWriter, EiscatDRFReader
import datetime as dt
import time
import numpy as np
from pathlib import Path
import shutil

DST = "/tmp/eiscatdrf"
VECTOR_LEN = 100


def make_ts_from_str(datetime_str):
    """
    make utc timestamp (seconds) from human readable datetime string (local time)
    """
    specific_datetime = dt.datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S")
    # Convert to UTC (assuming the datetime is in local time)
    datetime_utc = specific_datetime.replace(tzinfo=dt.timezone.utc)
    return datetime_utc.timestamp()


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
        sample_batch_length = VECTOR_LEN, 
        sample_file_cadence_millisecs = 100*1000 # 100 seconds per file
    )

    # make random data
    def get_data(vector_length):
        data_dtype = np.int16
        data_shape = (vector_length, 2)
        return np.random.randint(-32768, 32767, size=data_shape, dtype=np.int16)

    def get_pointing(vector_length):
        azimuth = np.random.uniform(low=0.0, high=180.0, size=(vector_length, 1)).astype(np.float64)
        elevation = np.random.uniform(low=0.0, high=90.0, size=(vector_length, 1)).astype(np.float64)
        return np.hstack((azimuth, elevation))

    pointing_data = get_pointing(36*2)

    batches = []
    for i in range(36*2):
        # write data
        uhf_batch = get_data(VECTOR_LEN)
        # write azimuth, elevation meta data
        azimuth, elevation = pointing_data[i]
        writer.write(uhf_batch, azimuth, elevation)
        batches.append((uhf_batch, azimuth, elevation))
    writer.close()

    return batches




class TestDrf:

    def test_specific_time(self):

        datetime_str = "2024-07-10T12:00:00"
        ts_origin_sec = make_ts_from_str(datetime_str)
        batches = write(DST, ts_origin_sec)        
        
        assert len(batches) == 72

        # read 2 hours from ts_origin_sec
        reader = EiscatDRFReader(
            DST, 
            ts_origin_sec=ts_origin_sec,
            sample_batch_length = VECTOR_LEN
        )

        # read 2 hours of data from specified time
        start_ts = ts_origin_sec
        end_ts = (dt.datetime.fromtimestamp(start_ts) + dt.timedelta(hours=2)).timestamp()

        idx_first = reader.get_index_from_ts(start_ts)
        idx_last = reader.get_index_from_ts(end_ts) - 1

        pointing, indexes = reader.read_pointing(idx_first, idx_last)
        assert len(pointing) == 72
        print(pointing)
        print(len(pointing))
        print(pointing.dtype)


        # cleanup 
        shutil.rmtree(DST)


