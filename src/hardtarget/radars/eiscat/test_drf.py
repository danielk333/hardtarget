import numpy as np
from hardtarget.radars.eiscat.util import EiscatDRFWriter, EiscatDRFReader
# import digital_rf as drf
import datetime as dt
import time

DST = "drf"
BATCH_LEN = 100

def make_ts_from_str(datetime_str):
    specific_datetime = dt.datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S")
    # Convert to UTC (assuming the datetime is in local time)
    datetime_utc = specific_datetime.replace(tzinfo=dt.timezone.utc)
    return datetime_utc.timestamp()

datetime_str = "2024-07-10T12:00:00"
funky_datetime_str = "2024-07-10T12:03:03"

#TS_ORIGIN_SEC = make_ts_from_str(datetime_str)
TS_ORIGIN_SEC = make_ts_from_str(funky_datetime_str)
# TS_ORIGIN_SEC = dt.datetime.now(dt.timezone.utc).timestamp()


def write():

    def get_data(vector_length):
        data_dtype = np.int16
        data_shape = (vector_length, 2)
        return np.random.randint(-32768, 32767, size=data_shape, dtype=np.int16)

    def get_pointing(vector_length):
        azimuth = np.random.uniform(low=0.0, high=180.0, size=(vector_length, 1)).astype(np.float64)
        elevation = np.random.uniform(low=0.0, high=90.0, size=(vector_length, 1)).astype(np.float64)
        return np.hstack((azimuth, elevation))

    writer = EiscatDRFWriter(DST,
        ts_origin_sec = TS_ORIGIN_SEC,
        sample_rate_numerator = 1,
        sample_rate_denominator = 1,
        sample_batch_length = BATCH_LEN,
        sample_file_cadence_millisecs = 100*1000 # 100 seconds per file
    )

    # write 2 hours of data in 100 sample vectors
    # 1 Hz and vector length 100 means 100 seconds of data per write operation.
    # 36 * 2 write operations should be 2 hours of data, divided into 2*36 100 second files = 72 files 
    pointing_data = get_pointing(36*2)
    for i in range(36*2):
        # write data
        uhf_batch = get_data(BATCH_LEN)
        azimuth, elevation = pointing_data[i]
        writer.write(uhf_batch, azimuth, elevation)
    writer.close()



def read_by_time():

    reader = EiscatDRFReader(
        DST, 
        ts_origin_sec=TS_ORIGIN_SEC,
        sample_batch_length = BATCH_LEN
    )

    bounds = reader.get_bounds()
    print("--idx bounds", bounds)
    print("--ts bounds", reader.get_ts_bounds())
    print("--str bounds", reader.get_str_bounds())

    # read 2 hours of data from specified time
    start_ts = TS_ORIGIN_SEC
    end_ts = (dt.datetime.fromtimestamp(start_ts) + dt.timedelta(hours=2)).timestamp()

    idx_first = reader.get_index_from_ts(start_ts)
    idx_last = reader.get_index_from_ts(end_ts) - 1

    pointing, indexes = reader.read_pointing(idx_first, idx_last)
    print(pointing)
    print(len(pointing))
    print(pointing.dtype)


def read_by_bounds():

    reader = EiscatDRFReader(
        DST, 
        ts_origin_sec=TS_ORIGIN_SEC,
        sample_batch_length = BATCH_LEN
    )
    bounds = reader.get_bounds()
    print("--idx bounds", bounds)
    print("--ts bounds", reader.get_ts_bounds())
    print("--str bounds", reader.get_str_bounds())

    pointing, indexes = reader.read_pointing(*bounds)
    print(pointing)
    print(len(pointing))
    print(pointing.dtype)

    print(indexes)
    print(len(indexes))


def read_data():

    reader = EiscatDRFReader(
        DST, 
        ts_origin_sec=TS_ORIGIN_SEC,
        sample_batch_length = BATCH_LEN
    )
    bounds = reader.get_bounds()
    print("--idx bounds", bounds)
    print("--ts bounds", reader.get_ts_bounds())
    print("--str bounds", reader.get_str_bounds())

    data = reader.read_data(*bounds)
    print(len(data))
    print(data[:5])
    print("..")
    print(data[-5:])



if __name__ == '__main__':

    # write()
    # read_by_time()
    read_by_bounds()
    # read_data()