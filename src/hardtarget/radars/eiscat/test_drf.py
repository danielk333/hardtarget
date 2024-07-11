import numpy as np
from hardtarget.radars.eiscat.util import EiscatDRFWriter, EiscatDRFReader
import digital_rf as drf
import datetime as dt
import time

DST = "drf"
BATCH_LEN = 100

datetime_str = "2024-07-10T12:00:00"
specific_datetime = dt.datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S")
# Convert to UTC (assuming the datetime is in local time)
datetime_utc = specific_datetime.replace(tzinfo=dt.timezone.utc)
ts_origin_sec = datetime_utc.timestamp()


def write_drf(dst, batch_len):

    def get_data(vector_length):
        data_dtype = np.int16
        data_shape = (vector_length, 2)
        return np.random.randint(-32768, 32767, size=data_shape, dtype=np.int16)

    def get_pointing(vector_length):
        azimuth = np.random.randint(0, 180, size=(vector_length, 1), dtype=np.int16).astype(np.float64)
        elevation = np.random.randint(0, 90, size=(vector_length, 1), dtype=np.int16).astype(np.float64)
        return np.hstack((azimuth, elevation))

    


    writer = EiscatDRFWriter(DST,
        # ts_origin_sec = ts_origin_sec,
        uhf_sample_rate_numerator = 1,
        uhf_sample_rate_denominator = 1,
        uhf_sample_batch_length = BATCH_LEN,
        uhf_file_cadence_millisecs = 100*1000 # 100 seconds per file
    )

    # write 2 hours of data in 100 sample vectors
    # 1 Hz and vector length 100 means 100 seconds of data per write operation.
    # 36 * 2 write operations should be 2 hours of data, divided into 2*36 100 second files = 72 files 
    pointing_data = get_pointing(36*2)
    print(pointing_data)
    print(pointing_data.shape)
    for i in range(36*2):
        # write data
        uhf_batch = get_data(BATCH_LEN)
        azimuth, elevation = pointing_data[i] 
        writer.write(uhf_batch, azimuth, elevation)
    writer.close()



def read_drf():

    reader = EiscatDRFReader(DST, uhf_sample_batch_length = BATCH_LEN)

    bounds = reader.get_bounds()
    print("--bounds", bounds)
    print("--ts bounds", reader.get_ts_bounds())
    print("--bounds pointing", reader.get_str_bounds())


    idx_first, idx_last = bounds
    length = idx_last + 1 - idx_first
    pointer_data, sample_indexes = reader.read_pointing(idx_first, length)
    print(pointer_data)
    # print(length)

    
    #uhf_data = reader.read_uhf(idx_first, length)
    #print(uhf_data[:95])
    #print("..")
    #print(uhf_data[-7:])



if __name__ == '__main__':

    # write_drf()
    read_drf()