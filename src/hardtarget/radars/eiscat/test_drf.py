import digital_rf as drf
import numpy as np
import time
from pathlib import Path


def get_data(vector_length):
    # data
    data_dtype = np.int16
    data_shape = (vector_length, 2)
    return np.random.randint(-32768, 32767, size=data_shape, dtype=np.int16)

def get_pointing():
    pointing_dtype = np.dtype([('a', np.float32), ('e', np.float32)])
    pointing_shape = (1,)
    arr = np.empty(pointing_shape, dtype=pointing_dtype)
    arr['a'] = np.random.randint(0, 180, size=pointing_shape, dtype=np.int16).astype(np.float32)
    arr['e'] = np.random.randint(0, 90, size=pointing_shape, dtype=np.int16).astype(np.float32)
    return arr





OUT = Path("drf")
OUT.mkdir(exist_ok=True)

DATA_SAMPLE_FREQUENCY_NUMERATOR = 100
DATA_SAMPLE_FREQUENCY_DENOMINATOR = 1
NOW = time.time()

POINTING_SAMPLE_FREQUENCY_NUMERATOR = 1
POINTING_SAMPLE_FREQUENCY_DENOMINATOR = 1



# global index
# first sample number after now
# number of seconds since epoch * sample_rate + 1 
DATA_SAMPLE_RATE = float(DATA_SAMPLE_FREQUENCY_NUMERATOR)/DATA_SAMPLE_FREQUENCY_DENOMINATOR
DATA_N0 = int(NOW*DATA_SAMPLE_RATE) + 1 

# data writer
data_writer = drf.DigitalRFWriter(
    str(OUT / "uhf"),  # destination directory
    np.int16,  # dtype string
    3600,  # subdir cadence secs    => one dir per hour
    100000,  # file cadence millisecs => one file per 100 second
    DATA_N0,  # start global index
    DATA_SAMPLE_FREQUENCY_NUMERATOR,  # sample rate numerator
    DATA_SAMPLE_FREQUENCY_DENOMINATOR,  # sample rate denominator
    uuid_str="data",
    compression_level=0,
    checksum=False,
    is_complex=True,
    num_subchannels=1,
    is_continuous=True,
    marching_periods=False, # no marching periods while writing
)

# pointing writer
pointing_writer = drf.DigitalRFWriter(
    str(OUT / "pointing"),  # destination directory
    np.float32,  # dtype string
    3600,  # subdir cadence secs    => one dir per hour
    100000,  # file cadence millisecs => one file per 100 second
    DATA_N0,  # start global index
    DATA_SAMPLE_FREQUENCY_NUMERATOR,  # sample rate numerator
    DATA_SAMPLE_FREQUENCY_DENOMINATOR,  # sample rate denominator
    uuid_str="data",
    compression_level=0,
    checksum=False,
    is_complex=True,
    num_subchannels=1,
    is_continuous=True,
    marching_periods=False, # no marching periods while writing
)




def test_write():
    """
    Quickly write 2 hours of data
    """
    VECTOR_LEN = 100
    # 1 hz and vector length 100 means 100 seconds data per write operation.
    # 36 * 2 write operations should be 2 hours of data 
    for i in range(36*2):
        data = get_data(VECTOR_LEN)
        data_writer.rf_write(data)

    # should produce 1 file per 100 sec - 36 files in each directory



if __name__ == '__main__':
    test_write()
    
