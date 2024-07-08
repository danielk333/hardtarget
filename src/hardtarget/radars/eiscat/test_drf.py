import digital_rf as drf
import numpy as np
from pathlib import Path

# writing parameters
sample_rate_numerator = 100 # 100 Hz sample rate - typically MUCH faster
sample_rate_denominator = 1
dtype_str = 'i2' # short int
sub_cadence_secs = 4  # Number of seconds of data in a subdirectory - typically MUCH larger
file_cadence_millisecs = 400  # Each fill will have up to 400 ms of data
compression_level = 0 # low level of compression
checksum = False # no checksum
is_complex = True # complex values
is_continuous = True
num_subchannels = 1 # only one subchannel
marching_periods = False # no marching periods when writing
uuid = "Fake UUID - use a better one!"
vector_length = 100 # number of samples written for each call - typically MUCH longer


def get_data(vector_length):
    # data
    data_dtype = np.dtype([('real', np.int16),('imag', np.int16)])
    data_shape = (vector_length,)
    arr = np.empty(data_shape, dtype=data_dtype)
    arr['real'] = np.random.randint(-32768, 32767, size=data_shape, dtype=np.int16)
    arr['imag'] = np.random.randint(-32768, 32767, size=data_shape, dtype=np.int16)
    return arr

def get_pointing():
    pointing_dtype = np.dtype([('a', np.float32), ('e', np.float32)])
    pointing_shape = (1,)
    arr = np.empty(pointing_shape, dtype=pointing_dtype)
    arr['a'] = np.random.randint(0, 180, size=pointing_shape, dtype=np.int16).astype(np.float32)
    arr['e'] = np.random.randint(0, 90, size=pointing_shape, dtype=np.int16).astype(np.float32)
    return arr



#arr = get_pointing()
#print(arr.dtype)
#print(arr.shape)
#print(arr)

zz = get_data(vector_length)

def to_i2x16(zz):
    zz2x16 = np.empty((len(zz), 2), dtype=np.int16)
    zz2x16[:, 0] = zz["real"].astype(np.int16)
    zz2x16[:, 1] = zz["imag"].astype(np.int16)
    return zz2x16

print(zz.dtype)
print(zz.shape)


# get the real part from zz





#print(zz[:, 0])

print("to_int")
arr = to_i2x16(zz)
print(arr.dtype)
print(arr.shape)
print(arr)



out = Path("tmp")
out.mkdir(exist_ok=True)

# write 2 hours worth of data

