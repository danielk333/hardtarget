import pytest
import hardtarget.digitalrf_wrapper as drf_wrapper
import datetime as dt
import time
import numpy as np
import numpy.testing as npt
from pathlib import Path
from hardtarget.utils import ts_from_str

####################################################################
# DATETIMES
####################################################################

TS_ORIGIN_SEC_ALIGNED = ts_from_str("2024-07-10T12:00:00.000")
TS_ORIGIN_SEC_MISALIGNED = ts_from_str("2024-07-10T12:03:04.555")

####################################################################
# TEST DRF
####################################################################

@pytest.mark.parametrize("ts_origin_sec", [
    TS_ORIGIN_SEC_ALIGNED,
    TS_ORIGIN_SEC_MISALIGNED
])
def test_drf(tmpdir, ts_origin_sec):
    """
    write 2 hours of random complex data with vectors of 100 samples
    1 Hz sample rate and vector length 100 means 100 seconds of data per write operation.
    2 hours of data is then 2*3600 samples == 2*36 write operations, 
    written data divided into 2*36 100 second files = 72 files 

    read it back to verify correctness
    1) read it back using known timestamps
    2) read it back using data bounds reported by the reader
    """
    # tmpdir
    tmpdir = Path(tmpdir)
    assert tmpdir.exists()

    # setup
    SAMPLE_RATE_NUMERATOR = 1 # 1 Hz 
    SAMPLE_RATE_DENOMINATOR = 1
    SUBDIR_CADENCE_SECS = 3600 # 1 dir per hour
    FILE_CADENCE_SECS = 100 # 100 seconds per file
    DTYPE = np.int16
    BATCH_LEN = 100
    CHNL = "data"

    sample_rate = SAMPLE_RATE_NUMERATOR / float(SAMPLE_RATE_DENOMINATOR)
    start_global_index = ts_origin_sec * sample_rate

    # create writer
    writer = drf_wrapper.DigitalRFWriter(tmpdir, CHNL,
        SAMPLE_RATE_NUMERATOR,
        SAMPLE_RATE_DENOMINATOR,
        DTYPE,
        start_global_index,
        subdir_cadence_secs=SUBDIR_CADENCE_SECS,
        file_cadence_secs=FILE_CADENCE_SECS,
        is_complex=True,
        ts_align_sec=ts_origin_sec
    )

    # make random data
    def get_data(samples):
        """make some random complex data"""
        dtype = np.dtype([('r', '<i2'), ('i', '<i2')])
        size = (samples, 1)
        real = np.random.randint(-32768, 32767, size=size, dtype='<i2')
        imag = np.random.randint(-32768, 32767, size=size, dtype='<i2')
        arr = np.zeros(size, dtype=dtype)
        arr['r'] = real
        arr['i'] = imag
        return arr

    # write
    n_batches = 36*2
    wr_data = get_data(n_batches * BATCH_LEN)
    for i in range(n_batches):
        # write data
        batch = wr_data[i*BATCH_LEN: (i+1)*BATCH_LEN]
        writer.write(batch)
    writer.close()

    # create reader
    reader = drf_wrapper.DigitalRFReader(tmpdir, CHNL, ts_align_sec=ts_origin_sec)

    # read 2 hours from ts_origin_sec
    start_ts = ts_origin_sec
    end_ts = (dt.datetime.fromtimestamp(start_ts) + dt.timedelta(hours=2)).timestamp()
    idx_start = int(reader.index_from_ts(start_ts))
    idx_end = int(reader.index_from_ts(end_ts))
    rd_idx, rd_data = next(iter(reader.read(idx_start, idx_end)))  

    # compare written data to read data
    npt.assert_array_equal(wr_data, rd_data)

    # read by bounds
    idx_start, idx_end = reader.get_bounds()
    rd_idx, rd_data = next(iter(reader.read(idx_start, idx_end)))

    # compare written data to read data
    # no guarantee that these are the same, as there can be padded with Nans



####################################################################
# TEST DRF METADATA
####################################################################


@pytest.mark.parametrize("ts_origin_sec", [
    TS_ORIGIN_SEC_ALIGNED,
    TS_ORIGIN_SEC_MISALIGNED
])
def test_drf_metadata(tmpdir, ts_origin_sec):

    # tmpdir
    tmpdir = Path(tmpdir)

    # SETUP
    SAMPLE_RATE_NUMERATOR = 1000000 
    SAMPLE_RATE_DENOMINATOR = 1
    BATCH_LENGTH = 12800000  # sample batch length is 12.8 seconds worth of samples
    CHNL = "meta"

    # create writer
    writer = drf_wrapper.DigitalMetadataWriter(
        tmpdir, CHNL,
        SAMPLE_RATE_NUMERATOR,
        SAMPLE_RATE_DENOMINATOR * BATCH_LENGTH,
    )

    # random pointing data
    def get_pointing(rows):
        azimuth = np.random.uniform(low=0.0, high=180.0, size=(rows, 1)).astype(np.float64)
        elevation = np.random.uniform(low=0.0, high=90.0, size=(rows, 1)).astype(np.float64)
        return np.hstack((azimuth, elevation))

    # write
    start_idx = int(writer.index_from_ts(ts_origin_sec))
    n_batches = 36*2
    wr_pointing = get_pointing(n_batches)
    for i in range(n_batches):
        azimuth, elevation = wr_pointing[i]
        d = {'azimuth': azimuth, 'elevation': elevation}
        writer.write(start_idx + i, d)

    # create reader
    reader = drf_wrapper.DigitalMetadataReader(tmpdir, CHNL)

    # covert to numpy array
    def convert(values):
        return np.array([(d['azimuth'], d['elevation']) for d in values], dtype=np.float64)

    # read 2 hours from ts_origin_sec
    start_ts = ts_origin_sec
    end_ts = (dt.datetime.fromtimestamp(start_ts) + dt.timedelta(hours=2)).timestamp()
    idx_start = int(reader.index_from_ts(start_ts))
    idx_end = int(reader.index_from_ts(end_ts))

    # read and compare
    indexes, values = zip(*reader.read(idx_start, idx_end))
    rd_pointing = convert(values)
    npt.assert_array_equal(wr_pointing, rd_pointing)

    # read by bounds
    idx_start, idx_end = reader.get_bounds()
    
    # read and compare
    indexes, values = zip(*reader.read(idx_start, idx_end))
    rd_pointing = convert(values)
    npt.assert_array_equal(wr_pointing, rd_pointing)

    # read outside bounds
    result = reader.read(idx_start - 1000, idx_start)
    assert len(result) == 0

    # read partial outside bounds
    result = reader.read(idx_start - 5000, idx_end)
    assert len(result) == n_batches


####################################################################
# TEST METADATA CHANGE
####################################################################

def test_metadata_change(tmpdir):

    """
    verify read where meta data value is changeing during read interval
    """

    # mockup metadata stream

    # create writer
    # sample rate 1 sample per 12.8 sec
    writer = drf_wrapper.DigitalMetadataWriter(
        tmpdir, "mock",
        10,
        128,
    )

    # write 5 samples == 64 seconds
    writer.write(0, {"azimuth": 4})
    writer.write(1, {"azimuth": 4})
    writer.write(2, {"azimuth": 80})
    writer.write(3, {"azimuth": 80})
    writer.write(4, {"azimuth": 80})

    # read back an interval where the value changes
    reader = drf_wrapper.DigitalMetadataReader(tmpdir, "mock")

    result = list(reader.read(1, 3))
    assert len(result) == 2
    assert result[0][1]['azimuth'] == 4
    assert result[1][1]['azimuth'] == 80



####################################################################
# TEST METADATA UPSAMPLE
####################################################################

def test_metadata_upsample():

    pass











####################################################################
# MAIN
####################################################################

if __name__ == "__main__":
    pytest.main()