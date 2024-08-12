import pytest
from hardtarget.radars.eiscat.util2 import EiscatDRFWriter, EiscatDRFReader
import datetime as dt
import time
import numpy as np
import numpy.testing as npt
from pathlib import Path

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
# RANDOM SAMPLE DATA WRITER
####################################################################

SAMPLE_RATE_NUMERATOR = 1000000 
SAMPLE_RATE_DENOMINATOR = 1
SAMPLE_BATCH_LENGTH = 12800000  # batch length is 12.8 seconds worth of samples
SAMPLE_SUBDIR_CADENCE_SECS = 3600 # 1 dir per hour
SAMPLE_FILE_CADENCE_SECS = 10 # 10 seconds per file

DATADIR = Path("/tmp/eiscat/")
DTYPE = np.int16
BATCH_LEN = 100

def write_data(drfdir, chnl, ts_origin_sec):
    """
    write 2 hours of random data with vectors of 100 samples
    1 Hz sample rate and vector length 100 means 100 seconds of data per write operation.
    2 hours of data is then 2*3600 samples == 2*36 write operations, 
    written data divided into 2*36 100 second files = 72 files 
    """
    datadir = drfdir / chnl
    datadir.mkdir(parents=True, exist_ok=True)

    # create writer
    writer = EiscatDRFWriter(datadir,
        SAMPLE_RATE_NUMERATOR,
        SAMPLE_RATE_DENOMINATOR,
        DTYPE,
        ts_origin_sec=ts_origin_sec,
        subdir_cadence_secs=SAMPLE_SUBDIR_CADENCE_SECS,
        file_cadence_secs=SAMPLE_FILE_CADENCE_SECS,
        is_complex=True
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


    # write
    n_batches = 36*2
    data = get_data(n_batches * BATCH_LEN)
    for i in range(n_batches):
        # write data
        batch = data[i*BATCH_LEN: (i+1)*BATCH_LEN]
        writer.write(batch)
    writer.close()

    return data




####################################################################
# TEST EISCAT DRF
####################################################################

@pytest.mark.parametrize("ts_origin_sec", [
    TS_ORIGIN_SEC_ALIGNED,
    #TS_ORIGIN_SEC_MISALIGNED,
    #TS_ORIGIN_SEC_NOW
])
def test_eiscat_drf(ts_origin_sec):
    
    tmpdir = Path("/tmp/eiscat")
    chnl = "data"

    assert not tmpdir.exists()
    tmpdir.mkdir(parents=True, exist_ok=True)

    # write test data
    wr_data = write_data(tmpdir, chnl, ts_origin_sec)        

    # reader
    reader = EiscatDRFReader(tmpdir)

    # read 2 hours from ts_origin_sec
    start_ts = ts_origin_sec
    end_ts = (dt.datetime.fromtimestamp(start_ts) + dt.timedelta(hours=2)).timestamp()
    idx_start = reader.index_from_ts(start_ts, chnl)
    idx_end = reader.index_from_ts(end_ts, chnl)

    rd_idx, rd_data = next(iter(reader.read(idx_start, idx_end, chnl)))  

    
    import pprint
    pprint.pprint(rd_data)

    # compare written data to read data
    npt.assert_array_equal(wr_data, rd_data)

    # read by bounds
    #idx_start, idx_end = reader.get_bounds()
    #rd_idx, rd_data = next(iter(reader.read(idx_start, idx_end, chnl)))

    # compare written data to read data
    #npt.assert_array_equal(wr_data, rd_data)


####################################################################
# MAIN
####################################################################

if __name__ == "__main__":
    pytest.main()