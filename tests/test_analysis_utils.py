import pytest
from hardtarget.analysis.utils import index_from_ts, ts_from_index
from hardtarget.analysis.utils import load_pointing_data
import numpy.testing as npt

def test_load_pointing_data():

    """
    test upsampling of pointing data to sample_rate
    """
    # data
    data_sample_rate = 1000000 # samples per second
    # pointing 
    pointing_sample_rate = 1/12.8 # 1 sample per 12.8 seconds

    # global start just over 123 seconds after 1970 starts
    origin_ts = 123.456789
        
    # pointing data is assumed to start at origin_ts
    # and have 5 pointing items below covering the first 64 seconds after origin_ts

    class PointingReader:
        def __init__(self, sample_rate):
            self.sample_rate = sample_rate
            self._data = {
                9: {'azimuth': 3, 'elevation': 14},
                10: {'azimuth': 4, 'elevation': 15},
                11: {'azimuth': 5, 'elevation': 16},
                12: {'azimuth': 6, 'elevation': 17},
                13: {'azimuth': 7, 'elevation': 18}
            }

        def read(self, idx_start, idx_end):
            assert idx_start < idx_end
            assert idx_start >= 9
            assert idx_end <= 13
            return ((idx, self._data[idx]) for idx in range(idx_start, idx_end))

    # reader
    reader = PointingReader(pointing_sample_rate)

    # interval
    sec_per_pointing = ts_from_index(1, pointing_sample_rate)
    # start 3 seconds into the second sample (idx 10)
    start_ts = origin_ts + 1*sec_per_pointing + 3
    # end 3 seconds into the fourth sample (idx 12)
    end_ts = origin_ts + 3*sec_per_pointing + 3
    interval = [start_ts, end_ts]

    # load pointing data
    data = load_pointing_data(reader, interval, data_sample_rate)

    # test pointing data
    assert data.ndim == 2
    rows, cols = data.shape
    assert cols == 2
    assert rows == index_from_ts(2*sec_per_pointing, data_sample_rate)
    npt.assert_array_equal(data[0], [4,15])
    npt.assert_array_equal(data[int(rows/2)], [5,16])
    npt.assert_array_equal(data[-1], [6,17])


