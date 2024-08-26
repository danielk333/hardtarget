import pytest
from hardtarget.utils import index_from_ts, ts_from_index
from hardtarget.analysis.utils import load_metadata, load_pointing_data
import numpy as np
import numpy.testing as npt
import hardtarget.digitalrf_wrapper as drf_wrapper
from pathlib import Path

PROJECT = Path("/cluster/projects/p106119-SpaceDebrisRadarCharacterization")
DRF = PROJECT / "drf/leo_bpark_2.1u_NO-20220408-UHF"


def test_upsample():

    """
    test upsampling of pointing data to sample_rate
    """
    # data
    target_rate = 1000000 # samples per second
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

    def target_value(item):
        return np.array([item['azimuth'], item['elevation']])

    # load pointing data
    data = load_metadata(reader, interval, target_rate, target_value)

    # test pointing data
    assert data.ndim == 2
    rows, cols = data.shape
    assert cols == 2
    assert rows == index_from_ts(2*sec_per_pointing, target_rate)
    npt.assert_array_equal(data[0], [4,15])
    npt.assert_array_equal(data[int(rows/2)], [5,16])
    npt.assert_array_equal(data[-1], [6,17])


@pytest.mark.skipif(not DRF.exists(), reason="Local file is missing")
def test_load_pointing_data():

    # TODO - switch to DRF on the cluster - right now these lack pointing info

    # DRF = "/cluster/home/inar/Data/hardtarget/leo_bpark_2.1u_NO@uhf_drf"
 
    ipp = 20000 # inter-pulse period in samples
    n_ipp = 10 # number of inter-pulse periods 
    num_cohints_per_file = 10 # number of coherent intergration periods in a file
    sample_rate = 1000000 # samples per second

    # integration_rate : coherent integration periods (per second)
    samples_per_integration = ipp * n_ipp
    integration_rate = sample_rate / samples_per_integration

    # task rate : tasks per second
    samples_per_task = ipp * n_ipp * num_cohints_per_file
    task_rate = sample_rate / samples_per_task

    # drf reader
    chnl = "pointing"
    reader = drf_wrapper.DigitalRFReader(DRF, "uhf")

    # specific task
    task_idx = 273

    # ts offset - timestamp assosicated with task 0
    ts_offset_sec = ts_from_index(reader.get_bounds()[0], reader.sample_rate)

    pointing = load_pointing_data(task_idx, DRF, "pointing", task_rate, ts_offset_sec, integration_rate)

    assert len(pointing) == num_cohints_per_file

    # ask for data outside bounds
    pointing = load_pointing_data(-4, DRF, "pointing", task_rate, ts_offset_sec, integration_rate)
    expected = np.full((num_cohints_per_file, 2), np.nan)
    npt.assert_array_equal(pointing, expected)


