import datetime as dt

from hardtarget.types import Bounds
from hardtarget.utils.time_conversion import time_interval_to_sample_bound


def test_time_interval_to_sample_bounds():

    time_max_min_us = Bounds(1772090400000000, 1772090580000000)  # 2026-02-26T07:20:00, 2026-02-26T07:23:00

    # Start time of 2026-02-26T07:21:00
    start_time_us = 1772090460000000
    start_time_relative = 60 * 1e6
    start_time_dt = dt.datetime.strptime("2026-02-26T07:21:00", "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=dt.timezone.utc
    )

    # End time of 2026-02-26T07:21:36
    end_time_us = 1772090496000000
    end_time_relative = 96 * 1e6
    end_time_dt = dt.datetime.strptime("2026-02-26T07:21:36", "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=dt.timezone.utc
    )

    sample_rate = 1

    # Test us since epoch
    sample_bounds = time_interval_to_sample_bound(
        time_bounds=time_max_min_us, sample_rate=sample_rate, start_time=start_time_us, end_time=end_time_us
    )
    assert sample_bounds.start == 60
    assert sample_bounds.end == 96

    # test datetime
    sample_bounds = time_interval_to_sample_bound(
        time_bounds=time_max_min_us, sample_rate=sample_rate, start_time=start_time_dt, end_time=end_time_dt
    )
    assert sample_bounds.start == 60
    assert sample_bounds.end == 96

    # Test relative time
    sample_bounds = time_interval_to_sample_bound(
        time_bounds=time_max_min_us,
        sample_rate=sample_rate,
        start_time=int(start_time_relative),
        end_time=int(end_time_relative),
        relative_time=True,
    )
    assert sample_bounds.start == 60
    assert sample_bounds.end == 96

    # Verify unbound
    sample_bounds = time_interval_to_sample_bound(
        time_bounds=time_max_min_us,
        sample_rate=sample_rate,
    )
    assert sample_bounds.start == 0
    assert sample_bounds.end == 180

    assert True
