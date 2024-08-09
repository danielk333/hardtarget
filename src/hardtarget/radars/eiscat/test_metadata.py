from pathlib import Path
import shutil
import digital_rf
import numpy as np
import datetime as dt
import time


class EiscatDRFMetadataWriter:
    
    """
    Wrapper for DRF MetadataWriter
    """

    def __init__(self, 
                 metadatadir, 
                 sample_rate_numerator,
                 sample_rate_denominator,
                 subdir_cadence_secs=3600,  # 1 dir per hour
                 file_candence_secs=3600,  # 1 hour per file
                 prefix="meta",
                ):

        # sample rate
        self._sample_rate = sample_rate_numerator / float(sample_rate_denominator)

        # meta data writer
        self._writer = digital_rf.DigitalMetadataWriter(
            str(metadatadir),
            subdir_cadence_secs,
            file_candence_secs,
            sample_rate_numerator,
            sample_rate_denominator,
            prefix
        )

    def ts_from_index(self, idx):
        return self.idx / self._sample_rate

    def index_from_ts(self, ts):
        return int(ts * self._sample_rate)

    def write(self, idx, azimuth, elevation):
        d = {
            "azimuth": azimuth,
            "elevation": elevation
        }
        self._writer.write(idx, d)


def get_pointing(rows):
    azimuth = np.random.uniform(low=0.0, high=180.0, size=(rows, 1)).astype(np.float64)
    elevation = np.random.uniform(low=0.0, high=90.0, size=(rows, 1)).astype(np.float64)
    return np.hstack((azimuth, elevation))

def make_ts_from_str(datetime_str):
    """
    make utc timestamp (seconds) from human readable datetime string (local time)
    """
    specific_datetime = dt.datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S")
    # Convert to UTC (assuming the datetime is in local time)
    datetime_utc = specific_datetime.replace(tzinfo=dt.timezone.utc)
    return datetime_utc.timestamp()



class EiscatDRFMetadataReader:

    def __init__(self, 
                 metadatadir
                ):
        
        self._reader = digital_rf.DigitalMetadataReader(str(metadatadir))
        self._sample_rate = self._reader.get_samples_per_second()

    def ts_from_index(self, idx):
        return self.idx / self._sample_rate

    def index_from_ts(self, ts):
        return int(ts * self._sample_rate)

    def get_bounds(self, ts_origin_sec=None):
        """
        return bounds for read function

        NOTE: digital rf read() includes idx_last - which breaks python range convention
        this is compensated for, so that this function returns according to convention
        return (start, end) where end is first sample not in bouds

        NOTE: digital rf read() may return padded values at start and and, originating
        from internal aspects of storage organization in files
        If ts_origin_sec is supplied, this will be used to correct this.

        TODO:
        define ts_origin_sec : is it a skew?

        """
        idx_first, idx_last = self._reader.get_bounds()
        idx_start = idx_first
        idx_end = idx_last + 1

        if ts_origin_sec is not None:
            # convert to time domain
            ts_start = self.get_ts_from_index(idx_start)
            ts_end = self.get_ts_from_index(idx_end)
            # calculate padding in time domain
            file_cadence_secs = self._reader.get_properties()["file_cadence_secs"]
            front_padding_sec = ts_origin_sec - ts_start
            if front_padding_sec == 0:
                return idx_start, idx_end 
            back_padding_sec = file_cadence_secs - front_padding_sec
            # convert back to index domain
            idx_start = self.get_index_from_ts(ts_start + front_padding_sec)
            idx_end = self.get_index_from_ts(ts_end - back_padding_sec)

        return idx_start, idx_end


    def get_fields(self):
        return self._reader.get_fields()

    def read(self, idx_start, idx_end):
        idx_first = idx_start
        idx_last = idx_end - 1
        d = self._reader.read(idx_first, idx_last)
        return next(iter(d.items()))[1]


if __name__ == "__main__":


    # DST
    metadir = Path("/tmp/eiscat/")
    shutil.rmtree(metadir, ignore_errors=True)
    metadir.mkdir(parents=True, exist_ok=True)

    # Sample rate
    SAMPLE_RATE_NUMERATOR = 1000000 
    SAMPLE_RATE_DENOMINATOR = 1
    SAMPLE_BATCH_LENGTH = 12800000  # sample batch length is 12.8 seconds worth of samples

    # writer
    writer = EiscatDRFMetadataWriter(
        metadir,
        SAMPLE_RATE_NUMERATOR,
        SAMPLE_RATE_DENOMINATOR * SAMPLE_BATCH_LENGTH,
    )

    # ts origin
    TS_ORIGIN_SEC_ALIGNED = make_ts_from_str("2024-07-10T12:00:00")
    TS_ORIGIN_SEC_MISALIGNED = make_ts_from_str("2024-07-10T12:03:03")
    TS_ORIGIN_SEC_NOW = dt.datetime.now(dt.timezone.utc).timestamp()
    ts_origin_sec = TS_ORIGIN_SEC_ALIGNED

    # write
    start_idx = writer.index_from_ts(ts_origin_sec)
    n_batches = 36*2
    pointing = get_pointing(n_batches)
    for i in range(n_batches):
        azimuth, elevation = pointing[i]
        writer.write(start_idx + i, azimuth, elevation)


    # read
    reader = EiscatDRFMetadataReader(
        metadir
    )

    bounds = reader.get_bounds()
    d = reader.read(*bounds)
    import pprint

    pprint.pprint(d)

