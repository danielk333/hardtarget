from pathlib import Path
import shutil
import digital_rf
import numpy as np
import datetime as dt
import time


class EiscatDRFWriter:
    
    """
    Convenience wrapper around digital_rf.DigitalRFWriter 
    """

    def __init__(self, 
                 datadir,
                 sample_rate_numerator,
                 sample_rate_denominator,
                 dtype_str,
                 ts_origin_sec=None,
                 subdir_cadence_secs=3600,  # 1 dir per hour
                 file_cadence_secs=3600,  # 1 hour per file
                 compression_level=0,
                 is_complex=False,
                 checksum=False,
                 num_subchannels=1,
                 is_continuous=True,
                 marching_periods=False,
                 uuid_str=None
                ):

        # check datadir
        datadir = Path(datadir)
        if not datadir.is_dir():
            raise Exception(f"<datadir> must be directory path, {datadir}")

        # sample rate
        self._sample_rate = sample_rate_numerator / float(sample_rate_denominator)
        # use construction time as default value for ts_origin_sec
        if ts_origin_sec is None:
            ts_origin_sec = dt.now(timezone.utc).timestamp()

        # meta data writer
        self._writer = digital_rf.DigitalRFWriter(
            str(datadir),
            dtype_str,
            subdir_cadence_secs,
            file_cadence_secs * 1000,  # file_cadence_milliseconds
            self.index_from_ts(ts_origin_sec),  # start global index
            sample_rate_numerator,
            sample_rate_denominator,
            uuid_str=uuid_str,
            compression_level=compression_level,
            checksum=checksum,
            is_complex=is_complex,
            num_subchannels=num_subchannels,
            is_continuous=is_continuous,
            marching_periods=marching_periods
        )

    def close(self):
        self._writer.close()

    def ts_from_index(self, idx):
        return self.idx / self._sample_rate

    def index_from_ts(self, ts):
        return int(ts * self._sample_rate)

    def write(self, batch):
        self._writer.rf_write(batch)



class EiscatDRFReader:

    """
    Convenience wrapper around digital_rf.DigitalRFReader 
    """

    def __init__(self, datadir):
        self._reader = digital_rf.DigitalRFReader(str(datadir))
        self.chnls = self._reader.get_channels()

    def _get_sample_rate(self, chnl):
        return float(self._reader.get_properties(chnl)["samples_per_second"])

    def ts_from_index(self, idx, chnl):
        return self.idx / self._get_sample_rate(chnl)

    def index_from_ts(self, ts, chnl):
        return int(ts * self._get_sample_rate(chnl))

    def get_bounds(self, chnl, ts_origin_sec=None):
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

        idx_first, idx_last = self._reader.get_bounds(chnl)
        idx_start = idx_first
        idx_end = idx_last + 1

        if ts_origin_sec is not None:
            # convert to time domain
            ts_start = self.get_ts_from_index(idx_start, chnl)
            ts_end = self.get_ts_from_index(idx_end, chnl)
            # calculate padding in time domain
            file_cadence_secs = self._reader.get_properties(chnl)["file_cadence_secs"]
            front_padding_sec = ts_origin_sec - ts_start
            if front_padding_sec == 0:
                return idx_start, idx_end 
            back_padding_sec = file_cadence_secs - front_padding_sec
            # convert back to index domain
            idx_start = self.get_index_from_ts(ts_start + front_padding_sec, chnl)
            idx_end = self.get_index_from_ts(ts_end - back_padding_sec, chnl)

        return idx_start, idx_end


    def read(self, idx_start, idx_end, chnl):
        """
        indexes in data sampling domain
        
        NOTE: digital rf read() includes idx_last - which breaks convention
        this function fixes this issue, by implementing "until idx_last" semantics
        """
        d =  self._reader.read(idx_start, idx_end-1, chnl)
        return list(d.items())


class EiscatDRFMetadataWriter:
    

    """
    Convenience wrapper around digital_rf.DigitalMetadataWriter 
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

    def close(self):
        pass


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

    """
    Convenience wrapper around digital_rf.DigitalMetadataReader 
    """

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
        """
        Returns list of tuples (idx, metadatat)
        """
        idx_first = idx_start
        idx_last = idx_end - 1
        d = self._reader.read(idx_first, idx_last)
        return list(d.items())
