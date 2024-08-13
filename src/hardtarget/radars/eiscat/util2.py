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

    def __init__(self, dst, chnl,
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

        # check dst
        dst = Path(dst)
        if not dst.is_dir():
            raise Exception(f"<dst> must be directory path, {dst}")

        # channel directory
        chnldir = dst / chnl
        chnldir.mkdir(parents=True, exist_ok=True)


        # sample rate
        self._sample_rate = sample_rate_numerator / float(sample_rate_denominator)
        # use construction time as default value for ts_origin_sec
        if ts_origin_sec is None:
            ts_origin_sec = dt.now(timezone.utc).timestamp()

        # meta data writer
        self._writer = digital_rf.DigitalRFWriter(
            str(chnldir),
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




_DRF_READERS = {} # path -> reader
def get_drf_reader(path):
    global _DRF_READERS
    path = str(path)
    if path not in _DRF_READERS:
        _DRF_READERS[path] = digital_rf.DigitalRFReader(path)
    return _DRF_READERS[path]


class EiscatDRFReader:

    """
    Convenience wrapper around digital_rf.DigitalRFReader 
    """

    def __init__(self, dst, chnl):

        # check dst
        dst = Path(dst)
        if not dst.is_dir():
            raise Exception(f"<dst> must be directory path, {dst}")

        # setup reader
        self._reader = get_drf_reader(dst)
        if chnl not in self._reader.get_channels():
            raise Exception(f"chnl {chnl} missing in {dir}") 
        self.chnl = chnl
        self._sample_rate = float(self._reader.get_properties(self.chnl)["samples_per_second"])

    def ts_from_index(self, idx):
        return idx / self._sample_rate

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

        idx_first, idx_last = self._reader.get_bounds(self.chnl)
        idx_start = idx_first
        idx_end = idx_last + 1

        if ts_origin_sec is not None:
            # convert to time domain
            ts_start = self.ts_from_index(idx_start)
            ts_end = self.ts_from_index(idx_end)
            # calculate padding in time domain
            file_cadence_secs = self._reader.get_properties(self.chnl)["file_cadence_millisecs"] / 1000.0
            front_padding_sec = ts_origin_sec - ts_start
            if front_padding_sec == 0:
                return idx_start, idx_end 
            back_padding_sec = file_cadence_secs - front_padding_sec
            # convert back to index domain
            idx_start = self.index_from_ts(ts_start + front_padding_sec)
            idx_end = self.index_from_ts(ts_end - back_padding_sec)
        
        return idx_start, idx_end


    def read(self, idx_start, idx_end):
        """
        indexes in data sampling domain
        
        NOTE: digital rf read() includes idx_last - which breaks convention
        this function fixes this issue, by implementing "until idx_last" semantics
        """
        idx_first = idx_start
        idx_last = idx_end - 1
        return self._reader.read(idx_first, idx_last, self.chnl).items()


class EiscatDRFMetadataWriter:
    

    """
    Convenience wrapper around digital_rf.DigitalMetadataWriter 
    """

    def __init__(self, dst, chnl,
                 sample_rate_numerator,
                 sample_rate_denominator,
                 subdir_cadence_secs=3600,  # 1 dir per hour
                 file_candence_secs=3600,  # 1 hour per file
                 prefix="meta",
                ):


        # check dst
        dst = Path(dst)
        if not dst.is_dir():
            raise Exception(f"<dst> must be directory path, {dst}")

        # metadir
        metadir = dst / chnl
        metadir.mkdir(parents=True, exist_ok=True)

        # sample rate
        self._sample_rate = sample_rate_numerator / float(sample_rate_denominator)

        # meta data writer
        self._writer = digital_rf.DigitalMetadataWriter(
            str(metadir),
            subdir_cadence_secs,
            file_candence_secs,
            sample_rate_numerator,
            sample_rate_denominator,
            prefix
        )

    def ts_from_index(self, idx):
        return idx / self._sample_rate

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




class EiscatDRFMetadataReader:

    """
    Convenience wrapper around digital_rf.DigitalMetadataReader 
    """

    def __init__(self, dst, chnl):


        # check dst and chnl
        dst = Path(dst)
        if not dst.is_dir():
            raise Exception(f"<dst> must be directory path, {dst}")

        metadir = dst / chnl
        if not metadir.is_dir():
            raise Exception(f"missing chnl {chnl} in {dst}")

        # setup reader
        self._reader = digital_rf.DigitalMetadataReader(str(metadir))
        self._sample_rate = self._reader.get_samples_per_second()

        self.chnl = chnl


    def ts_from_index(self, idx):
        return idx / self._sample_rate

    def index_from_ts(self, ts):
        return int(ts * self._sample_rate)

    def get_bounds(self, ts_origin_sec=None):
        """
        return bounds for read function
        ts_origin_sec obsolete, but included only to match signature of drf reader
        """
        idx_first, idx_last = self._reader.get_bounds()
        idx_start = idx_first
        idx_end = idx_last + 1
        return idx_start, idx_end


    def get_fields(self):
        return self._reader.get_fields()

    def read(self, idx_start, idx_end):
        """
        Returns list of tuples (idx, metadatat)
        """
        idx_first = idx_start
        idx_last = idx_end - 1
        return self._reader.read(idx_first, idx_last).items()
