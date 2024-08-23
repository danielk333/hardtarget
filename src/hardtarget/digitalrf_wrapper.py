from pathlib import Path
import digital_rf
import numpy as np

####################################################################
# DIGITAL_RF WRAPPER
####################################################################

"""
This module provides a wrapper around reader and writer objects
from 'digital_rf' - to provide a uniform API and correct some design issues.

NOTE - not sure if digital_rf is thread-safe with respect to concurrent
writes to same file (e.g. adjacent blocks). 
Also, it appears that digital_rf is not random access with respect to 
writing (even in non-continuous mode) as it maintains and internal index 
for next available sample.
"""


####################################################################
# UTIL
####################################################################

def ts_from_index(idx, sample_rate, ts_offset_sec=0):
    """
    convert from sample idx to timestamp
    
    Params
    ------

    idx: int
        sample index (first sample is index 0)
    sample_rate: Hz
        samples per seconds
    ts_offset_sec: float 
        timestamp in seconds since Epoch (1970:01:10T00:00:00) 
        ts_offset is the timestamp corresponding to index 0,
        by default this is 0, implying that indexing starts at Epoch (1970:01:10T00:00:00)
    
    Returns
    -------
    float:
        timestamp corresponding to given sample index
    
    """    
    return (idx / float(sample_rate)) + ts_offset_sec


def index_from_ts(ts, sample_rate, ts_offset_sec=0):
    """
    convert from timestamp to sample index

    Params
    ------

    ts: float
        timestamp in seconds from Epoch (1970:01:10T00:00:00)
    sample_rate: Hz
        samples per seconds
    ts_offset_sec: float 
        timestamp in seconds since Epoch (1970:01:10T00:00:00)
        ts_offset is the timestamp corresponding to index 0,
        by default this is 0, implying that indexing starts at Epoch (1970:01:10T00:00:00)
    
    Returns
    -------
    float:
        sample index (first sample is index 0)
    """
    return (ts-ts_offset_sec) * sample_rate


####################################################################
# BASE INDEXED TIME SEQUENCE
####################################################################

class BaseIndexedTimeSequence:

    def __init__(self, sample_rate, ts_align_sec=0, ts_offset_sec=0):

        # sample rate for time sequence
        self.sample_rate = sample_rate

        # ts_offset_sec
        # ts_offset_sec is a timestamp in seconds since Epoch
        # ts_offset_sec defines a time offset for the index space (i.e. index 0)
        # if ts_offset_sec is not provided, index space is assumed to start at Epoch
        self.__ts_offset_sec = ts_offset_sec

        # ts_align_sec
        self.__ts_align_sec = ts_align_sec

        # ts_align_sec is a timestamp in seconds since Epoch
        # ts_align_sec is a timestamp associated with some index.
        # (typically it could be the timestamp of the first sample)
        # ts_align_sec defines the precise alignment between timestamps and index-domain.        
        # ts_align_sec may not precisely match logical sample boundaries (which are
        # defined by sample rate and ts_offset_sec)
        # Essentially (ts_align_sec * sample_rate) may not produce a perfect integer, 
        # and must therefore be floored to produce an integer index. 
        # This discrepancy defines a static skew (in time domain), 
        # between logical integer indexes and timestamps.
        # if ts_align_sec is not provided, the assumption is that the skew is 0 

        self.__ts_skew = 0
        __idx_align = index_from_ts(ts_align_sec, self.sample_rate) 
        self.__ts_skew = __idx_align - np.floor(__idx_align)

    def ts_from_index(self, idx):
        ts = ts_from_index(idx, self.sample_rate, ts_offset_sec=self.__ts_offset_sec)
        return ts + self.__ts_skew

    def index_from_ts(self, ts):
        ts = ts - self.__ts_skew
        return index_from_ts(ts, self.sample_rate, ts_offset_sec=self.__ts_offset_sec)


####################################################################
# DIGITAL RF WRITER
####################################################################

class DigitalRFWriter(BaseIndexedTimeSequence):
    
    """
    Convenience wrapper around digital_rf.DigitalRFWriter 
    """

    def __init__(self, dst, chnl,
                 sample_rate_numerator,
                 sample_rate_denominator,
                 dtype_str,
                 start_global_index,
                 subdir_cadence_secs=3600,  # 1 dir per hour
                 file_cadence_secs=3600,  # 1 hour per file
                 compression_level=0,
                 is_complex=False,
                 checksum=False,
                 num_subchannels=1,
                 is_continuous=True,
                 marching_periods=False,
                 uuid_str=None,
                 ts_align_sec=None
                ):

        # check dst
        dst = Path(dst)
        if not dst.is_dir():
            raise Exception(f"<dst> must be directory path, {dst}")

        # channel directory
        chnldir = dst / chnl
        chnldir.mkdir(parents=True, exist_ok=True)

        # sample rate
        sample_rate = sample_rate_numerator / float(sample_rate_denominator)
        
        super().__init__(sample_rate, ts_align_sec=ts_align_sec)

        # meta data writer
        self._writer = digital_rf.DigitalRFWriter(
            str(chnldir),
            dtype_str,
            subdir_cadence_secs,
            file_cadence_secs * 1000,  # file_cadence_milliseconds
            start_global_index,  # start global index
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
    
    def write(self, batch):
        self._writer.rf_write(batch)


####################################################################
# DIGITAL RF READER
####################################################################


class DigitalRFReader(BaseIndexedTimeSequence):

    """
    Convenience wrapper around digital_rf.DigitalRFReader 
    """

    def __init__(self, dst, chnl, ts_align_sec=0):

        """
        Parameters
        ----------

        dst: str
            Path to root directory
        chnl: str
            Name of channel (i.e. subdirectory)
        ts_align_sec (optional): timestamp associated with some index

        In the writer, <start_global_index> is used to define a precise starting moment
        for the first sample in the time sequence. On the reader side, though, the time resolution
        is by default limited to the length of a sample, which implies that the
        the exact time-alignment of a sample can be off by e (0 <= e < sample_length).
        This would be increasingly problematic for longer samples. This skew also matters when
        trying to align data series with differet sample lengths.

        To resolve this, provide <ts_align_sec>, a timestamp which anchors the start of an index to
        time domain, making it possible to calculate a skew (could for instance be the timstamp
        associated with <start_global_index>).
        """

        # setup reader
        dst = Path(dst)
        if not dst.is_dir():
            raise Exception(f"<dst> must be directory path, {dst}")
        self._reader = digital_rf.DigitalRFReader(str(dst))

        # check chnl
        if chnl not in self._reader.get_channels():
            raise Exception(f"chnl {chnl} missing in {dir}") 
        self.chnl = chnl


        sample_rate = float(self._reader.get_properties(chnl)["samples_per_second"])

        # ts_align_sec defines the start of the first sample of the time-series
        # used to correct issue with file alignment (see get_bounds())
        super().__init__(sample_rate, ts_align_sec=ts_align_sec)

    def get_bounds(self):
        """
        return bounds for read function

        NOTE: digital rf read() includes idx_last - which breaks python range convention
        this is compensated for, so that this function returns according to convention
        return (start, end) where end is first sample not in bouds

        NOTE: digital rf read() may return padded values at start and and, originating
        from internal aspects of storage organization in files
        """

        idx_first, idx_last = self._reader.get_bounds(self.chnl)
        idx_start = idx_first
        idx_end = idx_last + 1        
        return idx_start, idx_end


    def read(self, idx_start, idx_end):
        """
        indexes in data sampling domain
        
        TODO: reads outside bounds will either return NaN data padding, or not
        return anything elements. This is related to the issue of bounds and
        internal storage organization.

        NOTE: digital rf read() includes idx_last - which breaks convention
        this function fixes this issue, by implementing "until idx_last" semantics
        """
        idx_first = idx_start
        idx_last = idx_end - 1
        return self._reader.read(idx_first, idx_last, self.chnl).items()



####################################################################
# DIGITAL METADATA WRITER
####################################################################

class DigitalMetadataWriter(BaseIndexedTimeSequence):
    
    """
    Convenience wrapper around digital_rf.DigitalMetadataWriter 
    """

    def __init__(self, dst, chnl,
                 sample_rate_numerator,
                 sample_rate_denominator,
                 subdir_cadence_secs=3600,  # 1 dir per hour
                 file_candence_secs=3600,  # 1 hour per file
                 prefix="meta",
                 ts_align_sec=0
                ):

        # check dst
        dst = Path(dst)
        if not dst.is_dir():
            raise Exception(f"<dst> must be directory path, {dst}")

        # metadir
        metadir = dst / chnl
        metadir.mkdir(parents=True, exist_ok=True)

        # sample rate
        sample_rate = sample_rate_numerator / float(sample_rate_denominator)

        super().__init__(sample_rate, ts_align_sec=ts_align_sec)

        # meta data writer
        self._writer = digital_rf.DigitalMetadataWriter(
            str(metadir),
            subdir_cadence_secs,
            file_candence_secs,
            sample_rate_numerator,
            sample_rate_denominator,
            prefix
        )

    def write(self, idx, d):
        self._writer.write(idx, d)

    def close(self):
        pass


####################################################################
# DIGITAL METADATA READER
####################################################################

class DigitalMetadataReader(BaseIndexedTimeSequence):

    """
    Convenience wrapper around digital_rf.DigitalMetadataReader 
    """

    def __init__(self, dst, chnl, ts_align_sec=0):

        # check dst and chnl
        dst = Path(dst)
        if not dst.is_dir():
            raise Exception(f"<dst> must be directory path, {dst}")

        metadir = dst / chnl
        if not metadir.is_dir():
            raise Exception(f"missing chnl {chnl} in {dst}")

        # setup reader
        self._reader = digital_rf.DigitalMetadataReader(str(metadir))
        sample_rate = self._reader.get_samples_per_second()
        super().__init__(sample_rate, ts_align_sec=ts_align_sec)
        self.chnl = chnl

    def get_bounds(self):
        """
        return bounds for read function
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

        TODO : semantics for reading non-existing data
        should be consistent with RFdata, 
        either to return nothing (empty collection)
        or to generate padding. Think this is an 
        argument for returning nothing both for reads of both RFData and Metadata
        """
        idx_first = idx_start
        idx_last = idx_end - 1
        return self._reader.read(idx_first, idx_last).items()
