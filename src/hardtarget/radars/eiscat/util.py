import numpy as np
from pathlib import Path
import digital_rf as drf
from datetime import timezone, datetime as dt

####################################################################
# MISC
####################################################################

def datetime_to_str(ts_utc):
    datetime_utc = dt.utcfromtimestamp(ts_utc)
    return datetime_utc.strftime("%Y-%m-%dT%H:%M:%S")

def match_shapes(shp1, shp2):
    """
    Check if shp1 matches shp2.
    
    Parameters:
    shp1 (list, tuple, or np.ndarray): Shape 1
    shp2 (list, tuple, or np.ndarray): Shape 2
    
    Returns:
    bool: True if the shapes match, False otherwise
    """
    # Convert shape to tuple if it is a list or a np.ndarray
    shp1 = tuple(shp1)
    shp2 = tuple(shp2)
    # compare as tuples
    return shp1 == shp2


####################################################################
# PARSE MATLAB FILE
####################################################################

PARBL_ELEVATION = 8
PARBL_AZIMUTH = 9
PARBL_END_TIME = 10
PARBL_SEQUENCE = 11
PARBL_START_TIME = 42 # upar[1]


def parse_matlab(mat, sample_rate, file_secs, zero_padding=True):

    """
    parse (and check) matlab file
    sample_rate - expected sample rate
    file_secs - expected file length in seconds
    if zero_padding is True: fix missing data by zero padding, else raise assert error 
    returns dictionary with meta information
    """
    d = {}

    parbl = mat["d_parbl"][0]

    # global start of recording - repeated for all files
    d["t_origin"] = t_origin = parbl[PARBL_START_TIME]

    # end time of last sample in this file (where the last sample ends)
    d["t_end"] = t_end = parbl[PARBL_END_TIME]

    # file duration in seconds from number of actual samples
    d["t_delta"] = t_delta = np.float64(file_secs)

    # number of data samples in file
    samples = mat["d_raw"]
    n_samples = len(samples)

    # check that number of samples match expected sample_rate and time
    expected_samples = round(t_delta * sample_rate)

    if zero_padding and n_samples < expected_samples:
        padding_len = expected_samples - n_samples
        padding = np.zeros((padding_len, 1), np.dtype("complex128"))
        samples = np.concatenate((samples, padding), axis=0)
        n_samples = len(samples)

    assert n_samples == expected_samples , f"mismatch samples: expected {expected_samples}, actual {n_samples}"

    d["samples"] = samples
    d["n_samples"] = n_samples

    # sequence number
    # files have a sequence number relative to t0, with 1 as the first logical sequence number.
    seq = int(parbl[PARBL_SEQUENCE])

    # check that sequence number match timestamps 
    expected_seq = round((t_end - t_origin) / t_delta)
    assert seq == expected_seq, f"mismatch sequence number {seq} with timestamps {expected_seq}" 

    # file index
    # subtract one to get a zero-indexed file index from sequence numbers
    d["idx_file"] = idx_file = seq - 1
        
    # global index of first sample in recording (t_origin)
    d["idx_origin"] = idx_origin = round(t_origin * sample_rate)

    # global index of first sample in file
    d["idx_first"] = idx_first = idx_origin + idx_file * n_samples

    # timestamp of first sample in file
    d["t_start"] = t_end - t_delta

    # global index of last sample in file
    d["idx_last"] = idx_first + n_samples - 1    

    # instrument pointing angles
    d["elevation"] = elevation = parbl[PARBL_ELEVATION]
    d["azimuth"] = azimuth = parbl[PARBL_AZIMUTH]

    return d



####################################################################
# DRF WRITER
####################################################################

def create_drf_writer(dst, chnl, dtype_str, ts_origin_sec, 
                      sample_rate_numerator,
                      sample_rate_denominator,
                      subdir_cadence_secs=3600,
                      file_cadence_millisecs=1000,
                      compression_level=0,
                      is_complex=False,
                      checksum=False,
                      num_subchannels=1,
                      is_continuous=True,
                      marching_periods=False):

    """
    This is a tiny convenience wrapper around the drf writer object.
    - adds chnl subdir creation
    - takes a timestamp (sec after epoch) as global start, instead of sample index
    - provides some default values for properties   
    """

    # destination
    dst = Path(dst)
    if not dst.is_dir():
        raise Exception(f"<dst> must be directory path, {dst}")
    chnldir = dst / chnl
    chnldir.mkdir(parents=True, exist_ok=True)

    # calculate origin index
    sample_rate = float(sample_rate_numerator) / sample_rate_denominator
    # convenience - construction time as default value for ts_origin_sec
    if ts_origin_sec is None:
        ts_origin_sec = dt.now(timezone.utc).timestamp()
    idx_origin = int(ts_origin_sec) * sample_rate

    # create digital rf writer
    return drf.DigitalRFWriter(
        str(chnldir),  # destination directory
        dtype_str,
        subdir_cadence_secs,
        file_cadence_millisecs,
        idx_origin,  # start global index
        sample_rate_numerator,
        sample_rate_denominator,
        uuid_str=chnl,
        compression_level=compression_level,
        checksum=checksum,
        is_complex=is_complex,
        num_subchannels=num_subchannels,
        is_continuous=is_continuous,
        marching_periods=marching_periods
    )


####################################################################
# EISCAT DRF WRITER
####################################################################

# sample rate 1 MHz
SAMPLE_RATE_NUMERATOR = 1000000 
SAMPLE_RATE_DENOMINATOR = 1
# sample batch lenght is 12.8 seconds worth of samples
SAMPLE_BATCH_LENGTH = 12800000

# file system resolution
SAMPLE_SUBDIR_CADENCE_SECS = 3600 # 1 dir per hour
SAMPLE_FILE_CADENCE_MILLISECS = 10000 # 10 seconds per file
POINTING_SUBDIR_CADENCE_SECS = 3600 # 1 dir per hour
POINTING_FILE_CADENCE_MILLISECS = 3600*1000 # 1 hour per file

class EiscatDRFWriter:

    """
    Specific DRF Writer for Eiscat data.
    Writes radar samples and pointing values (azimuth, elevation) to two
    separate channels, "data" and "pointing".
    One pointing value is written along with a batch of radar samples

    dst
        - path to root folder (must exist)

    ts_origin_sec
        - timstamp (utc in seconds) for the start of the recording
        - can be given to reader.get_bounds() to find exact bounds for data
        - default to current time (now) if not supplied

    sample_rate_numerator (int)
    sample_rate_denominator (int)
        - sample rate

    sample_batch_length
        - size of sample batches
        - implicitly defines sample rate for pointing data

    sample_subdir_cadence_secs
        - number of seconds per folder
    
    sample_file_cadence_millisecs
        - number of milliseconds per file
    
    """


    def __init__(self, dst, 
            ts_origin_sec = None,
            sample_rate_numerator=SAMPLE_RATE_NUMERATOR,
            sample_rate_denominator=SAMPLE_RATE_DENOMINATOR,
            sample_batch_length=SAMPLE_BATCH_LENGTH,
            sample_subdir_cadence_secs=SAMPLE_SUBDIR_CADENCE_SECS,
            sample_file_cadence_millisecs=SAMPLE_FILE_CADENCE_MILLISECS
        ):

        self.sample_batch_length = sample_batch_length

        # convenience - construction time as default value for ts_origin_sec
        if ts_origin_sec is None:
            ts_origin_sec = dt.now(timezone.utc).timestamp()
        
        # create dst if not exists
        self.dst = Path(dst)
        if self.dst.exists():
            if self.dst.is_dir():
                raise Exception (f"dst {dst} already exists, please remove")
            else:
                raise Exception (f"dst {dst} exists, but is not a directory")

        self.dst.mkdir(parents=True, exist_ok=True)

        # sample writer - writes one batch at a time
        self.sample_writer = create_drf_writer (
            self.dst, "data", np.int16, ts_origin_sec, 
            sample_rate_numerator,
            sample_rate_denominator,
            subdir_cadence_secs=sample_subdir_cadence_secs,
            file_cadence_millisecs=sample_file_cadence_millisecs,
            num_subchannels=1,
            is_complex=True
        )

        # pointing writer - write one value per batch
        self.pointing_writer = create_drf_writer (
            self.dst, "pointing", np.float64, ts_origin_sec,
            sample_rate_numerator,
            sample_rate_denominator * sample_batch_length,
            subdir_cadence_secs=POINTING_SUBDIR_CADENCE_SECS,
            file_cadence_millisecs=POINTING_FILE_CADENCE_MILLISECS,
            num_subchannels=2,
            is_complex=False
        )

    def write(self, sample_batch, azimuth, elevation):
        if azimuth is None:
            azimuth = 0.0
        if elevation is None:
            elevation = 0.0
        expected_shape = (self.sample_batch_length, 2)
        ok = match_shapes(sample_batch.shape, expected_shape)
        if not ok:
            raise Exception(f"shape mismatch {sample_batch.shape}, expected {expected_shape}")
        self.sample_writer.rf_write(sample_batch)
        pointing = np.array([[azimuth, elevation]], dtype=np.float64)
        self.pointing_writer.rf_write(pointing)

    def close(self):
        self.sample_writer.close()
        self.pointing_writer.close()





class EiscatDRFReader:

    """
    Specific DRF Reader for Eiscat data.
    Reads radar samples and pointing values (azimuth, elevation) from two
    separate channels, "sample" and "pointing".
    All methods working with indexes in input or output (eg. get_bounds, read)
    operate on sample indexes (not pointing indexes)

    dst
        - path to root folder (must exist)

    ts_origin_sec
        - timstamp (utc in seconds) for the start of the recording
        - not supplied by framework, so must be supplied by programmer

    sample_batch_length
        - size of sample batches
        - implicitly defines sample rate for pointing data
        - not supplied by framework, so must be supplied by programmer

    """

    def __init__(self, dst, 
            ts_origin_sec = None,
            sample_batch_length=SAMPLE_BATCH_LENGTH
        ):

        self.ts_origin_sec = ts_origin_sec
        self.sample_batch_length = sample_batch_length

        # path
        self.dst = Path(dst)
        if not self.dst.is_dir():
            raise Exception (f"dst {dst} does not exist or is not a directory")
        self.reader = drf.DigitalRFReader(dst)
        self.chnls = self.reader.get_channels()
        if "data" not in self.chnls:
            raise Exception("missing channel data")

        # sample rates
        self._sample_rate = float(self.reader.get_properties("data")["samples_per_second"])
        # in the case that pointing channel is missing
        # [0,0] pointing info will be provided.
        # in this case, the SAMPLE BATCH LENGTH is needed to provide the correct
        # sampling rate for pointing
        if "pointing" in self.chnls:
            props = self.reader.get_properties("pointing")
            self._pointing_sample_rate = float(props["samples_per_second"])
        else:
            self._pointing_sample_rate = self.sample_rate / self.sample_batch_length


    def get_sample_rate(self, chnl="data"):
        if chnl == "data":
            return self._sample_rate
        elif chnl == "pointing":
            return self._pointing_sample_rate

    def get_ts_from_index(self, idx, chnl="data"):
        return idx / self.get_sample_rate(chnl)

    def get_index_from_ts(self, ts, chnl="data"):
        return int(ts * self.get_sample_rate(chnl))

    def get_bounds(self):
        idx_first, idx_last = self.reader.get_bounds("data")
        if self.ts_origin_sec is not None:
            # convert to time domain
            ts_start = self.get_ts_from_index(idx_first, chnl="data")
            ts_end = self.get_ts_from_index(idx_last + 1 , chnl="data")
            # calculate padding in time domain
            file_cadence_millisecs = self.reader.get_properties("data")["file_cadence_millisecs"]
            front_padding_sec = self.ts_origin_sec - ts_start 
            back_padding_sec = file_cadence_millisecs/1000 - front_padding_sec
            # convert back to index domain
            idx_first = self.get_index_from_ts(ts_start + front_padding_sec, chnl="data")
            idx_last = self.get_index_from_ts(ts_end - back_padding_sec, chnl="data") - 1
        return idx_first, idx_last

    def get_ts_bounds(self):
        idx_first, idx_last = self.get_bounds()
        ts_start = self.get_ts_from_index(idx_first, "data")
        ts_end = self.get_ts_from_index(idx_last + 1, "data")
        return ts_start, ts_end

    def get_str_bounds(self):
        return [datetime_to_str(bound) for bound in self.get_ts_bounds()]

    def read_pointing(self, idx_first, idx_last):
        """
        indexes are in data sampling domain - must be converted to indexes for pointing data
        """

        # generate sample indexes for pointing values
        step = self.sample_batch_length
        indexes = np.arange(start=idx_first, step=step, stop=idx_last + 1, dtype=np.int64)

        # convert to time domain
        ts_start = self.get_ts_from_index(idx_first, "data")
        ts_end = self.get_ts_from_index(idx_last + 1, "data")

        # convert to pointing indexes
        idx_first = self.get_index_from_ts(ts_start, "pointing")
        idx_last = self.get_index_from_ts(ts_end, "pointing") - 1

        # calculate pointing indexes
        d = self.reader.read(idx_first, idx_last, "pointing")
        first = next(iter(d.items()))

        # TODO assert that there is only one item in dict
        return first[1], indexes

    def read_data(self, idx_first, idx_last):
        d =  self.reader.read(idx_first, idx_last, "data")
        first = next(iter(d.items()))
        return first[1]
    

