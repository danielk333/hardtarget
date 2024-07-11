import numpy as np
from pathlib import Path
import digital_rf as drf
from datetime import timezone, datetime as dt

####################################################################
# MISC
####################################################################

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

def create_drf_writer(dst, chnl, dtype, t_origin_sec, 
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
    dst = Path("drf")
    if not dst.is_dir():
        raise Exception(f"<dst> must be directory path, {dst}")
    chnldir = dst / chnl
    chnldir.mkdir(parents=True, exist_ok=True)

    # calculate origin index
    sample_rate = float(sample_rate_numerator) / sample_rate_denominator
    # convenience - construction time as default value for ts_origin_sec
    if ts_origin_sec is None:
        ts_origin_sec = dt.now(timezone.utc).timestamp()
    idx_origin = int(t_origin_sec) * sample_rate

    # create digital rf writer
    return drf.DigitalRFWriter(
        str(chnldir),  # destination directory
        dtype,
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

# UHF sample rate 1 MHz
UHF_SAMPLE_RATE_NUMERATOR = 1000000 
UHF_SAMPLE_RATE_DENOMINATOR = 1
# UHF sample batch is 12.8 seconds worth of samples
UHF_SAMPLE_BATCH_LENGTH = 12800000
# UHF file system resolution
UHF_SUBDIR_CADENCE_SECS = 3600 # 1 dir per hour
UHF_FILE_CADENCE_MILLISECS = 10000 # 10 seconds per file
POINTING_SUBDIR_CADENCE_SECS = 3600 # 1 dir per hour
POINTING_FILE_CADENCE_MILLISECS = 3600*1000 # 1 hour per file

class EiscatDRFWriter:

    """
    Specific DRF Writer for Eiscat data.
    Writes uhf data and pointing data (azimuth, elevation) to two
    separate channels, "uhf" and "pointing".


    NOTE. It appears that DRF reads by bounds fall on file boundaries, and that
    returned data from a file is NaN-padded if file is written for only a part of the time segment it
    covers. 
    
    E.g. if files contain an hour of data, and data was only written for half the hour or so.
    
    Also, the concept of continuous blocks includes such NaN-padded data.

    Solutions (not verified)
    - maintain independent info about segments written, and query directly for these segments.
    - remove segments of NaN from result data (remember to update indexes and lengths)

    """


    def __init__(self, dst, 
            ts_origin_sec = None,
            uhf_sample_rate_numerator=UHF_SAMPLE_RATE_NUMERATOR,
            uhf_sample_rate_denominator=UHF_SAMPLE_RATE_DENOMINATOR,
            uhf_sample_batch_length=UHF_SAMPLE_BATCH_LENGTH,
            uhf_subdir_cadence_secs=UHF_SUBDIR_CADENCE_SECS,
            uhf_file_cadence_millisecs=UHF_FILE_CADENCE_MILLISECS
        ):

        self.uhf_sample_batch_length = uhf_sample_batch_length

        # convenience - construction time as default value for ts_origin_sec
        if ts_origin_sec is None:
            ts_origin_sec = dt.now(timezone.utc).timestamp()
        
        # create dst if not exists
        self.dst = Path(dst)
        if self.dst.exists() and not self.dst.is_dir():
            raise Exception (f"dst {dst} exists, but is not a directory")
        self.dst.mkdir(parents=True, exist_ok=True)

        # uhf writer - write one batch at a time
        self.uhf_writer = create_drf_writer (
            self.dst, "uhf", np.int16, ts_origin_sec, 
            uhf_sample_rate_numerator,
            uhf_sample_rate_denominator,
            subdir_cadence_secs=uhf_subdir_cadence_secs,
            file_cadence_millisecs=uhf_file_cadence_millisecs,
            num_subchannels=1,
            is_complex=True
        )

        # pointing writer - write one value per batch
        self.pointing_writer = create_drf_writer (
            self.dst, "pointing", np.float64, ts_origin_sec,
            uhf_sample_rate_numerator,
            uhf_sample_rate_denominator * uhf_sample_batch_length,
            subdir_cadence_secs=POINTING_SUBDIR_CADENCE_SECS,
            file_cadence_millisecs=POINTING_FILE_CADENCE_MILLISECS,
            num_subchannels=2,
            is_complex=False
        )


    def write(self, uhf_batch, azimuth, elevation):
        if azimuth is None:
            azimuth = 0
        if elevation is None:
            elevation = 0
        expected_shape = (self.uhf_sample_batch_length, 2)
        ok = match_shapes(uhf_batch.shape, expected_shape)
        if not ok:
            raise Exception(f"shape mismatch {uhf_batch.shape}, expected {expected_shape}")
        self.uhf_writer.rf_write(uhf_batch)
        self.pointing_writer.rf_write(np.array([[azimuth, elevation]]))


    def close(self):
        self.uhf_writer.close()
        self.pointing_writer.close()



def to_str(ts_utc):
    datetime_utc = dt.utcfromtimestamp(ts_utc)
    return datetime_utc.strftime("%Y-%m-%dT%H:%M:%S")

class EiscatDRFReader:
    
    def __init__(self, dst, 
            ts_origin_sec = None,
            uhf_sample_batch_length=UHF_SAMPLE_BATCH_LENGTH
        ):

        # in the event that pointing channel is missing
        # [0,0] pointing info will be provided.
        # in this case, the UHF SAMPLE BATCH LENGTH is needed to provide the correct
        # sampling rate for pointing
        self.uhf_sample_batch_length = uhf_sample_batch_length

        self.dst = Path(dst)
        if not self.dst.is_dir():
            raise Exception (f"dst {dst} does not exist or is not a directory")
        self.reader = drf.DigitalRFReader(dst)
        self.chnls = self.reader.get_channels()
        if "uhf" not in self.chnls:
            raise Exception("missing channel uhf")

        # sample rates
        self._uhf_sample_rate = float(self.reader.get_properties("uhf")["samples_per_second"])

    def _get_sample_rate(self, chnl):
        if chnl == "uhf":
            return self._uhf_sample_rate
        elif chnl == "pointing":
            if "pointing" in self.chnls:
                props = self.reader.get_properties("pointing")
                return float(props["samples_per_second"])
            else:
                return self.uhf_sample_rate / self.uhf_sample_batch_length

    def _get_ts_from_index(self, idx, chnl):
        return idx / self._get_sample_rate(chnl)

    def _get_idx_from_ts(self, ts, chnl):
        return int(ts * self._get_sample_rate(chnl))

    def get_bounds(self):
        return self.reader.get_bounds("uhf")

    def get_ts_bounds(self):
        idx_first, idx_last = self.get_bounds()
        ts_start = self._get_ts_from_index(idx_first, "uhf")
        ts_end = self._get_ts_from_index(idx_last + 1, "uhf")
        return ts_start, ts_end

    def get_str_bounds(self):
        return [to_str(bound) for bound in self.get_ts_bounds()]

    def read_pointing(self, idx_first, length):
        idx_last = idx_first + length - 1

        # generate sample indexes for pointing values
        factor = self._get_sample_rate("uhf")/self._get_sample_rate("pointing")
        indexes = np.arange(start=idx_first, step=int(factor), stop=idx_last + 1, dtype=np.int64)

        # convert to timestamps
        ts_start = self._get_ts_from_index(idx_first, "uhf")
        ts_end = self._get_ts_from_index(idx_last + 1, "uhf")
        # convert to pointing indexes
        idx_first = self._get_idx_from_ts(ts_start, "pointing")
        idx_last = self._get_idx_from_ts(ts_end, "pointing") - 1
        length = idx_last + 1 - idx_first

        # calculate pointing indexes
        arr = self.reader.read_vector(idx_first, length, "pointing")

        return arr, indexes

    def read_uhf(self, idx_first, length):
        return self.reader.read_vector(idx_first, length, "uhf")
    
    def read_test(self):
        idx_first, idx_last = self.reader.get_bounds("pointing")
        length = idx_last + 1 - idx_first
        return self.reader.read_vector(idx_first, length, "pointing")    
    

