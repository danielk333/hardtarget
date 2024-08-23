from pathlib import Path
import glob
import re
import bz2
import numpy as np
import scipy.io as sio
import itertools as it
from hardtarget.radars.eiscat import load_expconfig
import hardtarget.digitalrf_wrapper as drf_wrapper
import configparser
from tqdm import tqdm

####################################################################
# GLOBALS
####################################################################

PARBL_ELEVATION = 8
PARBL_AZIMUTH = 9
PARBL_END_TIME = 10
PARBL_SEQUENCE = 11
PARBL_START_TIME = 42 # upar[1]
PARBL_RADAR_FREQUENCY = 54 # upar[13]

####################################################################
# UTIL
####################################################################

def to_i2x16(zz):
    """
    convert from (N,) complex128 to (N,2) int16 
    """
    zz2x16 = np.empty((len(zz), 2), dtype=np.int16)
    zz2x16[:, 0] = zz.real.astype(np.int16)
    zz2x16[:, 1] = zz.imag.astype(np.int16)
    return zz2x16


def all_files(top):
    """generate all files matching 'dddddddd_dd/dddddddd.mat' or '.mat.bz2'
    in sorted order
    """
    d = "[0-9]"

    def filter_func(pth):
        return pth.endswith(".mat") or pth.endswith("mat.bz2")

    dirs = sorted(glob.glob(f"{top}/{8*d}_{2*d}"))
    return it.chain(*(sorted(filter(filter_func, glob.glob(f"{dir}/{8*d}.mat*"))) for dir in dirs))


def loadmat(path):
    """
    Loat matlab file.
    Transparently unzips files on the fly (not in the filesystem)
    """
    if path.endswith(".mat.bz2"):
        path = bz2.open(path, "rb")
    return sio.loadmat(path)


def expinfo_split(xpinf):
    """Move from hard coded constants to loading config based on exp name/version

    'kst0 leo_bpark_2.1u_NO' -> ('kst0', 'leo_bpark', '2.1u', 'NO')
    """
    try:
        # host, name, versi, owner = \
        match = re.match(r"(\w+) +(\w+)_(\d+(?:\.\d+)?[a-z]*)_(\w+)", xpinf)
        return match.groups()
    except Exception as e:
        raise ValueError(f"d_ExpInfo: {xpinf} not understood: {e}")


def parse_matlab(mat, sample_rate, file_secs):

    """
    parse (and check) matlab file
    sample_rate - expected sample rate
    file_secs - expected file length in seconds
    returns dictionary with meta information

    timestamps are seconds since epoch
    """
    d = {
        "sample_rate": sample_rate,
        "file_secs": file_secs
    }

    parbl = mat["d_parbl"][0]

    # expected duration of file in seconds 
    d["duration"] = duration = float(file_secs)

    # global start of recording - repeated for all files in product
    d["ts_origin"] = ts_origin = float(parbl[PARBL_START_TIME])
    # end time of last sample in this file
    d["ts_end"] = ts_end = float(parbl[PARBL_END_TIME])
    # timestamp of first sample in file
    d["ts_start"] = ts_end - duration

    # expected number of samples in file
    d["n_samples"] = samples = int(duration * sample_rate)

    # sequence number
    # files have a hard coded sequence number 
    # unclear if/how this number relates to timestamps
    d["seq"] = seq = int(parbl[PARBL_SEQUENCE])

    # file index - relative to ts_origin - zero indexed
    d["file_idx"] = idx_file = int(np.floor((ts_end - ts_origin) / duration)) - 1

    # index of first sample in recording (t_origin)
    d["sample_idx_origin"] = idx_origin = int(np.floor(ts_origin * sample_rate))
    # index of first sample in file
    d["sample_idx_start"] = idx_start = idx_origin + idx_file * samples
    # index of next sample after file
    d["sample_idx_end"] = idx_start + samples    

    # instrument pointing angles
    d["elevation"] = elevation = float(parbl[PARBL_ELEVATION])
    d["azimuth"] = azimuth = float(parbl[PARBL_AZIMUTH]) % 360
   

    return d



def determine_n0(mat, cfv):
    """
    cfv - config for this version of the experiment

    find (global) index of first raw data sample in file, assuming
    continuous sampling.

    The epoch (in seconds) is the radar controller start time,
    and is found in the first user parameter (d_parbl[42] or upar[1]).

    The time (in seconds) of the last sample in the file is in d_parbl[10].
    This is used to find the (integral) number of the current file (counting
    from 0).

    The sample number of the first sample in the file is then the sample number
    of the epoch plus the number of this file times the number of samples in each file.
    """

    samp_rate = int(cfv.get("sample_rate"))  # assuming integral # samples per second
    t0 = float(mat["d_parbl"][0][PARBL_START_TIME])
    tx = float(mat["d_parbl"][0][PARBL_END_TIME])
    n_epoch = round(t0 * samp_rate)
    N_samp = len(mat["d_raw"])  # sanity check this value?
    N_sec = N_samp / samp_rate  # number of seconds in each file
    i_file = round((tx - t0) / N_sec) - 1  # -1 because parbl[10] records _end_ time of file
    return n_epoch + N_samp * i_file



####################################################################
# CONVERT
####################################################################


def convert(src, dst, name=None, compression=0, progress=False, logger=None):

    """
    Converts Eiscat raw data to Hardtarget DRF.

    Example
    -------

    .. code-block:: python

        path = convert('/data/leo_bpark_2.1u_NO@uhf', '/data')


    Parameters
    ----------

    src : str
        path to source directory (Eiscat raw data)
    dst : str
        path to destination directory
    name : str, None
        name of output directory (Hardtarget DRF)
    compression : int [0-9], 0
        compression level for h5 files in output
    logger : logging.Logger, None
        Optional logger object
    progress : bool, False
        Print download progress bar to stdout


    The output (Hardtarget DRF) folder will be placed within the 'dst'
    directory. By default, the name of the Hardtarget DRF folder is constructed
    from the name of the 'src' directory. The name of the Hardtarget DRF folder
    may be specified using the 'name' option. If 'name' is None, name is derived
    from 'src' (leo_bpark_2.1u_NO@uhf -> leo_bpark_2.1u_NO@uhf_drf).



    Returns
    -------

    str
        The absolute path to the Hardtarget DRF folder or None.


    Raises
    ------

    FileNotFoundError
        'src' does not exist 'dst' does not exist
    FileExistsError
        'dst/name' already exists

    """

    #######################################################################
    # CHECK SRC, DST, NAME
    #######################################################################

    src = Path(src)
    if not src.is_dir():
        raise FileNotFoundError(str(src))

    dst = Path(dst)
    if not dst.is_dir():
        raise FileNotFoundError(str(dst))

    if name is None:
        name = f"{src.name}_drf"
    hdrf = dst / name
    if hdrf.exists():
        raise FileExistsError(str(hdrf))
    hdrf.mkdir(parents=True, exist_ok=True)

    #######################################################################
    # SRC FILES
    #######################################################################

    # all files from Eiscat raw data product, in sorted order
    files = list(all_files(src))
   
    #######################################################################
    # META DATA
    #######################################################################

    # load experiment info from first matlab file
    first = files[0]
    mat = loadmat(first)
    host, expname, expvers, owner = expinfo_split(str(mat["d_ExpInfo"][0]))
    cfg = load_expconfig(expname)
    cfv = cfg[expvers]
    sample_rate = float(cfv.get("sample_rate"))
    file_secs = float(cfv.get("file_secs"))
    samples_per_file = int(file_secs * sample_rate)
    chnl = cfv.get("rx_channel", "tbd")
    radar_frequency = float(mat["d_parbl"][0][PARBL_RADAR_FREQUENCY])
    
    def index_of_filestart(mat):
        """
        Returns sample index of start of file
        """
        # global start time for sampling (repeated for all files)
        ts_origin_sec = float(mat["d_parbl"][0][PARBL_START_TIME])
        # sample index corresponding to global start time
        idx_origin = int(np.floor(ts_origin_sec * sample_rate))

        # end time of file
        ts_endfile_sec = float(mat["d_parbl"][0][PARBL_END_TIME])

        # NOTE: ts_endfile_sec can not be trusted to be precisely
        # consistent with ts_origin_sec - in terms of samples.
        # we use the approach of counting samples from ts_orgin_sec,
        # relying on a fixed sample count per file

        # NOTE: file_idx is a logical sequence number for files, starting from 
        # file_idx 0 at ts_origin_sec. This does not correspond to the order of files processed.
        # For instance, the first file might have file_idx 6, indicating that the
        # recording only started some time after ts_origin_sec 
        # also the inprecision in ts_endfile_sec goes away in division
        file_idx = round((ts_endfile_sec - ts_origin_sec) / file_secs) - 1
        # sample index for start of first file
        return idx_origin + file_idx * samples_per_file



    #######################################################################
    # WRITER SETUP
    #######################################################################

    # sample index start of first file
    idx_start = index_of_filestart(mat)
    # timestamp start of first file
    ts_start_sec = idx_start / sample_rate

    # create sample writer
    sample_writer = drf_wrapper.DigitalRFWriter(hdrf, chnl,
        sample_rate, # sample rate numerator
        1, # samplerate denominator
        np.int16,
        idx_start,
        subdir_cadence_secs=3600, # one dir per hour
        file_cadence_secs=1, # one file per second
        is_complex=True,
        compression_level=compression,
        uuid_str=chnl,
        ts_align_sec=ts_start_sec
    )

    # create pointing writer
    pointing_writer = drf_wrapper.DigitalMetadataWriter(
        hdrf, "pointing",
        sample_rate, # sample rate - numerator (int)
        samples_per_file, # sample rate - denominator (int)
    )

    #######################################################################
    # WRITE
    #######################################################################

    # NOTE: need to write file sequentially. Files are sorted.
    # Also data for all files must be fixed length. If data is too short
    # it must be padded, or truncated if too long
    # also - there could be missing files in the sequence

    def zeropad(n_pad, file):
        try:
            sample_writer.write(np.zeros(n_pad*2, dtype=np.int16))
        except Exception as e:
            err = f"unable to zero pad samples for {file}"
            if logger:
                logger.error(err)
            raise e
        if logger:
            logger.warning(f"zero padding {n_pad} samples for {file}")

    def write(data, file):
        try:
            sample_writer.write(data)
        except Exception as e:
            err = f"unable to write samples for {file}"
            if logger:
                logger.error(err)
            raise e


    # initialise writing loop

    idx_prev = idx_start - samples_per_file
    n_files = len(files)
    if logger:
        logger.info(f"writing DRF from {n_files} input files")

    if progress:
        pbar = tqdm(desc="Converting files to digital_rf", total=n_files)

    for progress_idx, file in enumerate(files):
        if logger:
            if progress_idx + 1 == n_files or progress_idx % 10 == 0:
                logger.debug(f"write progress {progress_idx+1}/{n_files}")

        mat = loadmat(file)
        # sample index
        idx = index_of_filestart(mat)

        # check idx of file
        # idx of file should be the logical continuation, samples_per_file larger
        # than the previous index
        # if it is not - a file could be missing - try to zero pad until where you
        # expected to be - given this idx
        if idx-idx_prev != samples_per_file:
            if idx - samples_per_file > idx_prev:
                # missing file
                n_pad = (idx-idx_prev) - samples_per_file
                zeropad(n_pad, file)
            else:
                err = f"wrong sample index from file {file}, got {idx}, expected {idx_prev + samples_per_file}"
                if logger:
                    logger.error(err)
                raise Exception(err)

        # data
        zz = to_i2x16(mat["d_raw"][:, 0])
        
        # check data
        n_samples = len(zz)
        if n_samples > samples_per_file:
            # truncate
            zz = zz[:samples_per_file]
            if logger:
                logger.warning(f"truncating from {n_samples} samples to {samples_per_file}")

        # write data
        write(zz, file)

        # if data was too short, zero pad
        if n_samples < samples_per_file:
            # zero padding
            n_pad = samples_per_file - n_samples
            zeropad(n_pad, file)

        # write pointing data
        ts = sample_writer.ts_from_index(idx)
        pointing_idx = int(pointing_writer.index_from_ts(ts))
        d = {
            'azimuth': float(mat["d_parbl"][0][PARBL_AZIMUTH]) % 360,
            'elevation': float(mat["d_parbl"][0][PARBL_ELEVATION])
        }
        pointing_writer.write(pointing_idx, d)

        # increment idx_prev
        idx_prev = idx
        if progress:
            pbar.update(1)
    if progress:
        pbar.close()

    if logger:
        logger.info("Done writing DRF files")


    #######################################################################
    # WRITE METADATA
    #######################################################################

    EXP_SECTION = "Experiment"
    meta = configparser.ConfigParser()
    meta.add_section(EXP_SECTION)
    exp = meta[EXP_SECTION]
    exp["name"] = expname
    exp["version"] = expvers

    # forward values from experiment config file
    props = [
        "sample_rate", "ipp", "file_secs", "tx_pulse_length",
        "rx_channel", "rx_start", "rx_end",
        "tx_channel", "tx_start", "tx_end",
        "cal_on", "cal_off"
    ]
    for prop in props:
        exp[prop] = cfv.get(prop)

    # add
    exp["radar_frequency"] = str(radar_frequency)

    # write metadata file
    metafile = hdrf / "metadata.ini"
    with open(metafile, 'w') as f:
        meta.write(f)

    return str(hdrf)



if __name__ == '__main__':
    PROJECT = Path("/cluster/projects/p106119-SpaceDebrisRadarCharacterization")
    SRC = PROJECT / "raw/leo_bpark_2.1u_NO-20220408-UHF/leo_bpark_2.1u_NO@uhf"
    CFG = "/cluster/home/inar/Dev/Git/hardtarget/examples/cfg/test.ini"


