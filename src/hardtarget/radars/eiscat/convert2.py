from pathlib import Path
import glob
import re
import bz2
import numpy as np
import scipy.io as sio
import itertools as it
from hardtarget.radars.eiscat import load_expconfig
import hardtarget.radars.eiscat.digitalrf_wrapper as drf_wrapper
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
    d["file_idx"] = idx_file = round((ts_end - ts_origin) / duration) - 1

    # global index of first sample in recording (t_origin)
    d["sample_idx_origin"] = idx_origin = round(ts_origin * sample_rate)
    # global index of first sample in file
    d["sample_idx_start"] = idx_start = idx_origin + idx_file * samples
    # global index of next sample after file
    d["sample_idx_end"] = idx_start + samples    

    # instrument pointing angles
    d["elevation"] = elevation = float(parbl[PARBL_ELEVATION])
    d["azimuth"] = azimuth = float(parbl[PARBL_AZIMUTH]) % 360
   

    return d



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
    sample_rate = int(cfv.get("sample_rate"))
    file_secs = float(cfv.get("file_secs"))
    chnl = cfv.get("rx_channel", "tbd")
    # start time for sampling (repeated for all files)
    ts_origin_sec = float(mat["d_parbl"][0][PARBL_START_TIME])
    radar_frequency = float(mat["d_parbl"][0][PARBL_RADAR_FREQUENCY])

    #######################################################################
    # WRITER SETUP
    #######################################################################

    # create sample writer
    sample_writer = drf_wrapper.DigitalRFWriter(hdrf, chnl,
        sample_rate, # sample rate numerator
        1, # samplerate denominator
        np.int16,
        ts_origin_sec=ts_origin_sec,
        subdir_cadence_secs=3600, # one dir per hour
        file_cadence_secs=1, # one file per second
        is_complex=True,
        compression_level=compression,
        uuid_str=chnl
    )

    # create pointing writer
    pointing_writer = drf_wrapper.DigitalMetadataWriter(
        hdrf, "pointing",
        sample_rate, # sample rate numerator (int)
        int(file_secs * sample_rate), # sample rate denominator (int)
    )


    #######################################################################
    # WRITE
    #######################################################################

    # all data have to be written sequentially, from ts_orgin_sec
    # files are sorted, but apparently there is no guarantee that the first
    # file starts at ts_origin_sec, so zero padding is needed
    n_files = len(files)
    if logger:
        logger.info(f"writing DRF from {n_files} input files")

    if progress:
        pbar = tqdm(desc="Converting files to digital_rf", total=n_files)

    # convert
    write_idx = sample_writer.index_from_ts(ts_origin_sec)
    for file_idx, file in list(enumerate(files))[:1]:
        if logger:
            if file_idx + 1 == n_files or file_idx % 10 == 0:
                logger.debug(f"write progress {file_idx+1}/{n_files}")


        mat = loadmat(file)
        meta = parse_matlab(mat, sample_rate, file_secs)

        # zero pad if start of file is ahead of write idx
        start_idx = sample_writer.index_from_ts(meta["ts_start"])
        if start_idx > write_idx:
            n_pad = start_idx - write_idx
            sample_writer.write(np.zeros(n_pad*2, dtype=np.int16))
            # advance write index
            write_idx += n_pad

        # load data
        data = mat["d_raw"][:, 0]
        # make sure data is not too long
        data = data[:meta["n_samples"]]

        # write data
        zz = to_i2x16(data)
        sample_writer.write(zz)

        # advance write_idx
        write_idx += len(data)

        # write pointing
        idx = pointing_writer.index_from_ts(meta["ts_start"])
        pointing_writer.write(idx, meta["azimuth"], meta["elevation"])

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


    DST = "/cluster/projects/p106119-SpaceDebrisRadarCharacterization/raw"
    PRODUCT = "leo_bpark_2.1u_NO-20180104-UHF/leo_bpark_2.1u_NO@uhf"
    SRC = Path(DST) / PRODUCT
    DST = "/tmp/eiscat"
    convert(SRC, DST)

