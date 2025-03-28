from pathlib import Path
import numpy as np
import hardtarget.digitalrf_wrapper as drf_wrapper
import configparser
from tqdm import tqdm
from hardtarget.radars.eiscat import eiscat_load_file, eiscat_files, eiscat_process


####################################################################
# CONVERT
####################################################################

def convert(src, dst, name=None, compression=0, progress=False, logger=None,
            start=None, end=None, count=None):

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
    files = eiscat_files(src)
    n_files = len(files)

    #######################################################################
    # META DATA
    #######################################################################

    # load experiment info from first matlab file
    meta_first = eiscat_load_file(files[0])[0]

    #######################################################################
    # WRITER SETUP
    #######################################################################

    # create sample writer
    sample_writer = drf_wrapper.DigitalRFWriter(
        hdrf,
        meta_first["exp"]["chnl"],
        meta_first["exp"]["sample_rate"],  # sample rate numerator
        1,  # samplerate denominator
        np.int16,
        meta_first["sample"]["file_start"],
        subdir_cadence_secs=3600,  # one dir per hour
        file_cadence_secs=1,  # one file per second
        is_complex=True,
        compression_level=compression,
        uuid_str=meta_first["exp"]["chnl"],
        ts_align_sec=meta_first["ts"]["file_start"]
    )

    # create pointing writer
    pointing_writer = drf_wrapper.DigitalMetadataWriter(
        hdrf, "pointing",
        meta_first["exp"]["sample_rate"],  # sample rate - numerator (int)
        meta_first["exp"]["samples_per_file"],  # sample rate - denominator (int)
    )

    #######################################################################
    # WRITE
    #######################################################################

    def pad_data(n_pad, file):
        try:
            sample_writer.write(np.zeros(n_pad*2, dtype=np.int16))
        except Exception as e:
            err = f"unable to zero pad samples for {file}"
            if logger:
                logger.error(err)
            raise e
        if logger:
            logger.warning(f"zero padding {n_pad} samples for {file}")

    def write_data(data, file):
        try:
            sample_writer.write(data)
        except Exception as e:
            err = f"unable to write samples for {file}"
            if logger:
                logger.error(err)
            raise e

    def drop_data(errors, file):
        if logger:
            logger.info(f"dropping : {errors} : {file}")

    def log_progress(idx, n_files, period=10):
        if logger:
            if idx + 1 == n_files or idx % period == 0:
                logger.debug(f"write progress {idx+1}/{n_files}")

    def write_pointing(sample, pointing_data):
        """write pointing data"""
        ts = sample_writer.ts_from_index(sample)
        pointing_idx = int(pointing_writer.index_from_ts(ts))
        pointing_writer.write(pointing_idx, pointing_data)

    if logger:
        logger.info(f"writing DRF from {n_files} input files")

    if progress:
        pbar = tqdm(desc="Converting files to digital_rf", total=n_files)

    # processing loop
    for meta in eiscat_process(files,
                               write_data=write_data,
                               pad_data=pad_data,
                               drop_data=drop_data,
                               write_pointing=write_pointing,
                               log_progress=log_progress
                               ):
        if meta["errors"]:
            if logger:
                logger.debug(meta["errors"])

        if progress:
            pbar.update(1)

    if progress:
        pbar.close()

    if logger:
        logger.info("Done writing DRF files")

    #######################################################################
    # WRITE METADATA
    #######################################################################

    meta = configparser.ConfigParser()

    # Experiment
    EXP_SECTION = "Experiment"

    meta.add_section(EXP_SECTION)
    exp = meta[EXP_SECTION]
    props = [
        "name", "version",
        "file_secs", "tx_pulse_length",
        "rx_start", "rx_end",
        "tx_start", "tx_end",
        "cal_on", "cal_off",
        "radar_frequency"
    ]
    for prop in props:
        exp[prop] = str(meta_first["exp"].get(prop))
    exp["rx_channel"] = meta_first["exp"]["chnl"]
    exp["tx_channel"] = meta_first["exp"]["chnl"]
    exp["sample_rate"] = str(int(meta_first["exp"]["sample_rate"]))
    exp["ipp"] = str(int(meta_first["exp"]["ipp"]))

    # Bounds
    BOUNDS_SECTION = "Bounds"
    meta.add_section(BOUNDS_SECTION)
    meta_last = eiscat_load_file(files[-1])[0]
    meta["Bounds"]["ts_start"] = str(meta_first["ts"]["file_start"])
    meta["Bounds"]["start"] = meta_first["date"]["file_start"]
    meta["Bounds"]["ts_end"] = str(meta_last["ts"]["file_end"])
    meta["Bounds"]["end"] = meta_last["date"]["file_end"]

    # write metadata file
    metafile = hdrf / "metadata.ini"
    with open(metafile, 'w') as f:
        meta.write(f)

    return str(hdrf)
