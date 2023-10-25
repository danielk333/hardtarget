#!/usr/bin/env python

import argparse
import glob
import re
import itertools as it
import scipy.io as sio
import digital_rf as drf
import os
import numpy as np
import bz2
import logging
import configparser
from pathlib import Path

from hardtarget.experiments import EXP_FILES

# Number of samples in file. 640 ipps. eiscat leo experiment specific number!
NUM_SAMPLES = 640 * 20000

LOGGER_NAME = "eiscat2drf"

icomplex32 = np.dtype([
    ('real', np.int16),
    ('imag', np.int16)])


####################################################################
# ISSUES
####################################################################

"""
DigitalRFWriter
- no control over output file names. This makes it difficult
  to gracefully detect that work is done
- Algorithm does not appear to support resuming of a previously
  interupted/partial job
- For this reason - output directory must be empty
- Internal exceptions and progress printed to stdout. Possibly
  there could be an option for keeping it silent.
"""

####################################################################
# MATLAB PARAMETER BLOCK
####################################################################


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
    t0 = float(mat["d_parbl"][0][42])
    tx = float(mat["d_parbl"][0][10])

    n_epoch = round(t0 * samp_rate)
    N_samp = len(mat["d_raw"])  # sanity check this value?
    N_sec = N_samp / samp_rate  # number of seconds in each file
    i_file = round((tx - t0) / N_sec) - 1  # -1 because parbl[10] records _end_ time of file

    return n_epoch + N_samp * i_file


# Original version of the above
def determine_t0(mat):
    t0 = int(mat["d_parbl"][0][42] * 1000000)
    dt = NUM_SAMPLES
    return t0 + int(np.round((1e6 * mat["d_parbl"][0][10] - t0) / dt) - 1) * dt


def to_icomplex32(zz):
    zz32 = np.empty(zz.shape, dtype=icomplex32)
    zz32['real'] = zz.real.astype(np.int16)
    zz32['imag'] = zz.imag.astype(np.int16)
    return zz32


def to_i2x16(zz):
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


def loadmat(pth):
    """A version of loadmat which transparently unzips files on the fly (not in the filesystem)
    """
    if pth.endswith(".mat.bz2"):
        pth = bz2.open(pth, "rb")
    return sio.loadmat(pth)


def expinfo_split(xpinf):
    """Move from hard coded constants to loading config based on exp name/version

    'kst0 leo_bpark_2.1u_NO' -> ('kst0', 'leo_bpark', '2.1u', 'NO')
    """
    try:
        # host, name, versi, owner = \
        match = re.match(r"(\w+) +(\w+)_(\d+(?:\.\d+)?[vu])_([A-Z]{2})", xpinf)
        # return host, name, '_'.join(ver.spl, [site]), owner
        return match.groups()
    except Exception as e:
        raise ValueError(f"d_ExpInfo: {xpinf} not understood: {e}")


def load_expconfig(xpname):
    cfg_name = xpname + ".ini"
    assert cfg_name in EXP_FILES, "experiment not found in pre-defined configurations"
    cfg_file = EXP_FILES[cfg_name]
    try:
        cfg = configparser.ConfigParser()
        cfg.read_file(open(cfg_file, "r"))
        return cfg
    except Exception as e:
        raise ValueError(f"Couldn't open config file for {xpname}:" + str(e))

    return


####################################################################
# EISCAT 2 DRF
####################################################################


def eiscat2drf(srcdir, dstdir=None, logger=None):
    """
    Converts folder with eiscat measurements to drf files.

    Parameters
    ----------
    srcdir: string
        path to source directory
    dstdir: string, optional
        path to destination directory (default: same as source)


    Returns
    -------


    Notes
    -----
    Assumes directory structure
    - <srcdir>/<YYYYMMDD_HH>/

    Files within each such directory will either be zipped (.b2z),
    or matlab files (.mat). Zipped files are expected to
    produce (.mat) files when extracted.

    Result is put in subfolder within output folder.
    - dstdir/drf/uhf/
    """

    if logger is None:
        logger = logging.getLogger(LOGGER_NAME)

    #######################################################################
    # SOURCE DIR
    #
    # support single file or folder with files
    #######################################################################

    if Path(srcdir).is_file():
        files = [srcdir]
    else:
        files = list(all_files(srcdir))

    #######################################################################
    # EXTRACT META DATA
    #
    # from one file
    #######################################################################

    pth = files[0]

    # load start time from parameter block of first matlab file
    mat = loadmat(pth)
    # upar = mat["d_parbl"][0, 41:62]
    # radar_frequency = upar[13]

    # Find experiment info from first file
    host, expname, expvers, owner = expinfo_split(str(mat["d_ExpInfo"][0]))

    cfg = load_expconfig(expname)
    cfv = cfg[expvers]  # config for this version of the experiment (mode)
    sample_rate = int(cfv.get("sample_rate"))  # assuming integral # samples per second
    file_secs = float(cfv.get("file_secs"))
    n_samples = round(file_secs * sample_rate)

    n0 = determine_n0(mat, cfv)

    #######################################################################
    # DESTINATION DIR
    #
    #######################################################################

    if dstdir is None:
        if Path(srcdir).is_file():
            dstdir = Path(srcdir).parent / "drf"
        else:
            dstdir = Path(srcdir) / "drf"
    else:
        dstdir = Path(dstdir)

    # add channel subdirectory
    chnl = cfv.get("rx_channel", "tbd")
    dstdir = dstdir / chnl
    # make sure dstdir exists
    if not dstdir.is_dir():
        os.makedirs(dstdir, exist_ok=True)

    # create digital rf writer
    rf_writer = drf.DigitalRFWriter(
        str(dstdir),  # directory
        np.int16,  # dtype
        3600,  # subdir cadence secs    => one dir per hour
        1000,  # file cadence millisecs => one file per second
        n0,  # start global index
        sample_rate,  # sample rate numerator
        1,  # sample rate denominator
        uuid_str=cfv.get("rx_channel", "tbd"),
        compression_level=0,
        checksum=False,
        is_complex=True,
        num_subchannels=1,
        is_continuous=True,
        marching_periods=True,
    )

    # TODO - compression, sensible data type, customizeable arg for DRF, limit, support 1 file
    n_prev = n0 - n_samples
    n_files = len(files)
    logger.info(f"writing DRF from {n_files} input files")
    for idx, pth in enumerate(files):
        if idx + 1 == n_files or idx % 10 == 0:
            logger.info(f"write progress {idx+1}/{n_files}")
        mat = loadmat(pth)
        n0 = determine_n0(mat, cfv)
        logger.debug(f"n_samp {n0 - n_prev} (should be {n_samples})")
        if n0 - n_prev != n_samples:
            # zero padding
            n_pad = (n0 - n_prev) - n_samples
            try:
                rf_writer.rf_write(np.zeros(n_pad*2, dtype=np.int16))
            except Exception:
                logging.warning("unable to pad out for missing files ... continuing")

        zz = to_i2x16(mat["d_raw"][:, 0])
        if len(zz) != n_samples:
            logging.warning(f"found {len(zz)} samples in {pth}['d_raw'], expected {n_samples}")
        try:
            rf_writer.rf_write(zz)
        except Exception as e:
            logging.warning(f"unable to write samples from {pth} to file ... continuing")
            raise e
        n_prev = n0

    logging.info("Done writing DRF files")


####################################################################
# SCRIPT ENTRY POINT
####################################################################

def parser_build(parser):
    # Add the arguments
    parser.add_argument(
        "input",
        help="Path to source directory, assumes folder structure 'input/2*/*.mat or *.mat.bz2'",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output directory, default output folder 'input/drf/uhf/'",
        default=None,
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the log level (default: INFO)",
    )
    return parser


def main(args, cli_logger):
    # Logging
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, args.log_level))

    eiscat2drf(args.input, dstdir=args.output, logger=logger)


####################################################################
# MAIN
####################################################################

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="Script converting eiscat data to drf format",
        usage="%(prog)s [options] input -o output_folder",
    )
    parser = parser_build(parser)
    # Parse the arguments
    args = parser.parse_args()
    main(args, None)
