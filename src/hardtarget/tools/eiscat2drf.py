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

# Number of samples in file. 640 ipps. eiscat leo experiment specific number!
NUM_SAMPLES = 640 * 20000

LOGGER_NAME = "eiscat2drf"

def find_configfile(name, ext='.ini'):
    # print(__file__)
    pth = __file__
    cfgdir = pth[:pth.rindex('src/hardtarget')] + 'cfg/'
    cfile = cfgdir + name + ext
    assert os.path.isfile(cfile), f"Config file {cfile} for {name} not found"
    return cfile

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


def determine_n0(mat, cfv=None):
    """
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

    samp_rate = int(cfv.get('sample_rate'))             # assuming integral # samples per second
    t0 = float(mat["d_parbl"][0][42])
    tx = float(mat["d_parbl"][0][10])

    n_epoch = round(mat["d_parbl"][0][42] * samp_rate)
    N_samp = len(mat['d_raw'])                  # sanity check this value?
    N_sec = N_samp/samp_rate                    # number of seconds in each file
    i_file = round((tx-t0)/N_sec)-1             # -1 because parbl[10] records _end_ time of file

    return n_epoch + N_samp * i_file


# Original version of the above
def determine_t0(mat):
    t0 = int(mat["d_parbl"][0][42] * 1000000)
    dt = NUM_SAMPLES
    return t0 + int(np.round((1e6 * mat["d_parbl"][0][10] - t0) / dt) - 1) * dt



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

    # Helper functions

    # generate all files matching 'dddddddd_dd/dddddddd.mat' or '.mat.bz2'
    # in sorted order
    def all_files(top):
        d = '[0-9]'
        f = lambda pth: pth.endswith('.mat') or pth.endswith('mat.bz2')

        dirs = sorted(glob.glob(f"{top}/{8*d}_{2*d}"))
        return it.chain( *(sorted(filter(f, glob.glob(f"{dir}/{8*d}.mat*"))) for dir in dirs) )

    # A version of loadmat which transparently unzips files on the fly (not in the filesystem)
    def loadmat(pth):
        if pth.endswith('.mat.bz2'):
            pth = bz2.open(pth, 'rb')
        return sio.loadmat(pth)

    # Move from hard coded constants to loading config based on exp name/version
    def expinfo_split(xpinf):
        # 'kst0 leo_bpark_2.1u_NO' -> ('kst0', 'leo_bpark', '2.1u', 'NO')
        try:
            # host, name, versi, owner = \
            match = \
                re.match(r'(\w+) +(\w+)_(\d+(?:\.\d+)?[vu])_([A-Z]{2})', xpinf)
            # return host, name, '_'.join(ver.spl, [site]), owner
            return match.groups()
        except:
            raise ValueError(f"d_ExpInfo: {xpinf} not understood")

    def load_expconfig(xpname, xpvers):
        try:
            cfg_file = find_configfile(xpname)
            cfg = configparser.ConfigParser()
            cfg.read_file(open(cfg_file, 'r'))
            return cfg
        except Exception as e:
            raise ValueError(f"Couldn't open config file for {xpname}:" + str(e))

    if dstdir is None:
        dstdir = os.path.join(srcdir, 'drf')

    if not os.path.isdir(dstdir):
        os.makedirs(dstdir, exist_ok=True)
    # Verify that folder is empty
    if len(os.listdir(dstdir)) > 0:
        logger.warning(f"output folder is not empty: {dstdir}")
        return


    files = list(all_files(srcdir))

    pth = files[0]

    # load start time from parameter block of first matlab file
    mat = loadmat(pth)
    upar = mat['d_parbl'][0, 41:62]
    radar_frequency = upar[13]

    # Find experiment info from first file
    host, expname, expvers, owner = expinfo_split(str(mat['d_ExpInfo'][0]))

    cfg = load_expconfig(expname, expvers)
    cfv = cfg[expvers]          # config for this version of the experiment (mode)
    sample_rate = int(cfv.get('sample_rate'))             # assuming integral # samples per second
    file_secs   = float(cfv.get('file_secs'))
    n_samples   = round(file_secs * sample_rate)

    n0 = determine_n0(mat, cfv)

    # create digital rf writer
    rf_writer = drf.DigitalRFWriter(
        dstdir,  # directory
        np.complex64,  # dtype
        3600,  # subdir cadence secs    => one dir per hour
        1000,  # file cadence millisecs => one file per second
        n0,    # start global index
        sample_rate,  # sample rate numerator
        1,  # sample rate denominator
        uuid_str=cfv.get('rx_channel', 'tbd'),
        compression_level=0,
        checksum=False,
        is_complex=True,
        num_subchannels=1,
        is_continuous=True,
        marching_periods=True,
    )

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
            n_pad = (n0 - n_prev) - n_samples
            try:
                rf_writer.rf_write(np.zeros(n_pad, dtype=np.complex64))
            except:
                logging.warning("unable to pad out for missing files ... continuing")

        zz = np.array(mat['d_raw'][:, 0], dtype=np.complex64)
        if len(zz) != n_samples:
            logging.warning(f"found {len(zz)} samples in {pth}['d_raw'], expected {n_samples}")
        try:
            rf_writer.rf_write(zz)
        except:
            logging.warning(f"unable to write samples from {pth} to file ... continuing")
        n_prev = n0

    logging.info("Done writing DRF files")



def old_eiscat2drf_dont_use(input, output=None, logger=None):
    """
    Converts folder with eiscat measurements to drf files.

    Parameters
    ----------
    input: string
        path to input directory
    output: string, optional
        path ot output direction (default input)

    Returns
    -------


    Notes
    -----
    Assumes folder structure
    - input/2*/

    Files within this folder will either be zipped (.b2z),
    or matlab files (.mat). Zipped files are expected to
    produce (.mat) files when extracted.

    Result is put in subfolder within output folder.
    - output/drf/uhf/
    """

    if logger is None:
        logger = logging.getLogger(LOGGER_NAME)

    # Check input dirpat
    if not os.path.isdir(input):
        logger.warning(f"input folder does not exist: {input}")
        return

    # Default output dirpath == input dirpath
    if output is None:
        output = input
    # Check output dirpath
    if not os.path.isdir(output):
        logger.warning(f"output folder does not exists: {output}")
        return
    # Create folder structure
    write_dirpath = os.path.join(output, "drf/uhf")
    if not os.path.isdir(write_dirpath):
        os.makedirs(write_dirpath, exist_ok=True)
    # Verify that folder is empty
    if len(os.listdir(write_dirpath)) > 0:
        logger.warning(f"output folder is not empty: {write_dirpath}")
        return

    # find zipped files
    zipped_files = glob.glob(f"{input}/2*/*.mat.bz2")

    # map to zip pairs
    def tup(f_in):
        f_out, ext = os.path.splitext(f_in)
        return f_in, f_out

    zip_tuples = [tup(f) for f in zipped_files]

    # filter zip pairs
    def keep(f_out):
        return not os.path.isfile(f_out)

    # unzip
    zip_tuples = [t for t in zip_tuples if keep(t[1])]
    n_tuples = len(zip_tuples)
    logger.info(f"unzip {n_tuples} bz2 files")
    for idx, (in_file, out_file) in enumerate(zip_tuples):
        if idx + 1 == n_tuples or idx % 10 == 0:
            logger.info(f"unzip progress {idx+1}/{n_tuples}")
        with bz2.open(in_file, "rb") as f_in:
            with open(out_file, "wb") as f_out:
                f_out.write(f_in.read())

    # find matlab files
    files = glob.glob(f"{input}/2*/*.mat")
    files.sort()

    # load start time from parameter block of first matlab file
    mat = sio.loadmat(files[0])
    t0 = determine_t0(mat)

    # create digital rf writer
    rf_writer = drf.DigitalRFWriter(
        write_dirpath,  # directory
        np.complex64,  # dtype
        3600,  # subdir cadence secs
        1000,  # file candence millisecs
        t0,  # start global index
        1000000,  # sample rate numerator
        1,  # sample rate denominator
        uuid_str="uhf",
        compression_level=0,
        checksum=False,
        is_complex=True,
        num_subchannels=1,
        is_continuous=True,
        marching_periods=True,
    )

    # write drf files
    t_prev = t0
    n_files = len(files)
    logger.info(f"write {n_files} rdf files")
    for idx, file in enumerate(files):
        if idx + 1 == n_files or idx % 10 == 0:
            logger.info(f"write progress {idx+1}/{n_files}")
        mat = sio.loadmat(file)
        # load start time from parameter block
        t0 = determine_t0(mat)
        logger.debug(f"n_samp {t0 - t_prev}")

        # if start time is not 12800000 samples more than previous one, we
        # have a missing data file. we must pad zeros into the data
        if t0 - t_prev != 12800000 and t0 - t_prev != 0:
            n_samp = (t0 - t_prev) - NUM_SAMPLES
            logger.debug(f"padding zeros {n_samp}")
            zz = np.zeros(n_samp, dtype=np.complex64)
            try:
                rf_writer.rf_write(zz)
            except Exception:
                pass

        # write data file
        z = np.array(mat["d_raw"][:, 0], dtype=np.complex64)
        if len(z) == NUM_SAMPLES:
            try:
                rf_writer.rf_write(z)
            except Exception:
                pass
        # save t0 as t_prev
        t_prev = t0


####################################################################
# SCRIPT ENTRY POINT
####################################################################


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="Script converting eiscat data to drf format",
        usage="%(prog)s [options] input -o output_folder",
    )

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

    # Parse the arguments
    args = parser.parse_args()

    # Logging
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, args.log_level))

    eiscat2drf(args.input, dstdir=args.output, logger=logger)
    print()


####################################################################
# MAIN
####################################################################

if __name__ == "__main__":
    main()
