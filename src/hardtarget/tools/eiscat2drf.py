#!/usr/bin/env python

import argparse
import glob
import scipy.io as sio
import digital_rf as drf
import os
import numpy as np
import bz2
import logging

# Number of samples in file. 640 ipps. eiscat leo experiment specific number!
NUM_SAMPLES = 640 * 20000

LOGGER_NAME = "eiscat2drf"


####################################################################
# ISSUES
####################################################################

"""
DigitalRFWriter
- no control over output file names. this makes it difficult
  to gracefully detect that work is done
- algorithm does not appear to support resuming of a previously
  interupted/partial job
- for this reason - output directory must be empty
- internal exceptions and progress printed to stdout
"""

####################################################################
# MATLAB PARAMETER BLOCK
####################################################################

def determine_t0(mat):
    t0=int(mat["d_parbl"][0][42]*1000000)
    dt=NUM_SAMPLES
    return t0+int(np.round((1e6*mat["d_parbl"][0][10]-t0)/dt)-1)*dt


####################################################################
# EISCAT 2 DRF
####################################################################

def eiscat2drf(input_dirpath, 
               output_dirpath=None, 
               logger=None):

    """
    Converts folder with eiscat measurements to drf files.

    Assume folder structure
    - dirpath/2*/

    Files within this folder will either be zipped (.b2z),
    or matlab files (.mat). Zipped files are expected to
    produce (.mat) files when extracted.

    Default output folder
    - dirpath/drf/uhf/
    """

    if logger is None:
        logger = logging.getLogger(LOGGER_NAME)

    # Check input dirpat
    if not os.path.isdir(input_dirpath):
        logger.warning(f'input folder does not exist: {input_dirpath}')
        return
    
    # Default output dirpath == input dirpath
    if output_dirpath is None:
        output_dirpath = input_dirpath
    # Check output dirpath
    if not os.path.isdir(output_dirpath):
        logger.warning(f'output folder does not exists: {output_dirpath}')
        return
    # Create folder structure
    write_dirpath = os.path.join(output_dirpath, "drf/uhf")
    if not os.path.isdir(write_dirpath):
        os.makedirs(write_dirpath, exist_ok=True)
    # Verify that folder is empty
    if len(os.listdir(write_dirpath)) > 0:
        logger.warning(f"output folder is not empty: {write_dirpath}")
        return

    # find zipped files
    zipped_files = glob.glob(f'{input_dirpath}/2*/*.mat.bz2')
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
    for idx, (in_file, out_file)  in enumerate(zip_tuples):
        if idx + 1 == n_tuples or idx % 10 == 0:
            logger.info(f"unzip progress {idx+1}/{n_tuples}")
        with bz2.open(in_file, "rb") as f_in:
            with open(out_file, "wb") as f_out:
                f_out.write(f_in.read())

    # find matlab files
    files = glob.glob(f'{input_dirpath}/2*/*.mat')
    files.sort()

    # load start time from parameter block of first matlab file
    mat = sio.loadmat(files[0])
    t0 = determine_t0(mat)

    # create digital rf writer
    rf_writer = drf.DigitalRFWriter(
        write_dirpath, # directory
        np.complex64, # dtype
        3600, # subdir cadence secs
        1000, # file candence millisecs
        t0, # start global index
        1000000, # sample rate numerator
        1, # sample rate denominator
        uuid_str="uhf",
        compression_level=0,
        checksum=False,
        is_complex=True,
        num_subchannels=1,
        is_continuous=True,
        marching_periods=True 
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
            except Exception as e:
                pass

        # write data file
        z = np.array(mat["d_raw"][:, 0], dtype=np.complex64)
        if len(z) == NUM_SAMPLES:
            try:
                rf_writer.rf_write(z)
            except Exception as e:
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
        usage= "%(prog)s [options] input -o output_folder"
    )

    # Add the arguments
    parser.add_argument("input", help="Path to source directory, assumes folder structure 'input/2*/*.mat or *.mat.bz2'")
    parser.add_argument(
        "-o", "--output", 
        help="Path to output directory, default output folder 'input/drf/uhf/'",
        default=None)
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the log level (default: INFO)"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Logging
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, args.log_level))

    eiscat2drf(
        args.input, 
        output_dirpath=args.output,
        logger = logger
    )
    print()

####################################################################
# MAIN
####################################################################

if __name__ == "__main__":
    main()