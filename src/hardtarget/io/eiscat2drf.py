#!/usr/bin/env python

import argparse
import glob
import scipy.io as sio
import digital_rf as drf
import os
import numpy as np
import bz2
import logging

# import from this directory. this is needed if this is a package
# from . import parbl
from hardtarget.io import parbl

# Number of samples in file. 640 ipps. eiscat leo experiment specific number!
L = 640 * 20000


####################################################################
# EISCAT 2 DRF
####################################################################


def eiscat2drf(dirpath, 
               output_dirpath=None, 
               log_level=logging.INFO):

    """
    Converts folder with eiscat measurements to drf files.

    Assume folder structure
    - dirpath/drf/uhf/S*/

    Files within this folder will either be zipped (.b2z),
    or matlab files (.mat). Zipped files are expected to
    produce (.mat) files when extracted.

    """

    # logging
    logging.basicConfig(
        level=log_level, 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    subdirpath = os.path.join(dirpath, "drf/uhf")
    if not os.path.isdir(subdirpath):
        logging.warning(f'folder does not exist: {subdirpath}')
        return

    # find zipped files
    zipped_files = glob.glob(f'{subdirpath}/2*/*.mat.bz2')
    # limit to one for now
    zipped_files = zipped_files[:1]
    # unzip zipped files
    logging.info(f"unzipping {len(zipped_files)} files")
    for zipped_file in zipped_files:
        unzipped_file, ext = os.path.splitext(zipped_file)
        with bz2.open(zipped_file, "rb") as f_in:
            with open(unzipped_file, "wb") as f_out:
                f_out.write(f_in.read())

    # find matlab files
    files = glob.glob(f'{subdirpath}/2*/*.mat')
    files.sort()

    # load start time from parameter block of first matlab file
    mat = sio.loadmat(files[0])
    t0 = parbl.determine_t0_24(mat)[0]

    # create digital rf writer
    if output_dirpath is None:
        output_dirpath = subdirpath
        if not os.path.isdir(output_dirpath):
            logging.warning(f'output folder does not exist: {output_dirpath}')
            return

    rf_writer = drf.DigitalRFWriter(
        output_dirpath,
        np.complex64,
        3600,
        1000,
        t0,
        1000000,
        1,
        "uhf",
        compression_level=0,
        checksum=False,
        is_complex=True,
        num_subchannels=1,
        is_continuous=True,
        marching_periods=True,
    )

    # ipp
    # ipp = 20000
    # ipp_idx = np.arange(ipp)
    # tvec = np.zeros(len(files))
    # pvec = np.zeros(len(files))

    # write drf files
    t_prev = t0
    logging.info(f"writing {len(files)} rdf files")
    for file in files:
        mat = sio.loadmat(file)
        # load start time from parameter block
        t0 = parbl.determine_t0_24(mat)[0]
        logging.debug(f"n_samp {t0 - t_prev}")

        # if start time is not 12800000 samples more than previous one, we
        # have a missing data file. we must pad zeros into the data
        if t0 - t_prev != 12800000 and t0 - t_prev != 0:
            n_samp = (t0 - t_prev) - 12800000
            logging.debug(f"padding zeros {n_samp}")
            zz = np.zeros(n_samp, dtype=np.complex64)
            rf_writer.rf_write(zz)

        # write data file
        z = np.array(mat["d_raw"][:, 0], dtype=np.complex64)
        # z_txp = np.zeros(20000)
        if len(z) == 12800000:
            rf_writer.rf_write(z)
        # save t0 as t_prev
        t_prev = t0


####################################################################
# SCRIPT ENTRY POINT
####################################################################

def main():

    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="Script converting eiscat data to drf format",
        usage= "%(prog)s [options] leo_bpark_2.1u_NO@uhf -o output_folder"
    )

    # Add the arguments
    parser.add_argument("input", help="Path to source directory")
    parser.add_argument(
        "-o", "--output", 
        help="Path to output directory", 
        default="./drf/uhf")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the log level (default: INFO)"
    )

    # Parse the arguments
    args = parser.parse_args()
    eiscat2drf(
        args.input, 
        output_dirpath=args.output, 
        log_level=args.log_level
    )


####################################################################
# MAIN
####################################################################

if __name__ == "__main__":
    main()