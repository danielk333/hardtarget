"""Process utilities"""

import datetime as dt
import logging
from pathlib import Path

import numpy as np

from hardtarget.types.types import Bounds, Job

logger = logging.getLogger(__name__)


def compute_job_tasks(job: Job, n_tasks: int) -> list[int]:
    """
    Generates a list of task indexes for given job.

    Args:
        job: Which job
        n_tasks: Total number of tasks

    Returns:
        List of task indexes for the given job.
    """
    return list(range(job.idx, n_tasks, job.N))


def compute_total_tasks(ipp_samps: int, n_ipp: int, num_cohints_per_file: int, bounds: Bounds) -> int:
    """
    Calculates the amount of tasks needed based on how many ipps should be coherently integrated, amount of
    samples for each ipp, the total amount of samples and the wanted amount of coherent integrations per file.

    Args:
        ipp_samps: Amount of interpulse period samples.
        n_ipp: Number or interpulse periods to coherently integrate.
        num_cohints_per_file: Number of coherent integrations per file.
        bounds: Measurement sample start and stop.
    Returns:
        Amount of tasks needed for the process.
    """

    n_tasks = np.ceil(
        np.floor((bounds.end - bounds.start) / (ipp_samps * n_ipp)) / num_cohints_per_file
    ).astype(int)
    return n_tasks


def calculate_tasks(
    job: Job, n_ipp: int, num_cohints_per_file: int, ipp_samps: int, bounds: Bounds
) -> tuple[list[int], int]:
    """
    Calculate the current jobs tasks and coherent integrations.

    Args:
        job: The current job.
        n_ipp:  Number or interpulse periods to coherently integrate.
        num_cohints_per_file: Number of coherent integrations per file.
        ipp_samps: Samples per interpulse period.
        bounds: Measurement sample start and stop.

    Returns:
        Job tasks for the current job and the jobs coherent integrations
    """

    total_tasks = compute_total_tasks(
        ipp_samps,
        n_ipp,
        num_cohints_per_file,
        bounds,
    )

    job_tasks = compute_job_tasks(job, total_tasks)

    if len(job_tasks) == 0:
        # Most likely more processes than tasks available
        return job_tasks, 0

    job_cohints = (len(job_tasks) - 2) * num_cohints_per_file

    num_cohints = num_cohints_per_file

    file_idx_sample = np.max(job_tasks) * ipp_samps * n_ipp * num_cohints_per_file + bounds.start
    if (file_idx_sample + num_cohints_per_file * ipp_samps * n_ipp) > bounds.end:
        num_cohints = int((bounds.end - file_idx_sample) // (ipp_samps * n_ipp))
    job_cohints += num_cohints
    num_cohints = num_cohints_per_file
    file_idx_sample = np.min(job_tasks) * ipp_samps * n_ipp * num_cohints_per_file + bounds.start
    if file_idx_sample < bounds.start:
        num_cohints = num_cohints_per_file - int((bounds.start - file_idx_sample) // (ipp_samps * n_ipp))
    job_cohints += num_cohints

    return job_tasks, job_cohints


def get_filepath(epoch_unix_us: int, sample_id_us: int) -> Path:
    """
    Generates a file path for a h5 file.

    Args:
        epoch_unix_us: Epoch in unix time of file in microseconds
        sample_id_us: sample time in microseconds

    Returns:
        a filepath based on the epoch and the current sample time
    """
    _dt = dt.datetime.fromtimestamp(epoch_unix_us * 1e-6, dt.timezone.utc)
    time_string = _dt.strftime("%Y-%m-%dT%H-00-00")
    return Path(time_string) / f"mf-{sample_id_us:08d}.h5"


def sample_interval_to_closest_ipp(sample_bounds: Bounds, ipp_samps: int) -> Bounds:

    start = sample_bounds.start
    if start % ipp_samps != 0:
        start = start - (start % ipp_samps)

    end = sample_bounds.end
    if end % ipp_samps != 0:
        end = end - (end % ipp_samps)

    return Bounds(start, end)
