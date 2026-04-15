import datetime as dt
import tempfile
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from radardef import Mu
from radardef.radar_def import RadarDef
from radardef.radar_stations.mu.experiments import mu_exp
from spacecoords import interpolation

from hardtarget import analyse
from hardtarget.constants import AnalysisMethod
from hardtarget.data_simulation import simulate_h5
from hardtarget.utils.global_mpi import get_mpi


@pytest.fixture(scope="module", autouse=False)
def get_test_data():

    # Simuate MU data
    station = Mu()
    experiment = mu_exp
    # Temporary directory
    tmp_dir = tempfile.TemporaryDirectory()
    # Data directory
    data_path = Path(tmp_dir.name) / "data"

    # Define trajectory of an object
    def trajectory_func(t: npt.NDArray) -> npt.NDArray:
        # simple trajectory moving the object from one end of the beam to the next
        path = np.array([[-1, 0.5, 1], [0, 0, 0.95], [1, -0.5, 0.90]]) * 220e3
        t_samps = np.linspace(0, 1.0, path.shape[0])
        fun = interpolation.Linear(states=path.T, t=t_samps)
        return fun.get_state(t)

    measurement_start = dt.datetime.now()
    measurement_end = measurement_start + dt.timedelta(seconds=1)  # + dt.timedelta(seconds=4)
    target_start_time_us = 100000  # 1500000
    target_end_time_us = 200000  # 2000000

    # Simulate data and write to file
    simulate_h5(
        output_dir=data_path,
        exp_params=experiment,
        start_time=measurement_start,
        end_time=measurement_end,
        target_start_time=target_start_time_us,
        target_end_time=target_end_time_us,
        target_relative_time=True,
        trajectory_function=trajectory_func,
        noise_sigma=0.1,
        beam=station.beam,
        beam_params=station.beam_parameters,
    )

    cfg = {
        "n_ipp": 1,
        "ipp_offset": 0,
        "min_range_gate": 81,
        "max_range_gate": 140,
        "range_gate_step": 1,
        "num_cohints_per_file": 50,
        "tx_amp_limit": 0.2,
        "min_acceleration": 0,
        "max_acceleration": 0,
        "range_gate_sub_resolution": 10,
        "frequency_decimation": 1,
        "acceleration_steps": 1,
        "doppler_freq_min": -30000,
        "doppler_freq_max": 5000,
        "doppler_freq_step": 1000,
        "elevation_limit": 0,
        "resolution": 150,
    }

    # Run test
    yield data_path, experiment, cfg
    # Teardown
    tmp_dir.cleanup()


@pytest.mark.parametrize(
    "method",
    [
        AnalysisMethod.target_estimation,
        AnalysisMethod.echo_search,
    ],
)
@pytest.mark.parallel(nprocs=4)
def test_parallel_analysis(method, get_test_data):
    comm = get_mpi()
    data_path, exp, cfg = get_test_data

    res = analyse(
        data=data_path,
        config=cfg,
        method=method,
        output=data_path.parent / "analysis",
        comm=comm,
    )

    full_res = comm.gather(res, root=0)

    if comm.rank == 0:
        assert full_res is not None, "No results from parallel analysis"

        data_loader = RadarDef().load_data(path=data_path, experiment=exp)
        assert data_loader is not None, "Not possible to load data"
        start_samp, end_samp = data_loader.bounds(exp.rx_channels[0])
        n_samps = end_samp - start_samp

        n_cohints = n_samps // exp.ipp_samps * cfg["n_ipp"]
        n_files = -(n_cohints // -cfg["num_cohints_per_file"])

        generated_files = [file for result in full_res for file in result["files"]]

        assert n_files == len(generated_files), (
            f"Mismatch in amount of file generated, expected: {n_files}, generated: {len(generated_files)})"
        )
        assert len(generated_files) == len(set(generated_files)), (
            "Duplicates found in the files, no file should have the same name as the other"
        )
