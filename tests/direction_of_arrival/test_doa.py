import tempfile
from pathlib import Path

import numpy as np
import numpy.typing as npt
from radardef.radar_stations import Mu
from radardef.radar_stations.mu.experiments import mu_exp

from hardtarget.analyse import direction_of_arrival
from hardtarget.data_simulation.simulate_h5 import simulate_h5
from hardtarget.interferometry.types import DOACfgParams


class TestDOA:
    def test_music_grid_search(self):
        """Verify that the music grid search locates the correct peak"""

        # Define station and experiment
        station = Mu()
        exp_def = mu_exp

        # Define data to test, we simulate a object stuck at a specific location, for just one ipp
        target_location = np.array(
            [0.25467, 0.3216, 1.0]
        )  # np.array([np.random.rand(), np.random.rand(), 1])

        def trajectory_func(t: npt.NDArray) -> npt.NDArray:
            return target_location * 210e3

        with tempfile.TemporaryDirectory() as tmp_dir:
            simulation_path = Path(tmp_dir) / "sim_data"
            simulate_h5(
                output_dir=simulation_path,
                exp_params=exp_def,
                start_time=0,
                end_time=exp_def.t_ipp_usec,
                target_start_time=0,
                target_end_time=exp_def.t_ipp_usec,
                trajectory_function=trajectory_func,
                beam=station.beam,
                beam_params=station.beam_parameters,
            )

            cfg = DOACfgParams(
                n_ipp=1,
                min_range_gate=81,
                max_range_gate=166,
                elevation_limit=0,
                resolution=80,
                distributed_peaks=5,
            )
            result = direction_of_arrival(
                path=simulation_path,
                config=cfg,
                array_beam=station.beam,
                beam_params=station.beam_parameters,
            )["data"]

            out_data, _, _, _ = result[0]

            assert len(out_data.k_vec) == 1, "More ipps than estimated has been analysed"

            k_diff = np.abs(target_location - out_data.k_vec[0])
            mean_error = np.mean(k_diff)

            assert np.mean(k_diff) < 0.15, (
                f"To large estimation error between real data and estimation, error: {mean_error}"
            )
