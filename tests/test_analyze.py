import pytest
from pathlib import Path
from hardtarget.analysis import analyze_gmf
from hardtarget.config import load_gmf_params


class TestAnalyze:

    """
    Test analyze function
    - test that it uses the right part of the data
    - test that output format is correct
    """

    # mark test as not ready to always be run
    @pytest.mark.skip(reason="dependency on local file")
    def test_analyze(self):

        """
        - should probably make a mockup rdf file
        - for now, use one in local filesystem
        """

        # DRF 
        drf_path = Path.home() / Path("Data/hard_target/leo_bpark_2.1u_NO@uhf/drf/")

        # bounds
        # setting bounds so that it hits one specific drf file
        # rf@1618228601.000.h5
        bounds = [1618228601, 1618228602]

        # GMF Params
        config_path = Path.home() / Path("Dev/Git/hard_target/cfg/test.ini")
        gmf_params = load_gmf_params(config_path)
        gmf_params["start_time"] = bounds[0]
        gmf_params["end_time"] = bounds[1]

        # TASK
        task = {
            "rx": (str(drf_path), "uhf"),
            "tx": (str(drf_path), "uhf"),
            "gmf_params": gmf_params,
        }

        # preprocess
        ok = analyze_gmf.preprocess(task)
        assert ok

        # process
        ok, results = analyze_gmf.process(task)
        assert ok

        # expect result to be one file
        assert len(results["out"]) == 1
        expected = f"gmf-{bounds[0]}000000.h5"
        out = results["out"].get(expected, None)
        assert out is not None

        gmf_max = out["gmf"]
        # gmf_dc = out["gmf_dc"]
        # r = out["r"]
        # v = out["v"]
        # a = out["a"]
        # tx_pwr = out["tx_pwr"]
        i0 = out["i0"]

        # TODO: figure out how to check output
        print(gmf_max.shape)
        print(gmf_max.dtype)
        print(i0)
