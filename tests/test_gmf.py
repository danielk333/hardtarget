import pytest
import numpy as np
from hardtarget.gmf import GMF_LIBS


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: mark test as cuda dependent")


class TestGMF:

    """
    It is possible that CUDA support is compilet, yet still non-functional.
    """

    @pytest.mark.parametrize("gmf_key", ["c", "numpy"])
    def test_gmf(self, gmf_key):
        self.run_gmf(gmf_key)

    @pytest.mark.cuda
    def test_gmf_cuda(self):
        self.run_gmf("cuda")

    def run_gmf(self, gmf_key):
        """Run the basic gmf function."""

        gmf = GMF_LIBS[gmf_key]

        # Expect output from this test
        expected = {"ri": 500, "gmf_vec": 1e04, "v_vec": 0.0, "a_vec": 0.0}

        z_tx = np.zeros(10000, dtype=np.complex64)
        z_rx = np.zeros(12000, dtype=np.complex64)

        for i in range(10):
            z_tx[(i * 1000): (i * 1000 + 20)] = 1.0
            z_rx[(i * 1000 + 500): (i * 1000 + (500 + 20))] = 0.5  # simulated "echo"

        dec = 10
        acc_phasors = np.zeros([20, 1000], dtype=np.complex64)
        acc_phasors[0, :] = 1.0
        rgs = np.arange(1000, dtype=np.float32)

        n_r = len(rgs)
        gmf_vec = np.zeros(n_r, dtype=np.float32)
        gmf_dc_vec = np.zeros(n_r, dtype=np.float32)
        v_vec = np.zeros(n_r, dtype=np.float32)
        a_vec = np.zeros(n_r, dtype=np.float32)

        # for i in range(20):
        gmf(z_tx, z_rx, acc_phasors, rgs, dec, gmf_vec, gmf_dc_vec, v_vec, a_vec)

        ri = np.argmax(gmf_vec)

        # Check that the output is as we expect
        assert expected["ri"] == ri
        assert expected["gmf_vec"] == gmf_vec[ri]
        assert expected["v_vec"] == v_vec[ri]
        assert expected["a_vec"] == a_vec[ri]
