import pytest
import numpy as np
from hardtarget.gmf import GMF_LIBS, Impl


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: mark test as cuda dependent")


class TestGMF:

    """
    It is possible that CUDA support is compilet, yet still non-functional.
    """

    @pytest.mark.parametrize("gmf_impl", [Impl.numpy, Impl.c])
    def test_gmf(self, gmf_impl):
        self.run_gmf(gmf_impl)

    @pytest.mark.cuda
    def test_gmf_cuda(self):
        self.run_gmf(Impl.cuda)

    def run_gmf(self, gmf_impl):
        """Run the basic gmf function."""

        raise NotImplementedError()

        """
        Should ideally be able to test the different implementations of the GMF function

        ISSUE 1
        Functions have a new signature now

        regular ones must be called with
        (z_tx, z_rx, gmf_variables, gmf_params)
        whereas optimized
        (z_tx, z_ipp, gmf_params, gmf_start)

        ISSUE 2
        implementations of GMF functions depend on gmf_params,
        gmf_params = {"DER": {}, "PRO": {}}
        This way of organizing parameters is not motivated in this context.

        ISSUE 3
        Implementation of GMF functions use different subsets of parameters from gmf_params

        ISSUE 4
        gmf_variables are defined in the analyis.util. If they are to be part of the
        gmf signature, they should be defined within the gmf module.

        ISSUE 5
        gmf_variables (as defined in analysis.util) include one attribute tx_pow,
        which is not needed by gmf functions.

        """

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
        gmf_func(z_tx, z_rx, acc_phasors, rgs, dec, gmf_vec, gmf_dc_vec, v_vec, a_vec)

        ri = np.argmax(gmf_vec)

        # Check that the output is as we expect
        assert expected["ri"] == ri
        assert expected["gmf_vec"] == gmf_vec[ri]
        assert expected["v_vec"] == v_vec[ri]
        assert expected["a_vec"] == a_vec[ri]
