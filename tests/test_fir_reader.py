from pathlib import Path

import numpy as np
import pytest

from hardtarget.data_simulation.receiver_chain import get_impresp, read_fir

FIR_CONTENT = """\
FIRPAR_VS 0.1
% representative b414d15_gaus.fir parameters
H_STAGES 5
H_DRATE 4
F_TAPS 2
F_ESYM 1
F_DRATE 2
TAP 0 0x29F17 % trailing comments are allowed
TAP 1 0x53E2D
"""


@pytest.fixture
def fir_file(tmp_path: Path) -> Path:
    path = tmp_path / "b414d15_gaus.fir"
    path.write_text(FIR_CONTENT)
    return path


def test_read_fir_hex_taps(fir_file: Path) -> None:
    fir, f_dec, h_dec, h_order = read_fir(fir_file)
    np.testing.assert_array_equal(fir, [0x29F17, 0x53E2D, 0x29F17])
    assert (f_dec, h_dec, h_order) == (3, 5, 5)


def test_get_impresp(fir_file: Path) -> None:
    _, _, taps, decimation = get_impresp(fir_file, 0.05)
    assert decimation == 15
    assert taps.size == 31
    assert taps.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(taps, taps[::-1])
