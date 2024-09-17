import pytest

from hardtarget.radars.eiscat import get_download_nodes


OLD_PRODUCT = {
    'day': '20220408',
    'mode': 'leo_bpark_2.1u_NO',
    'instrument': 'uhf'
}    

NEW_PRODUCT = {
    'day': '20240408',
    'mode': 'leo_bpark_2.1u_NO',
    'instrument': 'uhf'
}    


@pytest.mark.parametrize("product", [
    OLD_PRODUCT,
    NEW_PRODUCT
])


def test_download_nodes(product):
    day = product['day']
    mode = product['mode']
    nodes = get_download_nodes(day, mode)
    assert len(nodes) > 0

