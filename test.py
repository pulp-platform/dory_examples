import pytest
from TestUtils import *
from dims import *


def dimslist():
    l = []
    l += dimslist_direct
    # l += perfect_3x3_nol3_dims
    # l += imperfect_3x3_nol3_dims
    # l += perfect_1x1_nol3_dims
    # l += imperfect_1x1_nol3_dims
    return l


# Check conftest.py for information about env and perf arguments
@pytest.mark.parametrize('dims', dimslist())
def test_single(dims, capsys, target, env, perf):
    conf = set_dims(default_conf, dims)
    layer_generate_test(conf, target)
    passed, output = execute_layer(env, timeout=120)
    assert passed, output
    match_checksum = regex_checksum.search(output)

    preamble = f'Layer {dims}:'

    if perf:
        match_perf = regex_perf.search(output)
        with capsys.disabled():
            print(preamble + match_perf.group(0))

    assert match_checksum is not None, f'{preamble} regex not found.'
    assert match_checksum.group(1) == 'Checksum OK', f'{preamble} {match_checksum.group(1)}'
