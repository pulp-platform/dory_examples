import pytest
from TestUtils import *
from dims import *


def dimslist():
    l = []
    #l += dimslist_test_stride_2
    l += dimslist_direct_3x3
    #l += dimslist_direct_1x1
    #l += dimslist_direct_dw
    # l += perfect_3x3_nol3_dims
    # l += imperfect_3x3_nol3_dims
    # l += perfect_1x1_nol3_dims
    # l += imperfect_1x1_nol3_dims
    # l += MobileNetV1_dimslist(128)
    return l


# Check conftest.py for information about env and perf arguments
@pytest.mark.parametrize('dims', dimslist())
def test_single(dims, capsys, target, env, perf):
    conf = set_dims(default_conf, dims)
    layer_generate_test(conf, target)
    passed, output = execute_layer(str(dims), env, timeout=120)
    assert passed, output
    match_checksum = regex_checksum.search(output)

    preamble = f'Layer {dims}:'

    if perf:
        match_perf = regex_perf.search(output)
        with capsys.disabled():
            print(preamble + match_perf.group(0))

    assert match_checksum is not None, f'{preamble} regex not found\nOutput:\n{output}'
    assert match_checksum.group(1) == 'Checksum OK', f'{preamble} {match_checksum.group(1)}'
