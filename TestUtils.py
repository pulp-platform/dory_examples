import re
import shutil
import subprocess
from math import ceil
from layer_generate import layer_generate


default_conf = {
    "layer_type": "Convolution",
    "operation_type": "Conv",
    "BNRelu_bits": 8,
    "onnx_file": "examples/layer_test/nonexistent.onnx",
    "code reserved space": 152000,
    "input_channels": 20,
    "output_channels": 20,
    "kernel_shape": [1, 1],
    "input_dimensions": [10, 10],
    "input_bits": 8,
    "input_type": "uint",
    "output_bits": 8,
    "output_type": "uint",
    "group": 1,
    "padding": [0, 0, 0, 0],
    "stride": [1, 1],
    "weight_bits": 8,
    "intermediate_bits": 32,
    "use_relu": True,
    "batchnorm": True
}


def set_dims(conf, dims):
    hi, wi, ki, ko, ks, g, p, s = dims
    conf['input_channels'] = ki
    conf['output_channels'] = ko
    conf['input_dimensions'] = [hi, wi]
    conf['kernel_shape'] = [ks, ks]
    conf['group'] = g
    conf['padding'] = [p, p, p, p]
    conf['stride'] = [s, s]
    return conf

def dims_size(dims):
    hi, wi, ki, ko, ks, g, p, s = dims
    ho = (hi - ks + 2*p) / s + 1
    wo = (wi - ks + 2*p) / s + 1
    input_size = hi * wi * ki
    output_size = ho * wo * ko
    if g == 1:
        weights_size = ko * ceil(ki / 16) * 8 * ks * ks * 2  # 8 -> qw, 2 -> TP_IN / 8
    else:
        weights_size = ceil(ki / 16) * 8 * ks * ks * 2  # 8 -> qw, 2 -> TP_IN / 8
    return input_size + weights_size + output_size


def filter_dims_smaller_then_mem(dimslist, mem):
    return [dims for dims in dimslist if dims_size(dims) <= mem]


def execute_layer(name, env, timeout=30, app_cflags=None):
    cmd = ['make', '-C', 'application', 'clean', 'all', 'run']
    if app_cflags is not None and isinstance(app_cflags, list):
        env['APP_CFLAGS'] = ' '.join(app_cflags)

    def exception_output(exception, name, msg):
        stdout = exception.stdout
        try:
            stdout = stdout.decode('ascii')
        except (UnicodeDecodeError, AttributeError):
            pass

        stderr = exception.stderr
        try:
            stderr = stderr.decode('ascii')
        except (UnicodeDecodeError, AttributeError):
            pass

        h_line = '\n' + '-' * 100 + '\n'
        return f"Layer {name}: {msg}\n" \
               f"  - command: {' '.join(cmd)}\n" + \
               h_line +                            \
               f"\nCaptured stdout:\n{stdout}\n" + \
               h_line +                            \
               f"\nCaptured stderr:\n{stderr}\n" + \
               h_line

    output = None
    status = None

    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout, env=env)
        output = proc.stdout
        try:
            output = output.decode('ascii')
        except (UnicodeDecodeError, AttributeError):
            pass
        status = True
    except subprocess.CalledProcessError as e:
        output = exception_output(e, name, f"build failure with exit status {e.returncode}")
        status = False
    except subprocess.TimeoutExpired as e:
        output = exception_output(e, name, f"timeout after {timeout}s")
        status = False

    return (status, output)


def layer_generate_test(conf, target):
    return layer_generate(conf, '.', 'examples/layer_test', target, verbose_level='Check_all+Perf_final')


regex_perf = re.compile(r'Final performance - MACs: (\d*), Cycles: (\d*), MAC/cycle: ([\d.]*)')
regex_perf_components = re.compile(r'Measured time - preamble: (\d*), first conf: (\d*), first dma: (\d*), nnx: (\d*), postamble: (\d*)')
regex_checksum = re.compile(r'Checking final output: (.*)')
