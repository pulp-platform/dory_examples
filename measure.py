import csv
import os
import sys
import datetime
from tqdm import tqdm
import argparse

from dims import *
from TestUtils import *
from local_env import local_env, targets

csv_header_key = ['mode', 'Hi', 'Wi', 'Ki', 'Ko', 'Kernel Shape', 'Groups', 'Padding', 'Stride']
csv_header = csv_header_key + ['Operations', 'Latency', 'Performance']
csv_header_components = csv_header_key + ['Preamble', 'First configuration', 'First DMA', 'Execution', 'Postamble']
current_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M")
measurements_dir = 'measurements'


def measure_perf(name, dimslist, target, csv_filename, component, dma_confs, defines):
    path = os.path.join(measurements_dir, target, name)
    logpath = os.path.join(path, 'log')
    os.makedirs(logpath, exist_ok=True)
    csv_filepath = os.path.join(path, csv_filename)
    with open(csv_filepath, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_header_components if component else csv_header)

    app_cflags = ['-D' + define for define in defines]

    modes = ['original']
    if dma_confs:
        modes += ['no_dma_sync_on_border_tile', 'no_dma_sync', 'no_dma']

    for dims in tqdm(dimslist):
        conf = set_dims(default_conf, dims)

        # Redirect all prints to devnull
        tmp_stdout = sys.stdout
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull
            layer_generate_test(conf, target)
        sys.stdout = tmp_stdout

        for mode in modes:
            if component:
                app_cflags.append('-DMEASURE_LAYER_COMPONENT_PERF')
            if mode != 'original':
                app_cflags.append('-D' + mode.upper())

            passed, output = execute_layer(f'{dims}', local_env(target), timeout=240, app_cflags=app_cflags)
            if not passed:
                logfile = f'{current_time}-{dims[0]}_{dims[1]}_{dims[2]}_{dims[3]}_{dims[4]}_{dims[5]}_{dims[6]}' \
                          f'_{dims[7]}-{mode}{"-comp" if component else ""}.log'
                print(f'Layer {dims} failed execution. Writing output to {logfile}')
                with open(os.path.join(logpath, logfile), 'w') as log:
                    log.write(output)
                continue
            regex = regex_perf_components if component else regex_perf
            match = regex.search(output)
            match_pass = regex_checksum.search(output)
            if match is None:
                print(f'ERROR: Didn\'t match the perf regular expression.\n'
                      f'Received output:\n{output}')
                continue
            row = [
                match_pass is not None and match_pass.group(1) == 'Checksum OK',
                mode,
                conf['input_dimensions'][0],
                conf['input_dimensions'][1],
                conf['input_channels'],
                conf['output_channels'],
                conf['kernel_shape'][0],
                conf['group'],
                conf['padding'][0],
                conf['stride'][0],
                match.group(1),
                match.group(2),
                match.group(3)
            ]

            if component:
                row += [match.group(4), match.group(5)]

            with open(csv_filepath, 'a') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('target', choices=targets)
    parser.add_argument('-o', '--output', type=str,
                        help='CSV output file name. Default=Current time')
    parser.add_argument('-g', '--group', type=str)
    parser.add_argument('--measure-dma-configurations', dest='dma_confs', action='store_true', default=False)
    parser.add_argument('--measure-components', dest='component', action='store_true', default=False)
    parser.add_argument('--defines', '-d', nargs='+', default=[])
    args = parser.parse_args()

    csv_filename_default = f'{current_time}{"-comp" if args.component else ""}.csv'

    output = args.output if args.output is not None else csv_filename_default

    dimslist = []
    dimslist += MobileNetV1_dimslist(128)
    #dimslist += ResNetTinyML_dimslist()
    dimslist = [dim for dim in dimslist if dim[-1] == 2]
    group = args.group if args.group is not None else "mnV1-128"

    print(f"Measuring {group}:")

    measure_perf(group, dimslist, args.target, output, args.component, args.dma_confs, args.defines)
