#!/bin/bash
# network_generate.py
# Alessio Burrello <alessio.burrello@unibo.it>
#
# Copyright (C) 2019-2020 University of Bologna
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
sys.path.append('..')
from Parsers.DORY_node import DORY_node
from Parsers.Layer_node import Layer_node
import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import torch
import torch.nn.functional as f


def clip(x, bitwidth):
    low = 0
    high = 2**bitwidth - 1
    x[x > high] = high
    x[x < low] = low
    return x


def create_dory_node(name, op_type, layout, input_activation_bits, output_activation_bits):
    node = DORY_node()
    node.branch_out = 0
    node.branch_in = 0
    node.branch_last = 0
    node.branch_change = 0

    node.name = name
    node.op_type = op_type

    node.layout = layout

    # TODO check if bias_bits = constant_bits = input_activation_bits
    node.bias_bits = input_activation_bits
    node.constant_type = 'int'
    node.constant_bits = input_activation_bits
    node.constant_names = []

    node.input_activation_type = 'int'
    node.input_activation_bits = input_activation_bits

    node.output_activation_type = 'int'
    node.output_activation_bits = output_activation_bits

    node.weight_type = 'int'
    node.weight_bits = None

    # TODO check
    node.min = 0
    node.max = 2**output_activation_bits - 1

    # TODO
    node.input_indexes = []
    node.output_index = None
    node.number_of_input_nodes = None
    node.number_of_input_constants = None

    return node


def create_layer_node(kernel_shape, ch_in, ch_out, group, stride, pads,
                      input_dim, output_dim, w_bits, in_bits, out_bits):
    node = Layer_node()
    node.pads = pads
    node.input_dimensions = input_dim
    node.output_dimensions = output_dim
    node.input_channels = ch_in
    node.output_channels = ch_out
    node.kernel_shape = kernel_shape
    node.constant_names = []
    node.constant_type = 'int'
    node.constants_memory = None
    node.constant_bits = None

    node.name = 'Conv'
    if kernel_shape[0] == 1 and input_dim[0] == 1 and output_dim[0] == 1:
        node.name = 'Gemm'

    node.group = group
    if node.group > 1:
        # TODO: I would remove this -> node.output_channels = 1
        node.name += 'DW'

    node.strides = stride
    node.output_activation_bits = out_bits
    node.input_activation_bits = in_bits
    node.weight_bits = w_bits
    node.weight_type = 'int'
    node.weight_memory = None
    node.MACs = output_dim[0] * output_dim[1] * ch_out * kernel_shape[1] * kernel_shape[0] * ch_in
    return node


def compress(x, bitwidth):
    compressed = []
    n_elements_in_byte = 8 // bitwidth
    i_element_in_byte = 0
    for el in x:
        if i_element_in_byte == 0:
            compressed.append(el.item())
        else:
            compressed[-1] += el.item() << i_element_in_byte * bitwidth

        i_element_in_byte += 1
        if i_element_in_byte == n_elements_in_byte:
            i_element_in_byte = 0

    return np.asarray(compressed, dtype=np.uint8)


def calculate_shift(x, bitwidth):
    """
    Calculate shift

    This function calculates the shift in a way that it maximizes the number of values
    that are in between min and max after shifting. It looks only at positive values since
    all the negative ones are going to bi clipped to 0.
    Tries to shift the mean towards the middle of the possible values: [0, 2**bits - 1]
    """
    x = x.type(torch.float)
    mean = x[x > 0].mean().item()
    ratio = mean / (2 ** (bitwidth - 1))
    shift = round(np.log2(ratio))
    return shift


def batchnorm(x, scale, bias):
    return scale * x + bias


def calculate_batchnorm_params(x, output_bitwidth, constants_bitwidth):
    """
    Calculate batchnorm

    Calculate Batch-Normalization parameters scale and bias such that we maximize the number
    of values that fall into range [0, 2**bitwidth - 1].
    Shifts the mean towards the center of the range and changes the standard deviation so that
    most of the values fall into the range.
    """
    x = x.type(torch.float)

    desired_mean = (2 ** output_bitwidth - 1) / 2
    desired_std = 2 ** output_bitwidth / 2

    # Calculate mean and std for each output channel
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True)

    scale = desired_std / std
    scale = scale.round()
    scale = clip(scale, constants_bitwidth)
    scale[scale == 0] = 1

    bias = scale * (desired_mean - mean)
    bias = bias.round()
    bias = clip(bias, constants_bitwidth)

    return scale, bias


def create_input(node):
    size = (1, node.input_channels * node.group, node.input_dimensions[0], node.input_dimensions[1])
    bitwidth = node.input_activation_bits
    return torch.randint(low=0, high=2**bitwidth - 1, size=size)


def create_layer(i_layer, layer_node, dory_node, input=None):
    if input is None:
        x = create_input(layer_node)
    else:
        x = input

    x_to_compress = x.permute(0, 2, 3, 1).flatten()
    x_compressed = compress(x_to_compress, layer_node.input_activation_bits)
    x_save = np.concatenate((np.asarray([0]), x_compressed), axis=0)
    np.savetxt(os.path.join(args.network_dir, 'input.txt'), x_save, delimiter=',')

    w_low = -(2**(layer_node.weight_bits - 1))
    w_high = 2**(layer_node.weight_bits - 1)
    w_size = (layer_node.output_channels, layer_node.input_channels, layer_node.kernel_shape[0], layer_node.kernel_shape[1])
    w = torch.randint(low=w_low, high=w_high, size=w_size)
    w_name = 'conv1.weight'
    layer_node.constant_names.append(w_name)
    layer_node.__dict__[w_name] = w.permute(0, 2, 3, 1).numpy()

    y = f.conv2d(input=x, weight=w, stride=layer_node.strides, padding=layer_node.pads[0], groups=layer_node.group)

    if 'BN' in dory_node.op_type:
        k, l = calculate_batchnorm_params(y, dory_node.output_activation_bits, dory_node.constant_bits)
        dory_node.constant_names.append('k')
        dory_node.k = {'value': k.numpy(), 'layout': ''}
        dory_node.constant_names.append('l')
        dory_node.l = {'value': l.numpy(), 'layout': ''}
        y = batchnorm(y, k.type(torch.int64), l.type(torch.int64))

    dory_node.constant_names.append('outshift')
    dory_node.outshift = {
        'value': calculate_shift(y, dory_node.output_activation_bits),
        'layout': ''
    }
    y = y >> dory_node.outshift['value']
    y = clip(y, dory_node.output_activation_bits)

    y_save = y.permute(0, 2, 3, 1).flatten().numpy()
    y_save = np.concatenate((np.asarray([0]), y_save), axis=0)
    np.savetxt(os.path.join(args.network_dir, f'out_layer{i_layer}.txt'), y_save, delimiter=',')

    return y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-ks', '--kernel_shape', type=int, required=True, nargs=2, help='Kernel shape HxW')
    parser.add_argument('-ic', '--input_channels', type=int, required=True, help='Input channels')
    parser.add_argument('-oc', '--output_channels', type=int, required=True, help='Output channels')
    parser.add_argument('-id', '--input_dimensions', type=int, required=True, nargs=2, help='Input dimensions HxW')
    parser.add_argument('-od', '--output_dimensions', type=int, required=True, nargs=2, help='Output dimensions HxW')
    parser.add_argument('-g', '--groups', type=int, default=1, help='Groups. Default: 1')
    parser.add_argument('-p', '--padding', type=int, default=[0, 0, 0, 0], nargs=4, help='Padding [top, bottom, right, left]. Default: [0, 0, 0, 0]')
    parser.add_argument('-s', '--stride', type=int, default=1, help='Stride. Default: 1')
    parser.add_argument('-wb', '--weight_bitwidth', type=int, default=8, help='Weights bitwidth. Default: 8')
    parser.add_argument('-ib', '--input_bitwidth', type=int, default=8, help='Input bitwidth. Default: 8')
    parser.add_argument('-ob', '--output_bitwidth', type=int, default=8, help='Output bitwidth. Default: 8')
    parser.add_argument('-iab', '--intermediate_activation_bitwidth', type=int, default=32, help='Intermediate activation bitwidth. Default: 32')
    parser.add_argument('--network_dir', default="./examples/layer_test", help='directory of the onnx file of the network')
    parser.add_argument('--Bn_Relu_Bits', type=int, default=32, help='Number of bits for Relu/BN')
    parser.add_argument('-bn', '--batchnorm', default='Yes', help='Use Batch Normalization. Default: Yes')
    parser.add_argument('--perf_layer', default='Yes', help='Yes: MAC/cycles per layer. No: No perf per layer.')
    parser.add_argument('--verbose_level', default='Check_all+Perf_final', help="None: No_printf.\nPerf_final: only total performance\nCheck_all+Perf_final: all check + final performances \nLast+Perf_final: all check + final performances \nExtract the parameters from the onnx model")
    parser.add_argument('--backend', default='MCU', help='MCU or Occamy')
    parser.add_argument('--layer_number', type=int, default=1, help='Number of layer from excel.')
    parser.add_argument('--optional', default='mixed-sw', help='auto (based on layer precision, 8bits or mixed-sw), 8bit, mixed-hw, mixed-sw')
    args = parser.parse_args()

    os.makedirs(args.network_dir, exist_ok=True)

    torch.manual_seed(0)

    layer_node = create_layer_node(
        args.kernel_shape,
        args.input_channels,
        args.output_channels,
        args.groups,
        args.stride,
        args.padding,
        args.input_dimensions,
        args.output_dimensions,
        args.weight_bitwidth,
        args.input_bitwidth,
        args.intermediate_activation_bitwidth
    )

    dory_node_name = 'BNRelu' if args.batchnorm == 'Yes' else 'Relu'

    dory_node = create_dory_node(
        dory_node_name,
        dory_node_name,
        'CHW',
        args.intermediate_activation_bitwidth,
        args.output_bitwidth
    )

    create_layer(0, layer_node, dory_node)

    # TODO copy from network generator steps after frontend
