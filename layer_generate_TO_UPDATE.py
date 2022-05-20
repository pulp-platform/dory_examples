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
import torch.nn as nn
import torch.nn.functional as F


def clip(x, bitwidth):
    low = 0
    high = 2**bitwidth - 1
    x[x > high] = high
    x[x < low] = low
    return x


def create_layer_node(
        kernel_shape,
        ch_in,
        ch_out,
        depthwise,
        stride,
        pads,
        input_dim,
        output_dim,
        W_BITS, IN_BITS, OUT_BITS):

    layer_node = Layer_node()
    layer_node.pads = pads
    layer_node.input_dimensions = input_dim
    layer_node.output_dimensions = output_dim
    layer_node.input_channels = ch_in
    layer_node.output_channels = ch_out # = node.input_channels if DW
    layer_node.kernel_shape = kernel_shape

    layer_node.name = 'Conv'
    if kernel_shape[0] == 1 and input_dim[0] == 1 and output_dim[0] == 1:
        layer_node.name = 'Gemm'

    if not depthwise:
        layer_node.group = 1
    else:
        layer_node.group = ch_out
        layer_node.output_channels = 1
        layer_node.name += 'DW'

    layer_node.strides = stride
    layer_node.output_activation_bits = OUT_BITS
    layer_node.input_activation_bits = IN_BITS
    layer_node.weight_bits = W_BITS
    layer_node.MACs = output_dim[0] * output_dim[1] * ch_out * kernel_shape[1] * kernel_shape[0] * ch_in
    return layer_node


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
    mean = x[x > 0].mean()
    ratio = mean / (2 ** (bitwidth - 1))
    shift = round(np.log2(ratio))
    return shift


def batchnorm(x, scale, bias):
    return scale * x + bias


def calculate_batchnorm_params(x, bitwidth):
    """
    Calculate batchnorm

    Calculate Batch-Normalization parameters scale and bias such that we maximize the number
    of values that fall into range [0, 2**bitwidth - 1].
    Shifts the mean towards the center of the range and changes the standard deviation so that
    most of the values fall into the range.
    """
    x = x.type(float)

    desired_mean = (2 ** bitwidth - 1) / 2
    desired_std = 2 ** bitwidth / 2

    # Calculate mean and std for each output channel
    mean = x.mean(dim=(-3, -2, -1), keepdim=True)
    std = x.std(dim=(-3, -2, -1), keepdim=True)

    scale = desired_std / std
    bias = scale * (desired_mean - mean)

    return scale.round(), bias.round()


def create_input(node):
    size = (1, node.input_channels * node.group, node.input_dimensions[0], node.input_dimensions[1])
    bitwidth = node.input_activation_bits
    return torch.randint(low=0, high=2**bitwidth - 1, size=size)


def main(nodes_list):
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--network_dir', default="./examples/layer_test", help='directory of the onnx file of the network')
    parser.add_argument('--Bn_Relu_Bits', type=int, default=32, help='Number of bits for Relu/BN')
    parser.add_argument('--batchnorm', action=argparse.BooleanOptionalAction, help='Enable Batch Normalization')
    parser.add_argument('--perf_layer', default='Yes', help='Yes: MAC/cycles per layer. No: No perf per layer.')
    parser.add_argument('--verbose_level', default='Check_all+Perf_final', help="None: No_printf.\nPerf_final: only total performance\nCheck_all+Perf_final: all check + final performances \nLast+Perf_final: all check + final performances \nExtract the parameters from the onnx model")
    parser.add_argument('--backend', default='MCU', help='MCU or Occamy')
    parser.add_argument('--layer_number', type=int, default=1, help='Number of layer from excel.')
    parser.add_argument('--optional', default='mixed-sw', help='auto (based on layer precision, 8bits or mixed-sw), 8bit, mixed-hw, mixed-sw')
    args = parser.parse_args()

    os.makedirs(args.network_dir)

    torch.manual_seed(0)

    nodes = nodes_list

    # INPUT CREATION
    x = create_input(nodes[0])

    x_to_compress = x.permute(0, 2, 3, 1).flatten()
    x_compressed = compress(x_to_compress, nodes[0].input_activation_bits)
    x_save = np.concatenate((np.asarray([0]), x_compressed), axis=0)
    np.savetxt(os.path.join(args.network_dir, 'input.txt'), x_save, delimiter=',')

    # GENERATION OF INTERMEDIATE DATA
    for i, node in enumerate(nodes):
        dory_node = DORY_node()

        w_low = -(2**(node.weight_bits - 1))
        w_high = 2**(node.weight_bits - 1)
        w_size = (node.output_channels, node.input_channels, node.kernel_shape[0], node.kernel_shape[1])
        w = torch.randint(low=w_low, high=w_high, size=w_size)
        node.weights = w.permute(0, 2, 3, 1).numpy()
        node.weights_raw = w.permute(0, 2, 3, 1).numpy()
        dory_node.constant_names.append('weights')
        dory_node.weights = w

        y = F.conv2d(input=x, weight=w, stride=node.strides, padding=node.pads[0], groups=node.group)

        if args.batchnorm:
            scale, bias = calculate_batchnorm_params(y, node.output_activation_bits)
            scale = clip(scale, node.output_activation_bits)
            scale[scale == 0] = 1
            bias = clip(bias, node.output_activation_bits)
            dory_node.constant_names.append('k')
            dory_node.k = scale
            dory_node.constant_names.append('lambda')
            dory_node.__dict__['lambda'] = bias
            y = batchnorm(y, scale, bias)

        node.outshift = calculate_shift(y, node.output_activation_bits)
        y = y << node.outshift
        y = clip(y, node.output_activation_bits)

        y_save = y.permute(0, 2, 3, 1).flatten().numpy()
        y_save = np.concatenate((np.asarray([0]), y_save), axis=0)
        np.savetxt(os.path.join(args.network_dir, f'out_layer{i}.txt'), y_save, delimiter=',')

        x = y

    # TODO copy from network generator steps after frontend


if __name__ == '__main__':
    ##############################################
    ###### INPUT PARAMETERS TO DEFINE ############
    ##############################################
    kernel_shape = [3, 3]
    ch_in = 64
    ch_out = 64
    DW = 0
    stride = 1
    pads = [1, 1, 1, 1]
    input_dim = [32, 32]
    output_dim = [32, 32]
    # kernel_shape = [1,1]
    # ch_in = 1024
    # ch_out = 64
    # DW = 0
    # stride = 1
    # pads = [0,0,0,0]
    # input_dim = [1,1]
    # output_dim = [1, 1]
    W_BITS = 8
    IN_BITS = 8
    OUT_BITS = 8

    ##############################################
    ##########DON'T TOUCH AFTER###################
    ##############################################

    list_nodes = []
    new_node = create_layer_node(
        kernel_shape,
        ch_in,
        ch_out,
        DW,
        stride,
        pads,
        input_dim,
        output_dim,
        W_BITS, IN_BITS, OUT_BITS)
    list_nodes.append(new_node) 

    main(list_nodes)
