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
import json
import os
import importlib
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import sys
sys.path.append('..')
from Parsers.DORY_node import DORY_node
from Parsers.Layer_node import Layer_node


def clip(x, bits):
    low = 0
    high = 2**bits - 1
    x[x > high] = high
    x[x < low] = low
    return x


def create_dory_node(name, op_type, layout, input_activation_bits, output_activation_bits, constant_bits):
    node = DORY_node()
    node.branch_out = 0
    node.branch_in = 0
    node.branch_last = 0
    node.branch_change = 0

    node.name = name
    node.op_type = op_type

    node.layout = layout

    node.bias_bits = 32

    # constant -> bn and relu
    node.constant_type = 'int'
    node.constant_bits = constant_bits
    node.constant_names = []

    node.input_activation_type = 'int'
    node.input_activation_bits = input_activation_bits

    node.output_activation_type = 'int'
    node.output_activation_bits = output_activation_bits

    node.weight_type = 'int'
    node.weight_bits = None

    node.min = 0
    node.max = 2**output_activation_bits - 1

    # Ids of previous nodes, node can have multiple input nodes
    node.number_of_input_nodes = 1
    node.input_indexes = ['1']  # layer_node is the input
    node.output_index = '2'
    # Constants: weights, bias, k, lambda
    node.number_of_input_constants = 4

    return node


def create_layer_node(kernel_shape, input_channels, output_channels, group, stride, pads,
                      input_dimensions, output_dimensions, weight_bits, input_activation_bits, output_activation_bits):
    node = Layer_node()
    node.pads = pads
    node.input_dimensions = input_dimensions
    node.output_dimensions = output_dimensions
    node.input_channels = input_channels
    node.output_channels = output_channels
    node.kernel_shape = kernel_shape
    node.constant_names = []
    node.constant_type = 'int'
    node.constants_memory = None
    node.constant_bits = None
    node.name = 'Convolution'
    node.op_type = 'Conv'
    node.group = group
    node.strides = stride
    node.output_activation_type = 'int'
    node.output_activation_bits = output_activation_bits
    node.input_activation_type = 'int'
    node.input_activation_bits = input_activation_bits
    node.weight_type = 'int'
    node.weight_bits = weight_bits
    node.weight_memory = None
    node.MACs = output_dimensions[0] * output_dimensions[1] * output_channels * kernel_shape[1] * kernel_shape[0] * input_channels

    # Ids of previous nodes, node can have multiple input nodes
    node.number_of_input_nodes = 1
    node.input_indexes = ['0']  # '0' is the network input
    node.output_index = '1'
    # Constants: weights
    node.number_of_input_constants = 1
    return node


def compress(x, bits):
    compressed = []
    n_elements_in_byte = 8 // bits
    i_element_in_byte = 0
    for el in x:
        if i_element_in_byte == 0:
            compressed.append(el.item())
        else:
            compressed[-1] += el.item() << i_element_in_byte * bits

        i_element_in_byte += 1
        if i_element_in_byte == n_elements_in_byte:
            i_element_in_byte = 0

    return np.asarray(compressed, dtype=np.uint8)


def calculate_shift(x, bits):
    """
    Calculate shift

    This function calculates the shift in a way that it maximizes the number of values
    that are in between min and max after shifting. It looks only at positive values since
    all the negative ones are going to bi clipped to 0.
    Tries to shift the mean of positive values towards the middle of the range [0, 2**bits - 1]
    """
    x = x.type(torch.float)
    mean = x[x > 0].mean().item()
    ratio = mean / (2 ** (bits - 1))
    shift = round(np.log2(ratio))
    return shift


def batchnorm(x, scale, bias):
    return scale * x + bias


def calculate_batchnorm_params(x, output_bits, constant_bits):
    """
    Calculate batchnorm

    Calculate Batch-Normalization parameters scale and bias such that we maximize the number
    of values that fall into range [0, 2**output_bits - 1].
    Shifts the mean towards the center of the range and changes the standard deviation so that
    most of the values fall into the range.
    """
    x = x.type(torch.float)

    desired_mean = (2 ** output_bits - 1) / 2
    desired_std = 2 ** output_bits / 2

    # Calculate mean and std for each output channel
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True)

    scale = desired_std / std
    scale = scale.round()
    scale = clip(scale, constant_bits)
    scale[scale == 0] = 1

    bias = scale * (desired_mean - mean)
    bias = bias.round()
    bias = clip(bias, constant_bits)

    return scale, bias


def create_input(node):
    size = (1, node.input_channels * node.group, node.input_dimensions[0], node.input_dimensions[1])
    bits = node.input_activation_bits
    return torch.randint(low=0, high=2**bits - 1, size=size)


def create_layer(i_layer, layer_node, dory_node, network_dir, input=None):
    if input is None:
        x = create_input(layer_node)
    else:
        x = input

    x_to_compress = x.permute(0, 2, 3, 1).flatten()
    x_compressed = compress(x_to_compress, layer_node.input_activation_bits)
    np.savetxt(os.path.join(network_dir, 'input.txt'), x_compressed, delimiter=',')

    w_low = -(2**(layer_node.weight_bits - 1))
    w_high = 2**(layer_node.weight_bits - 1)
    w_size = (layer_node.output_channels, layer_node.input_channels, layer_node.kernel_shape[0], layer_node.kernel_shape[1])
    w = torch.randint(low=w_low, high=w_high, size=w_size)
    layer_node.constant_names.append('weights')
    layer_node.weights = {
        'value': w.permute(0, 2, 3, 1).numpy(),
        'layout': 'CoutCinK'
    }

    y = F.conv2d(input=x, weight=w, stride=layer_node.strides, padding=layer_node.pads[0], groups=layer_node.group)

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
    np.savetxt(os.path.join(network_dir, f'out_layer{i_layer}.txt'), y_save, delimiter=',')

    return y


def create_graph(params, network_dir):
    layer_node = create_layer_node(
        params['kernel_shape'],
        params['input_channels'],
        params['output_channels'],
        params['group'],
        params['stride'],
        params['padding'],
        params['input_dimensions'],
        params['output_dimensions'],
        params['weight_bits'],
        params['input_bits'],
        params['intermediate_activation_bits']
    )

    dory_node_name = 'BNRelu' if params['batchnorm'] else 'Relu'

    dory_node = create_dory_node(
        dory_node_name,
        dory_node_name,
        'CHW',
        params['intermediate_activation_bits'],
        params['output_bits'],
        params['BNRelu_bits']
    )

    with torch.no_grad():
        create_layer(0, layer_node, dory_node, network_dir)

    return [layer_node, dory_node]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hardware_target', type=str, choices=["GAP8", "nnx", "Occamy", "Diana"], help='Hardware platform for which the code is optimized')
    parser.add_argument('--config_file', default='config_files/config_single_layer.json', type=str, help='Path to the JSON file that specifies the ONNX file of the network and other information. Default: config_files/config_single_layer.json')
    parser.add_argument('--app_dir', default='./application', help='Path to the generated application. Default: ./application')
    parser.add_argument('--perf_layer', default='Yes', help='Yes: MAC/cycles per layer. No: No perf per layer.')
    parser.add_argument('--verbose_level', default='Check_all+Perf_final', help="None: No_printf.\nPerf_final: only total performance\nCheck_all+Perf_final: all check + final performances \nLast+Perf_final: all check + final performances \nExtract the parameters from the onnx model")
    parser.add_argument('--backend', default='MCU', help='MCU or Occamy')
    parser.add_argument('--optional', default='mixed-sw', help='auto (based on layer precision, 8bits or mixed-sw), 8bit, mixed-hw, mixed-sw')
    args = parser.parse_args()

    json_configuration_file_root = os.path.dirname(args.config_file)
    with open(args.config_file, 'r') as f:
        json_configuration_file = json.load(f)

    network_dir = os.path.join(json_configuration_file_root, os.path.dirname(json_configuration_file['onnx_file']))
    os.makedirs(network_dir, exist_ok=True)

    torch.manual_seed(0)

    DORY_Graph = create_graph(json_configuration_file, network_dir)

    # Including and running the transformation from DORY IR to DORY HW IR
    onnx_manager = importlib.import_module(f'Hardware-targets.{args.hardware_target}.HW_Parser')
    DORY_to_DORY_HW = onnx_manager.onnx_manager
    DORY_Graph = DORY_to_DORY_HW(DORY_Graph, json_configuration_file, json_configuration_file_root).full_graph_parsing()

    # Deployment of the model on the target architecture
    onnx_manager = importlib.import_module(f'Hardware-targets.{args.hardware_target}.C_Parser')
    DORY_HW_to_C = onnx_manager.C_Parser
    DORY_Graph = DORY_HW_to_C(DORY_Graph, json_configuration_file, json_configuration_file_root,
                              args.verbose_level, args.perf_layer, args.optional, args.app_dir).full_graph_parsing()
