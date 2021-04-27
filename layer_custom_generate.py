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

import sys
sys.path.append('../')
from ONNX_management import ONNX_management as onnx_m
from ONNX_management import node_element as node
from Model_deployment import Model_deployment as model_deploy
import os
import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import logging

def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--l1_buffer_size', type=int, default = 38000, help = 'L1 buffer size. IT DOES NOT INCLUDE SPACE FOR STACKS.')
    parser.add_argument('--l2_buffer_size', type=int, default = 380000, help = 'L2 buffer size.')
    parser.add_argument('--master_stack', type=int, default = 4096, help = 'Cluster Core 0 stack')
    parser.add_argument('--slave_stack', type=int, default = 3072, help = 'Cluster Core 1-7 stack')
    parser.add_argument('--Bn_Relu_Bits', type=int, default = 32, help = 'Number of bits for Relu/BN')
    parser.add_argument('--perf_layer', default = 'Yes', help = 'Yes: MAC/cycles per layer. No: No perf per layer.')
    parser.add_argument('--verbose_level', default = 'Perf_final', help = "None: No_printf.\nPerf_final: only total performance\nCheck_all+Perf_final: all check + final performances \nLast+Perf_final: all check + final performances \nExtract the parameters from the onnx model")
    parser.add_argument('--chip', default = 'GAP8v3', help = 'GAP8v2 for fixing DMA issue. GAP8v3 otherise')
    parser.add_argument('--sdk', default = 'pulp_sdk', help = 'gap_sdk or pulp_sdk')
    parser.add_argument('--dma_parallelization', default = '8-cores', help = '8-cores or 1-core')
    parser.add_argument('--fc_frequency', default = 100000000, help = 'frequency of fabric controller')
    parser.add_argument('--cl_frequency', default = 100000000, help = 'frequency of cluster')
    parser.add_argument('--optional', default = 'mixed-sw', help = '8bit, mixed-sw, mixed-hw, 1D_Conv') # change here to change backend kernels
    args = parser.parse_args()

    new_node = node()
    new_node.input_index = 0 # don't touch
    new_node.output_index = 1 # don't touch
    new_node.padding_top    = 1
    new_node.padding_left   = 1
    new_node.padding_bottom = 1
    new_node.padding_right  = 1
    new_node.input_h = 16 
    new_node.input_w = 16
    new_node.output_h = new_node.input_h
    new_node.output_w = new_node.input_w
    new_node.input_channels = 32
    new_node.output_channels = 64 # = new_node.input_channels if DW
    new_node.outshift = 3 # don't touch
    new_node.filter_size_h = 3
    new_node.filter_size_w = 3
    new_node.groups = 1 # put new_node.input_channels if DW
    new_node.stride = 1 # don't touch
    new_node.MACs = new_node.output_h * new_node.output_w * new_node.output_channels * new_node.filter_size_h * new_node.filter_size_w * new_node.input_channels
    new_node.name = 'Conv'
    new_node.weights = np.random.randint(low = -127, high = 128, size = (new_node.input_channels, new_node.filter_size_h, new_node.filter_size_w, new_node.output_channels))
    precision_dict_act_out     = [2,8] # change first number to change BitOut of the layer
    precision_dict_weights = [2,8] # change first number to change BitW of the laye
    BitIn = 2 # BitIn of the layer
    final_node = node()
    final_node.input_index = 1
    final_node.output_index = 2
    final_node.padding_top    = 0
    final_node.padding_left   = 0
    final_node.padding_bottom = 0
    final_node.padding_right  = 0
    final_node.input_h = 1
    final_node.input_w = 1
    final_node.output_h = final_node.input_h
    final_node.output_w = final_node.input_w
    final_node.input_channels = 32
    final_node.output_channels = 16
    final_node.filter_size_h = 1
    final_node.filter_size_w = 1
    final_node.groups = 1
    final_node.stride = 1
    final_node.MACs = final_node.output_h * final_node.output_w * final_node.output_channels * final_node.filter_size_h * final_node.filter_size_w * final_node.input_channels
    final_node.name = 'Gemm'
    final_node.weights = np.random.randint(low = -127, high = 128, size = (final_node.input_channels, final_node.filter_size_h, final_node.filter_size_w, final_node.output_channels))
    PULP_Nodes_Graph = []
    PULP_Nodes_Graph.append(new_node)
    PULP_Nodes_Graph.append(final_node)
    model_deploy('GAP8', args.chip).print_model_network(PULP_Nodes_Graph,
                            100,
                            './examples/8-bits-2D/MobilenetV1/',
                            100,
                            args.verbose_level,
                            args.perf_layer,
                            args.l1_buffer_size,
                            args.master_stack,
                            args.slave_stack,
                            args.l2_buffer_size,
                            args.fc_frequency,
                            args.cl_frequency,
                            2, 2, BitIn, args.Bn_Relu_Bits, 
                            args.sdk,
                            args.dma_parallelization,args.
                            optional,
                            precision_dict_act_out,
                            precision_dict_weights)
if __name__ == '__main__':
    main()
