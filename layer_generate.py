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
from PULP_node import node_element as node
from Model_deployment_MCU import Model_deployment_MCU as model_deploy
import os
import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import logging
import torch
import torch.nn as nn

def clip8(conv, bits, shift):
    conv = conv.astype(int)>>shift;
    conv[conv >= +(2**(bits) - 1)] = +(2**(bits) - 1)
    conv[conv <= 0] = 0
    out = np.uint8(conv)
    return out

def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--network_dir', default = "./examples/layer_test/", help = 'directory of the onnx file of the network')
    parser.add_argument('--l1_buffer_size', type=int, default = 38000, help = 'L1 buffer size. IT DOES NOT INCLUDE SPACE FOR STACKS.')
    parser.add_argument('--l2_buffer_size', type=int, default = 380000, help = 'L2 buffer size.')
    parser.add_argument('--master_stack', type=int, default = 4000, help = 'Cluster Core 0 stack')
    parser.add_argument('--slave_stack', type=int, default = 3000, help = 'Cluster Core 1-7 stack')
    parser.add_argument('--Bn_Relu_Bits', type=int, default = 32, help = 'Number of bits for Relu/BN')
    parser.add_argument('--perf_layer', default = 'Yes', help = 'Yes: MAC/cycles per layer. No: No perf per layer.')
    parser.add_argument('--verbose_level', default = 'Check_all+Perf_final', help = "None: No_printf.\nPerf_final: only total performance\nCheck_all+Perf_final: all check + final performances \nLast+Perf_final: all check + final performances \nExtract the parameters from the onnx model")
    parser.add_argument('--chip', default = 'GAP8v3', help = 'GAP8v2 for fixing DMA issue. GAP8v3 otherise')
    parser.add_argument('--sdk', default = 'gap_sdk', help = 'gap_sdk or pulp_sdk')
    parser.add_argument('--dma_parallelization', default = '8-cores', help = '8-cores or 1-core')
    parser.add_argument('--fc_frequency', default = 100000000, help = 'frequency of fabric controller')
    parser.add_argument('--cl_frequency', default = 100000000, help = 'frequency of cluster')
    parser.add_argument('--frontend', default = 'Nemo', help = 'Nemo or Quantlab')
    parser.add_argument('--backend', default = 'MCU', help = 'MCU or Occamy')
    parser.add_argument('--number_of_clusters', type=int, default = 1, help = 'Number of clusters in the target architecture.')
    args = parser.parse_args()
    new_node = node()
    ##############################################
    ###### INPUT PARAMETERS TO DEFINE ############
    ##############################################
    kernel_shape = [1,1]
    ch_in = 256
    ch_out = 32
    DW = 0
    stride = 1
    pads = [0,0,0,0]
    input_dim = [6,36]
    output_dim = [6, 36]
    W_BITS = 8; IN_BITS = 8; OUT_BITS = 8;
    ##############################################
    ##########DON'T TOUCH AFTER###################
    ##############################################
    new_node.input_index = 0 # don't touch
    new_node.output_index = 1 # don't touch
    new_node.pads    = pads
    new_node.input_dim = input_dim
    new_node.output_dim = output_dim
    new_node.ch_in = ch_in
    new_node.ch_out = ch_out # = new_node.input_channels if DW
    new_node.kernel_shape = kernel_shape
    new_node.name = 'Conv'
    if DW == 0:
        new_node.group = 1 # put new_node.input_channels if DW
    else:
        new_node.group = ch_out
        new_node.ch_in = 1
        new_node.name += 'DW'
    new_node.strides = stride # don't touch
    new_node.out_activation_bits = OUT_BITS
    new_node.input_activation_bits = IN_BITS
    new_node.weight_bits = W_BITS
    new_node.MACs = new_node.output_dim[0] * new_node.output_dim[1] * new_node.ch_out * new_node.kernel_shape[1] * new_node.kernel_shape[0] * new_node.ch_in
    torch.manual_seed(0)
    x = torch.Tensor(1, new_node.ch_in * new_node.group, new_node.input_dim[0], new_node.input_dim[1]).uniform_(0, (2**(new_node.input_activation_bits + 1)))
    x[x > (2**new_node.input_activation_bits - 1)] = 0
    x = torch.round(x)
    net = nn.Sequential(nn.Conv2d(new_node.ch_in * new_node.group, new_node.ch_out, kernel_size=(new_node.kernel_shape[0],new_node.kernel_shape[1]), stride=new_node.strides, padding=new_node.pads[0], groups=new_node.group, bias=False))
    net[0].weight.data.random_(-(2**(new_node.weight_bits - 1)), (2**(new_node.weight_bits - 1)))
    y = net(x)
    y = y.permute(0,2,3,1)
    y_shift = y.detach().numpy()
    new_node.outshift = 0 # don't touch
    i = 1
    while True:
        new_node.outshift = 0 # don't touch
        y_shift = y.detach().numpy()
        while True and new_node.outshift < 10:
            if float(sum(sum(sum(sum(np.logical_and(y_shift>0, y_shift<(2**OUT_BITS-1))))))) < len(y_shift.flatten())/(6.0*i):
                new_node.outshift+=1
            else:
                break
            y_shift = (y_shift / 2).astype('int')
        if new_node.outshift < 10:
            break
        else: 
            i += 1
    y = clip8(y.detach().numpy(), new_node.out_activation_bits, new_node.outshift)
    x = x.permute(0,2,3,1).flatten()
    Input_compressed = []
    z = 0
    import copy
    Loop_over = copy.deepcopy(x)
    for _, i_x in enumerate(Loop_over):
        if (z % int(8 / OUT_BITS)) == 0:
            Input_compressed.append(int(i_x.item()))
        else:
            Input_compressed[-1] += int(i_x.item()) << (OUT_BITS * (z % int(8 / OUT_BITS)))
        z += 1
    x_save = np.concatenate((np.asarray([0]), np.asarray(Input_compressed)), axis = 0)
    y = np.concatenate((np.asarray([0]), y.flatten()), axis = 0)    
    np.savetxt(args.network_dir + 'input.txt', x_save, delimiter=',')
    np.savetxt(args.network_dir + 'out_layer0.txt', y, delimiter=',')
    np.savetxt(args.network_dir + 'out_layer1.txt', y, delimiter=',')
    new_node.weights = net[0].weight.data.permute(0,2,3,1).numpy()
    final_node = node()
    final_node.input_index = 1
    final_node.output_index = 2
    final_node.pads    = [0, 0, 0, 0]
    final_node.input_dim = [1, 1]
    final_node.output_dim = [1, 1]
    final_node.ch_in =  new_node.ch_out
    final_node.ch_out = 1
    final_node.kernel_shape = [1, 1]
    final_node.group = 1
    final_node.strides = 1
    final_node.out_activation_bits = 32
    final_node.input_activation_bits = OUT_BITS
    final_node.weight_bits = 8
    final_node.MACs = final_node.output_dim[0] * final_node.output_dim[1] * final_node.ch_out * final_node.kernel_shape[1] * final_node.kernel_shape[0] * final_node.ch_in
    final_node.name = 'Gemm'
    final_node.weights = np.random.randint(low = -127, high = 128, size = (final_node.ch_in, final_node.kernel_shape[0], final_node.kernel_shape[1], final_node.ch_out))
    PULP_Nodes_Graph = []
    PULP_Nodes_Graph.append(new_node)
    # PULP_Nodes_Graph.append(new_node2)
    PULP_Nodes_Graph.append(final_node)
    model_deploy('GAP8', args.chip).print_model_network(PULP_Nodes_Graph,
                            100,
                            args.network_dir,
                            100,
                            args.verbose_level,
                            args.perf_layer,
                            args.l1_buffer_size,
                            args.master_stack,
                            args.slave_stack,
                            args.l2_buffer_size,
                            args.fc_frequency,
                            args.cl_frequency,
                            args.Bn_Relu_Bits, 
                            args.sdk,
                            args.backend,
                            args.dma_parallelization,
                            args.number_of_clusters,
                            type_data = 'char')
if __name__ == '__main__':
    main()
