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
sys.path.append('../Frontend/')
sys.path.append('../NN_Deployment/')
sys.path.append('../Tiler/')
sys.path.append('../Templates_writer/')
from PULP_node import node_element as node
import os
import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import logging
import torch
import torch.nn as nn
import copy

def clip8(conv, bits, shift):
    conv = conv.astype(int)>>shift;
    conv[conv >= +(2**(bits) - 1)] = +(2**(bits) - 1)
    conv[conv <= 0] = 0
    out = np.uint8(conv)
    return out

def add_node(
        kernel_shape,
        ch_in,
        ch_out,
        DW,
        stride,
        pads,
        input_dim,
        output_dim,
        W_BITS, IN_BITS, OUT_BITS):

    new_node = node()
    new_node.pads    = pads
    new_node.input_dim = input_dim
    new_node.output_dim = output_dim
    new_node.ch_in = ch_in
    new_node.ch_out = ch_out # = new_node.input_channels if DW
    new_node.kernel_shape = kernel_shape
    new_node.name = 'Conv'
    h_dimension = new_node.get_parameter('kernel_shape')[0] + new_node.get_parameter('input_dim')[0] + new_node.get_parameter('output_dim')[0]
    if h_dimension == 3:
        new_node.name = 'Gemm'
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
    return new_node

def main(nodes_list
        ):
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
    parser.add_argument('--layer_number', type=int, default = 1, help = 'Number of layer from excel.')
    parser.add_argument('--optional', default = 'mixed-sw', help = 'auto (based on layer precision, 8bits or mixed-sw), 8bit, mixed-hw, mixed-sw')
    args = parser.parse_args()

    list_nodes_final = []

    ch_in_first_layer = 1
    L2_memory = nodes_list[0].input_dim[0]*nodes_list[0].input_dim[1]*ch_in_first_layer + ch_in_first_layer*nodes_list[0].ch_in*1*1 + nodes_list[0].input_dim[0]*nodes_list[0].input_dim[1]*nodes_list[0].ch_in*nodes_list[0].group
    if L2_memory > args.l2_buffer_size:
        ch_in_first_layer = int(100000/nodes_list[0].input_dim[0]/nodes_list[0].input_dim[1])
    new_node = add_node(
        [1,1],
        ch_in_first_layer,
        nodes_list[0].ch_in * nodes_list[0].group,
        0,
        1,
        [0,0,0,0],
        nodes_list[0].input_dim,
        nodes_list[0].input_dim,
        8, 8, nodes_list[0].input_activation_bits)
    new_node.input_index = 0 # don't touch
    new_node.output_index = 1 # don't touch
    list_nodes_final.append(new_node)

    for i, node in enumerate(nodes_list):
        node.input_index = i+1
        node.output_index = i+2
        list_nodes_final.append(node)

    new_node = add_node(
        [1,1],
        # list_nodes_final[-1].ch_out*list_nodes_final[-1].output_dim[0]*list_nodes_final[-1].output_dim[1],
        list_nodes_final[-1].ch_out, ### MEMORY OK BUT LAST LAYER FAILED. NOT A PROBLEM FOR THE OTHER CHECKS
        8,
        0,
        1,
        [0,0,0,0],
        [1, 1],
        [1, 1],
        8, nodes_list[-1].out_activation_bits, 32)
    new_node.input_index = i+2 # don't touch
    new_node.output_index = i+3 # don't touch
    list_nodes_final.append(new_node)
    
    torch.manual_seed(0)
    ##### INPUT CREATION #####
    x = torch.Tensor(1, list_nodes_final[0].ch_in * list_nodes_final[0].group, list_nodes_final[0].input_dim[0], list_nodes_final[0].input_dim[1]).uniform_(0, (2**(list_nodes_final[0].input_activation_bits + 1)))
    x[x > (2**list_nodes_final[0].input_activation_bits - 1)] = 0
    x = torch.round(x)
    x_save = copy.deepcopy(x)
    x_save = x_save.permute(0,2,3,1).flatten()
    Input_compressed = []
    z = 0
    Loop_over = copy.deepcopy(x_save)
    for _, i_x in enumerate(Loop_over):
        if (z % int(8 / list_nodes_final[0].input_activation_bits)) == 0:
            Input_compressed.append(int(i_x.item()))
        else:
            Input_compressed[-1] += int(i_x.item()) << (list_nodes_final[0].input_activation_bits * (z % int(8 / list_nodes_final[0].input_activation_bits)))
        z += 1
    x_save = np.concatenate((np.asarray([0]), np.asarray(Input_compressed)), axis = 0)
    np.savetxt(args.network_dir + 'input.txt', x_save, delimiter=',')

    #### GENERATION OF INTERMEDIATE DATA ####
    for i, node in enumerate(list_nodes_final):
        net = nn.Sequential(nn.Conv2d(list_nodes_final[i].ch_in * list_nodes_final[i].group, list_nodes_final[i].ch_out, kernel_size=(list_nodes_final[i].kernel_shape[0],list_nodes_final[i].kernel_shape[1]), stride=list_nodes_final[i].strides, padding=list_nodes_final[i].pads[0], groups=list_nodes_final[i].group, bias=False))
        net[0].weight.data.random_(-(2**(list_nodes_final[i].weight_bits - 1)), (2**(list_nodes_final[i].weight_bits - 1)))
        # if i == len(list_nodes_final) - 1:
        #     x = x.reshape(1, list_nodes_final[i].ch_in, 1, 1)
        y = net(x)
        y = y.permute(0,2,3,1)
        y_shift = y.detach().numpy()
        list_nodes_final[i].outshift = 0 # don't touch
        shift_index = 1
        while True:
            list_nodes_final[i].outshift = 0 # don't touch
            y_shift = y.detach().numpy()
            while True and list_nodes_final[i].outshift < 10:
                if float(sum(sum(sum(sum(np.logical_and(y_shift>0, y_shift<(2**list_nodes_final[i].out_activation_bits-1))))))) < len(y_shift.flatten())/(6.0*shift_index):
                    list_nodes_final[i].outshift+=1
                else:
                    break
                y_shift = (y_shift / 2).astype('int')
            if list_nodes_final[i].outshift < 10:
                break
            else: 
                shift_index += 1
            if shift_index == 10:
                break
        y = clip8(y.detach().numpy(), list_nodes_final[i].out_activation_bits, list_nodes_final[i].outshift)
        y_new = copy.deepcopy(y)
        y = np.concatenate((np.asarray([0]), y.flatten()), axis = 0) 
        x = torch.tensor(y_new).permute(0,3,1,2).float()
        np.savetxt(args.network_dir + f'out_layer{i}.txt', y, delimiter=',')
        list_nodes_final[i].weights = net[0].weight.data.permute(0,2,3,1).numpy()
        list_nodes_final[i].weights_raw = net[0].weight.data.permute(0,2,3,1).numpy()

    if args.backend == 'MCU':
        sys.path.append('../NN_Deployment/MCU/')
        from Model_deployment_MCU import Model_deployment_MCU as model_deploy
        type_data = 'char'
    elif args.backend == 'Occamy':
        sys.path.append('../NN_Deployment/Occamy/')
        from Model_deployment_Occamy import Model_deployment_Occamy as model_deploy
        type_data = 'float'

    model_deploy('GAP8', args.chip).print_model_network(list_nodes_final,
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
                            args.optional,
                            type_data = type_data)
if __name__ == '__main__':
    ##############################################
    ###### INPUT PARAMETERS TO DEFINE ############
    ##############################################
    kernel_shape = [3,3]
    ch_in = 64
    ch_out = 64
    DW = 0
    stride = 1
    pads = [1,1,1,1]
    input_dim = [32,32]
    output_dim = [32, 32]
    # kernel_shape = [1,1]
    # ch_in = 1024
    # ch_out = 64
    # DW = 0
    # stride = 1
    # pads = [0,0,0,0]
    # input_dim = [1,1]
    # output_dim = [1, 1]
    W_BITS = 8; IN_BITS = 8; OUT_BITS = 8;
    ##############################################
    ##########DON'T TOUCH AFTER###################
    ##############################################

    list_nodes = []
    new_node = add_node(
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
