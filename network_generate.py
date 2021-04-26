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
from Model_deployment import Model_deployment as model_deploy
import os
import argparse
from argparse import RawTextHelpFormatter
import numpy as np

def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--network_dir', default = "./examples/8-bits-2D/MobilenetV1/", help = 'directory of the onnx file of the network')
    parser.add_argument('--l1_buffer_size', type=int, default = 38000, help = 'L1 buffer size. IT DOES NOT INCLUDE SPACE FOR STACKS.')
    parser.add_argument('--l2_buffer_size', type=int, default = 380000, help = 'L2 buffer size.')
    parser.add_argument('--master_stack', type=int, default = 4096, help = 'Cluster Core 0 stack')
    parser.add_argument('--slave_stack', type=int, default = 3072, help = 'Cluster Core 1-7 stack')
    parser.add_argument('--Bn_Relu_Bits', type=int, default = 32, help = 'Number of bits for Relu/BN')
    parser.add_argument('--perf_layer', default = 'No', help = 'Yes: MAC/cycles per layer. No: No perf per layer.')
    parser.add_argument('--verbose_level', default = 'Check_all+Perf_final', help = "None: No_printf.\nPerf_final: only total performance\nCheck_all+Perf_final: all check + final performances \nLast+Perf_final: all check + final performances \nExtract the parameters from the onnx model")
    parser.add_argument('--chip', default = 'GAP8v3', help = 'GAP8v2 for fixing DMA issue. GAP8v3 otherise')
    parser.add_argument('--sdk', default = 'pulp_sdk', help = 'gap_sdk or pulp_sdk')
    parser.add_argument('--dma_parallelization', default = '8-cores', help = '8-cores or 1-core')
    parser.add_argument('--fc_frequency', default = 100000000, help = 'frequency of fabric controller')
    parser.add_argument('--cl_frequency', default = 100000000, help = 'frequency of cluster')
    parser.add_argument('--optional', default = '8bit', help = '8bit, mixed-sw, mixed-hw, 1D_Conv')
    args = parser.parse_args()

    for files in os.listdir(args.network_dir):
        if 'onnx' in files:
            net = files
        elif '.' not in files:
            for sub_files in files:
                if 'onnx' in files:
                    net = files
    if args.optional == '8bit' or args.optional == '1D_Conv':
        precision_dict_act     = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
        precision_dict_weights = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
    else:
        precision_dict_act     = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8]
        precision_dict_weights = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
    # if 'mixed' in args.optional: ### works only for MV1
    #     precision_dict_act     = []
    #     precision_dict_weights = []
    #     quant = args.network_dir.split('/')[3].split('_')
    #     precision_dict_act.append(int(quant[1][-2]))
    #     precision_dict_weights.append(int(quant[1][-4]))
    #     for i in np.arange(13):
    #         precision_dict_act.append(int(quant[3][-1]))
    #         precision_dict_act.append(int(quant[4][-2]))
    #         precision_dict_weights.append(int(quant[3][-3]))
    #         precision_dict_weights.append(int(quant[4][-4]))
    #     precision_dict_act.append(int(quant[4][-2]))
    #     precision_dict_act.append(32)
    #     precision_dict_weights.append(int(quant[4][-4]))
    #     precision_dict_weights.append(int(quant[2][-2]))
    PULP_Nodes_Graph = onnx_m('GAP8', args.chip, args.network_dir + net).parameters_from_onnx(100)
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
                            8, 8, 8, args.Bn_Relu_Bits, 
                            args.sdk,
                            args.dma_parallelization,args.
                            optional,
                            precision_dict_act,
                            precision_dict_weights)

if __name__ == '__main__':
    main()
