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

#####################CONFIG PARAMETERS #########################
# BNRelu_bits. Number of bits for lambda and k parameters in BNRelu. 32 or 64
# onnx file.

# Libraries
import argparse
from argparse import RawTextHelpFormatter
import json
import sys

# Directory
sys.path.append('../00_Parsers/')
sys.path.append('../01_Utils/')
sys.path.append('../01_Utils/Templates_writer/')
sys.path.append('../NN_Deployment/')
sys.path.append('../')

Frontends = ["NEMO", "Quantlab"]
Hardware_targets = ["GAP8", "Occamy", "Diana"]
NoneType = type(None)


def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('frontend', type=str, choices=Frontends, help='Frontend from which the onnx is produced and from which the network has been trained')
    parser.add_argument('hardware_target', type=str, choices=Hardware_targets, help='Hardware platform for which the code is optimized')
    parser.add_argument('config_file', type=str, help='Path to the JSON file that specifies the ONNX file of the network and other information.')
    parser.add_argument('--verbose_level', default='Check_all+Perf_final',
                        help="None: No_printf.\nPerf_final: only total performance\nCheck_all+Perf_final: all check + "
                             "final performances \nLast+Perf_final: all check + final performances \nExtract the "
                             "parameters from the onnx model")
    parser.add_argument('--perf_layer', default='No', help='Yes: MAC/cycles per layer. No: No perf per layer.')
    parser.add_argument('--optional', default='auto',
                        help='auto (based on layer precision, 8bits or mixed-sw), 8bit, mixed-hw, mixed-sw')

    args = parser.parse_args()

    #############################################################
    #### DORY FRONTEND --> From ONNX TO JSON FORMAT OF ALL NODES.
    #############################################################

    print("Using {} as frontend. Targeting {} platform. ".format(args.frontend, args.hardware_target))

    ## Reading the json configuration file
    with open(args.config_file) as f:
        json_configuration_file = json.load(f)

    ## Reading the onnx file
    onnx_file = json_configuration_file["onnx_file"]
    print("Using {} target input onnx.\n".format(onnx_file))

    ## Including and running the transformation from Onnx to a DORY compatible graph
    sys.path.append('../02_Frontend-frameworks/{}/'.format(args.frontend))
    onnx_manager = __import__('Parser')
    onnx_to_DORY = getattr(onnx_manager, '{}_onnx'.format(args.frontend))
    DORY_Graph = onnx_to_DORY(onnx_file, json_configuration_file).full_graph_parsing()

    ## Including and running the transformation from DORY IR to DORY HW IR
    sys.path.append('../03_Hardware-targets/{}/'.format(args.hardware_target))
    sys.path.append('../03_Hardware-targets/{}/Tiler/'.format(args.hardware_target))
    onnx_manager = __import__('HW_Parser')
    DORY_to_DORY_HW = getattr(onnx_manager, '{}_onnx'.format(args.hardware_target))
    DORY_Graph = DORY_to_DORY_HW(DORY_Graph, json_configuration_file).full_graph_parsing()

    ## Deployment of the model on the target architecture
    onnx_manager = __import__('C_Parser')
    DORY_HW_to_C = getattr(onnx_manager, 'C_Parser')
    DORY_Graph = DORY_HW_to_C(DORY_Graph, json_configuration_file, args.verbose_level, args.perf_layer, args.optional).full_graph_parsing()


if __name__ == '__main__':
    main()
