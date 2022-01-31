import os
import numpy as np
import torch
from mako.template import Template
import re
from collections import OrderedDict
import pandas as pd
from dory.ONNX_management import ONNX_management as onnx_m
from dory.Model_deployment import Model_deployment as model_deploy
from dory.tiling import Tiling
import os
import logging
import argparse
from argparse import RawTextHelpFormatter

def copy_files(platform, chip, BitA, BitW, BitO, optional, layer_mixed_list, version):
    os.system('rm -rf application')
    os.system('mkdir application')
    os.system('mkdir application/DORY_network')
    os.system('mkdir application/DORY_network/inc')
    os.system('mkdir application/DORY_network/src')
    os.system('cp ./dory/templates/dory.h ./application/DORY_network/inc/')
    os.system(
        'cp ./dory/templates/mem_controller.c  ./application/DORY_network/src/')
    os.system(
        'cp ./dory/templates/mem_controller.h  ./application/DORY_network/inc/')
    tk = OrderedDict([])
    tk['platform'] = platform
    tk = OrderedDict([])
    tk['platform'] = platform
    tk['chip'] = chip
    tmpl = Template(filename="dory/templates/dory.c")
    s = tmpl.render(**tk)
    save_string = './application/DORY_network/src/dory.c'
    with open(save_string, "w") as f:
        f.write(s)
    os.system('cp ./dory/templates/network.h ./application/DORY_network/inc/')
    os.system('cp ./dory/pulp-nn-q/' + version +'/pulp_nn_utils.h ./application/DORY_network/inc/')
    os.system('cp ./dory/pulp-nn-q/' + version +'/pulp_nn_utils.c ./application/DORY_network/src/')
    if optional == "1D-8bit":
        os.system('cp ./dory/pulp-nn-q/1D-8bit/include/*  ./application/DORY_network/inc/')
        os.system('cp ./dory/pulp-nn-q/1D-8bit/src/* ./application/DORY_network/src/')
    elif optional == "8bit":
        os.system('cp ./dory/pulp-nn-q/' + version +'/8bit/include/*  ./application/DORY_network/inc/')
        os.system('cp ./dory/pulp-nn-q/' + version +'/8bit/src/* ./application/DORY_network/src/')
    elif optional == "mixed":
        os.system(
            'cp ./dory/pulp-nn-q/' + version +'/mixed/include/*  ./application/DORY_network/inc/')
        for layer in layer_mixed_list:
            os.system('cp ./dory/pulp-nn-q/' + version +'/mixed/src/' + layer + ' ./application/DORY_network/src/')

def print_model_layer(PULP_Nodes_Graph,
                      number_of_deployed_layers=29,
                      load_dir='./MobilenetV1/',
                      check_layer=1,
                      layer_target=1,
                      performance_single_layer='Yes',
                      L1_buffer=44000,
                      platform = 'GAP8',
                      chip = 'GAP8v3',
                      BitIn = 8, BitW = 8, BitOut = 8, BitActivation = 64, optional = '8bit'):
    L1_dimension = L1_buffer
    l1_dimension_vector = []
    activation_dimensions = []
    name_list = []
    file_list_w = []
    # random input generated for test purpose.
    # Fetching weights,biases, k, and lambda for each node_iterating
    weights_to_write = []
    layer_mixed_list = []
    if optional == 'mixed':
        for i, nodes_to_deploy in enumerate(PULP_Nodes_Graph[:number_of_deployed_layers]):
            BitIn = BitOut
            if nodes_to_deploy.outshift != 'empty':
                BitOut = 32 - int(nodes_to_deploy.outshift)
            BitW = 8
            if BitOut != 2 and BitOut!= 4 and BitOut!= 8:
                BitOut = 8
            if i == layer_target:
	            if nodes_to_deploy.groups > 1:
	                layer_mixed_list.append(f'pulp_nn_dw_u{BitIn}_u{BitOut}_i{BitW}.c')
	            else:
	                layer_mixed_list.append(f'pulp_nn_conv_u{BitIn}_u{BitOut}_i{BitW}.c')
	            layer_mixed_list.append(f'pulp_nn_matmul_u{BitOut}_i{BitW}.c')
    version = str(BitActivation)+'bit'
    copy_files(platform, chip, BitIn, BitW, BitOut, optional, layer_mixed_list, version)
    for i, nodes_to_deploy in enumerate(PULP_Nodes_Graph[:number_of_deployed_layers]):
        if str(nodes_to_deploy.weights) != 'empty':
            nodes_to_deploy.weights = nodes_to_deploy.weights.flatten().tolist()
            for i_w, _ in enumerate(nodes_to_deploy.weights):
                nodes_to_deploy.weights[i_w] = np.uint8(
                    nodes_to_deploy.weights[i_w])
            weights = nodes_to_deploy.weights
        if str(nodes_to_deploy.k) != 'empty':
            if str(nodes_to_deploy.outmul) != 'empty':
                out_mult = np.int32(nodes_to_deploy.outmul)
            k_byte = []
            for i_k, _ in enumerate(nodes_to_deploy.k[0]):
                if BitActivation == 64:
                    val = np.int64(nodes_to_deploy.k[0][i_k][0][0])*out_mult
                else:
                    val = np.int32(nodes_to_deploy.k[0][i_k][0][0])*out_mult
                if BitActivation == 32:
                    k_byte.append(np.uint8(val &         0x000000FF))
                    k_byte.append(np.uint8((val >> 8) &  0x000000FF))
                    k_byte.append(np.uint8((val >> 16) & 0x000000FF))
                    k_byte.append(np.uint8((val >> 24) & 0x000000FF))
                if BitActivation == 64:
                    k_byte.append(np.uint8(val &         0x00000000000000FF))
                    k_byte.append(np.uint8((val >> 8) &  0x00000000000000FF))
                    k_byte.append(np.uint8((val >> 16) & 0x00000000000000FF))
                    k_byte.append(np.uint8((val >> 24) & 0x00000000000000FF))
                    k_byte.append(np.uint8((val >> 32) & 0x00000000000000FF))
                    k_byte.append(np.uint8((val >> 40) & 0x00000000000000FF))
                    k_byte.append(np.uint8((val >> 48) & 0x00000000000000FF))
                    k_byte.append(np.uint8((val >> 56) & 0x00000000000000FF))
            nodes_to_deploy.k = k_byte
            weights = np.concatenate((weights, nodes_to_deploy.k))
            lambd = np.float64(nodes_to_deploy.lambd[0]) * out_mult
            if str(nodes_to_deploy.lambd) != 'empty':
                lambd_byte = []
                for i_l, _ in enumerate(nodes_to_deploy.lambd[0]):
                    if BitActivation == 64:
                        val = np.int64(lambd[i_l][0][0])
                    else:
                        val = np.int32(lambd[i_l][0][0])
                    if BitActivation == 32:
                        lambd_byte.append(np.uint8(val &         0x000000FF))
                        lambd_byte.append(np.uint8((val >> 8) &  0x000000FF))
                        lambd_byte.append(np.uint8((val >> 16) & 0x000000FF))
                        lambd_byte.append(np.uint8((val >> 24) & 0x000000FF))
                    if BitActivation == 64:
                        lambd_byte.append(np.uint8(val &         0x00000000000000FF))
                        lambd_byte.append(np.uint8((val >> 8) &  0x00000000000000FF))
                        lambd_byte.append(np.uint8((val >> 16) & 0x00000000000000FF))
                        lambd_byte.append(np.uint8((val >> 24) & 0x00000000000000FF))
                        lambd_byte.append(np.uint8((val >> 32) & 0x00000000000000FF))
                        lambd_byte.append(np.uint8((val >> 40) & 0x00000000000000FF))
                        lambd_byte.append(np.uint8((val >> 48) & 0x00000000000000FF))
                        lambd_byte.append(np.uint8((val >> 56) & 0x00000000000000FF))

            nodes_to_deploy.lambd = lambd_byte
            weights = np.concatenate((weights, nodes_to_deploy.lambd))
            if str(nodes_to_deploy.outmul) != 'empty':
                PULP_Nodes_Graph[i].outmul = 1
        if str(nodes_to_deploy.bias) != 'empty':
            nodes_to_deploy.bias = nodes_to_deploy.bias.flatten().tolist()
            for i_b, _ in enumerate(nodes_to_deploy.bias):
                nodes_to_deploy.bias[i_b] = np.uint8(nodes_to_deploy.bias[i_b])
            weights = np.concatenate((weights, nodes_to_deploy.bias))
        if str(nodes_to_deploy.weights) != 'empty':
            while len(weights) % 4 != 0:
                weights = np.concatenate((weights, np.asarray([0])))
            weights = np.asarray(weights)
            weights_to_write.append(weights)
            string_layer = nodes_to_deploy.name + str(i) + "_weights.hex"
            file_list_w.append(string_layer)
            save_s = './application/DORY_network/' + string_layer
            with open(save_s, 'wb') as f:
                for l in weights.astype('uint8').flatten():
                    f.write(bytes((l,)))
    n_layers = number_of_deployed_layers
    # creation of the directory with static and not wrapping files.

    layer_list = []
    stringa_features = []
    name_layer_list = []
    name_layer_list_internal = []
    i_w = 0
    BitOut = 8
    for i, nodes_to_deploy in enumerate(PULP_Nodes_Graph[:(layer_target + 1)]):
        if str(nodes_to_deploy.outmul) != 'empty':
            out_mult = np.uint32(nodes_to_deploy.outmul)
        if str(nodes_to_deploy.outshift) != 'empty':
            out_shift = np.uint32(nodes_to_deploy.outshift)
        if 'Conv' in nodes_to_deploy.name or 'Gemm' in nodes_to_deploy.name or 'MatMul' in nodes_to_deploy.name:
            i_w += 1
            layer = 'Conv'
        if 'Pool' in nodes_to_deploy.name:
            layer = 'Pool'
        if 'Add' in nodes_to_deploy.name:
            layer = 'Add'

        name_layer = "layer" + nodes_to_deploy.name + str(i)
        if i < len(PULP_Nodes_Graph)-1:
            if PULP_Nodes_Graph[i+1].input_channels*PULP_Nodes_Graph[i+1].output_channels*PULP_Nodes_Graph[i+1].filter_size_h*PULP_Nodes_Graph[i+1].filter_size_w > 100000:
                weight_overhead = 10000
            else:
                weight_overhead = PULP_Nodes_Graph[i+1].input_channels*PULP_Nodes_Graph[i+1].output_channels*PULP_Nodes_Graph[i+1].filter_size_h*PULP_Nodes_Graph[i+1].filter_size_w 
        else:
            weight_overhead = 0
        if optional != '8bit':
            BitIn = BitOut
            if nodes_to_deploy.outshift != 'empty':
                BitOut = 32 - int(nodes_to_deploy.outshift)
            BitW = 8
            if BitOut != 2 and BitOut!= 4 and BitOut!= 8:
                BitOut = 8
        if i == layer_target:
            fileh = logging.FileHandler('logs/Tiling_profiling.log', 'a')
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            fileh.setFormatter(formatter)
            fileh.setLevel(logging.DEBUG)

            log = logging.getLogger()  # root logger
            for hdlr in log.handlers[:]:  # remove all old handlers
                log.removeHandler(hdlr)
            log.addHandler(fileh)
            print("Creating tiling profiling in Tiling_profling.log")
            if performance_single_layer == 'Yes':
                tile_gen = Tiling(layer,
                  nodes_to_deploy.output_channels,
                  [nodes_to_deploy.filter_size_h, nodes_to_deploy.filter_size_w],
                  nodes_to_deploy.stride,
                  [nodes_to_deploy.padding_top,nodes_to_deploy.padding_left,nodes_to_deploy.padding_bottom,nodes_to_deploy.padding_right],
                  nodes_to_deploy.groups,
                  [nodes_to_deploy.input_channels * nodes_to_deploy.groups,
                  nodes_to_deploy.input_h, nodes_to_deploy.input_w],
                  L1_dimension,
                  400000-weight_overhead,
                  platform,
                  chip,
                  test_location='L2+performance+partial',
                  BitIn=BitIn,
                  BitW=BitW,
                  BitOut=BitOut,
                  BitActivation = BitActivation,
                  optional_type=optional)
            else:
                tile_gen = Tiling(layer,
                  nodes_to_deploy.output_channels,
                  [nodes_to_deploy.filter_size_h, nodes_to_deploy.filter_size_w],
                  nodes_to_deploy.stride,
                  [nodes_to_deploy.padding_top,nodes_to_deploy.padding_left,nodes_to_deploy.padding_bottom,nodes_to_deploy.padding_right],
                  nodes_to_deploy.groups,
                  [nodes_to_deploy.input_channels * nodes_to_deploy.groups,
                  nodes_to_deploy.input_h, nodes_to_deploy.input_w],
                  L1_dimension,
                  400000-weight_overhead,
                  platform,
                  chip,
                  test_location='L2+partial',
                  BitIn=BitIn,
                  BitW=BitW,
                  BitOut=BitOut,
                  BitActivation = BitActivation,
                  optional_type=optional)

            # Decomment to shrink the code --> out_mul and out_shift not
            # working however
            str_l = 'ch_in' + str(nodes_to_deploy.input_channels) + 'ch_out' + str(nodes_to_deploy.output_channels) + 'groups' + str(
                nodes_to_deploy.groups) + 'dim_image' + str(nodes_to_deploy.input_h,) + 'stride' + str(nodes_to_deploy.stride)
            name = nodes_to_deploy.name
            for scan_i, _ in enumerate(stringa_features):
                if str_l == stringa_features[scan_i] and str(layer) == str(layer_list[scan_i]):
                    name_layer = name_layer_list[scan_i]
                    name = name_layer_list_internal[scan_i]
            stringa_features.append(str_l)
            layer_list.append(layer)
            name_layer_list.append(name_layer)
            name_layer_list_internal.append(name)
            if layer_target == 0:
                X_in = pd.read_csv(load_dir + "input.txt")
            else:
                X_in = pd.read_csv(load_dir + "out_layer" +
                                   str(layer_target - 1) + ".txt")    
            X_in = np.ceil(X_in.values[:, 0]).astype(int)
            for i_x, _ in enumerate(X_in):
                X_in[i_x] = np.uint8(X_in[i_x])
            #X_in =X_in[:(nodes_to_deploy.input_channels*nodes_to_deploy.input_h*nodes_to_deploy.input_w)]
            X_out = pd.read_csv(load_dir + "out_layer" +
                                str(layer_target) + ".txt")
            X_out = X_out.values[:, 0].astype(int)
            for i_x, _ in enumerate(X_out):
                X_out[i_x] = np.uint8(X_out[i_x])
            #X_out =X_in[:int(nodes_to_deploy.output_channels*nodes_to_deploy.input_h*nodes_to_deploy.input_w/nodes_to_deploy.stride)]
            Input_compressed = []
            z = 0
            import copy
            Loop_over = copy.deepcopy(X_in)
            for _, i_x in enumerate(Loop_over):
                if (z % int(8 / BitIn)) == 0:
                    Input_compressed.append(int(i_x.item()))
                else:
                    Input_compressed[-1] += int(i_x.item()
                                                ) << (BitIn * (z % int(8 / BitIn)))
                z += 1
            Output_compressed = []
            z = 0
            Loop_over = copy.deepcopy(X_out)
            for _, i_x in enumerate(Loop_over):
                if (z % int(8 / BitOut)) == 0:
                    Output_compressed.append(int(i_x.item()))
                else:
                    Output_compressed[-1] += int(i_x.item()
                                                 ) << (BitOut * (z % int(8 / BitOut)))
                z += 1            
            relu = 0
            BN = 0
            DW = 0
            if 'Relu' in nodes_to_deploy.name:
                relu = 1
            if 'BN' in nodes_to_deploy.name:
                BN = 1
            if 'DW' in nodes_to_deploy.name:
                DW = 1
            if 'Gemm' in nodes_to_deploy.name or 'Conv' in nodes_to_deploy.name:
                in_dim2, out_dim2, weights_dim, l1_dim2, L3_tiling, factor_ch_out, factor_h_out, factor_h_in = tile_gen.get_tiling(X=np.asarray(Input_compressed), 
                															Y=np.asarray(Output_compressed), 
                															W=weights_to_write[i_w - 1].astype('uint8').flatten(),
                                                                            relu=relu, BN=BN, DW=DW,
                                                                            has_bias=0,
                                                                            out_mul=nodes_to_deploy.outmul,
                                                                            out_shift=nodes_to_deploy.outshift,
                                                                            name=name_layer)
                PULP_Nodes_Graph[i].L3_allocation = L3_tiling
            elif 'Pool' in nodes_to_deploy.name:
                in_dim2, out_dim2, l1_dim2 = tile_gen.get_tiling(X=0, Y=0, W=0,
                                                                 relu=relu,
                                                                 out_mul=nodes_to_deploy.outmul,
                                                                 out_shift=nodes_to_deploy.outshift,
                                                                 name=name_layer,
                                                                 type=name)
                L3_tiling = 0
            elif 'Add' in nodes_to_deploy.name:
                in_dim2, out_dim2, l1_dim2 = tile_gen.get_tiling(X=0, Y=0, W=0,
                                                                 relu=relu,
                                                                 out_mul1=nodes_to_deploy.inmul1,
                                                                 out_mul2=nodes_to_deploy.inmul2,
                                                                 out_shift=nodes_to_deploy.outshift,
                                                                 name=name_layer,
                                                                 type=name)
                L3_tiling = 0
            if L3_tiling == 1:
                name_layer = name_layer + 'L3'
            name_list.append(name_layer)
            PULP_Nodes_Graph[i].input_activation_dimensions = in_dim2
            if i != 0:
                PULP_Nodes_Graph[i - 1].output_activation_dimensions = in_dim2
            PULP_Nodes_Graph[i].l1_dimensions = l1_dim2
            if i == len(PULP_Nodes_Graph[:number_of_deployed_layers]) - 1:
                PULP_Nodes_Graph[i].output_activation_dimensions = out_dim2

def main():
    """
    Create an instance of ONNX_management.
    Then, extract infos and generate all the application folder.
    Give the folder on which you have 1 onnx file and the intermediate results
    """
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--network_dir', default = "./Test_suite_DORY/PenguiNet_32/", help = 'directory of the onnx file of the network')
    parser.add_argument('--n_layer', type=int, default = 0, help = 'Layer that you want to deploy')
    parser.add_argument('--l1_buffer_size', type=int, default = 35000, help = 'L1 buffer size. IT DOES NOT INCLUDE SPACE FOR STACKS.')
    parser.add_argument('--Bn_Relu_Bits', type=int, default = 32, help = 'Number of bits for Relu/BN')
    parser.add_argument('--perf_layer', default = 'Yes', help = 'Yes: MAC/cycles per layer. No: No perf per layer.')
    parser.add_argument('--chip', default = 'GAP8v3', help = 'GAP8v2 for fixing DMA issue. GAP8v3 otherise')
    args = parser.parse_args()

    for files in os.listdir(args.network_dir):
        if 'onnx' in files:
            net = files
        elif '.' not in files:
            for sub_files in files:
                if 'onnx' in files:
                    net = files
    PULP_Nodes_Graph = onnx_m('GAP8', args.chip, args.network_dir + net).parameters_from_onnx(args.n_layer+1)
    print_model_layer(PULP_Nodes_Graph,
                      number_of_deployed_layers=args.n_layer+1,
                      load_dir=args.network_dir,
                      check_layer=args.n_layer,
                      layer_target=args.n_layer,
                      performance_single_layer=args.perf_layer,
                      L1_buffer=args.l1_buffer_size,
                      platform = 'GAP8',
                      chip = args.chip,
                      BitIn = 8, BitW = 8, BitOut = 8, BitActivation = args.Bn_Relu_Bits, optional = '8bit')

if __name__ == '__main__':
    main()
