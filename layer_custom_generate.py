import os
import numpy as np
import torch
import torch.nn as nn
from mako.template import Template
import re
from collections import OrderedDict
import pandas as pd
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
    os.system('cp ./dory/templates/mem_controller.c  ./application/DORY_network/src/')
    os.system('cp ./dory/templates/mem_controller.h  ./application/DORY_network/inc/')
    tk = OrderedDict([])
    tk['platform'] = platform
    tmpl = Template(filename="dory/templates/mchan_test.h")
    s = tmpl.render(**tk)
    save_string = './application/DORY_network/inc/mchan_test.h'
    with open(save_string, "w") as f:
        f.write(s)
    tk = OrderedDict([])
    tk['platform'] = platform
    tk['chip'] = chip
    tmpl = Template(filename="dory/templates/dory.c")
    s = tmpl.render(**tk)
    save_string = './application/DORY_network/src/dory.c'
    with open(save_string, "w") as f:
        f.write(s)
    os.system('cp ./dory/templates/network.h ./application/DORY_network/inc/')
    if optional == "1D-8bit":
        os.system('cp ./dory/pulp-nn-q/1D_conv/include/*  ./application/DORY_network/inc/')
        os.system('cp ./dory/pulp-nn-q/1D_conv/src/* ./application/DORY_network/src/')
    elif optional == "8bit":
        os.system('cp ./dory/pulp-nn-q/' + version +'/pulp_nn_utils.h ./application/DORY_network/inc/')
        os.system('cp ./dory/pulp-nn-q/' + version +'/pulp_nn_utils.c ./application/DORY_network/src/')
        os.system('cp ./dory/pulp-nn-q/' + version +'/8bit/include/*  ./application/DORY_network/inc/')
        os.system('cp ./dory/pulp-nn-q/' + version +'/8bit/src/* ./application/DORY_network/src/')
    elif optional == "mixed":
        os.system('cp ./dory/pulp-nn-q/' + version +'/pulp_nn_utils.h ./application/DORY_network/inc/')
        os.system('cp ./dory/pulp-nn-q/' + version +'/pulp_nn_utils.c ./application/DORY_network/src/')
        os.system('cp ./dory/pulp-nn-q/' + version +'/mixed/include/*  ./application/DORY_network/inc/')
        for layer in layer_mixed_list:
            os.system('cp ./dory/pulp-nn-q/' + version +'/mixed/src/' + layer + ' ./application/DORY_network/src/')


def print_model_layer(chip,optional, Input_act, Weights, Output_act, mult, summ, shift, L1_dim,L2_dim, s, p, dilation, groups, BitIn, BitW, BitOut, BitActivation, input_L3, forcing):
    L1_dimension = L1_dim
    l1_dimension_vector = []
    activation_dimensions = []
    name_list = []
    file_list_w = []
    # random input generated for test purpose.
    # Fetching weights,biases, k, and lambda for each node_iterating
    weights_to_write = []
    layer_mixed_list = []
    if optional == 'mixed':
        if groups > 1:
            layer_mixed_list.append(f'pulp_nn_dw_u{BitIn}_u{BitOut}_i{BitW}.c')
        else:
            layer_mixed_list.append(f'pulp_nn_conv_u{BitIn}_u{BitOut}_i{BitW}.c')
        layer_mixed_list.append(f'pulp_nn_matmul_u{BitOut}_i{BitW}.c')
    copy_files('GAP8', chip,BitIn, BitW, BitOut, optional, layer_mixed_list, str(BitActivation)+'bit')
    # Formula for manual check:
    # int(((Input_act[0][0][2][0]*Weights[0])*mult+summ)>>shift)
    layer_list = []
    stringa_features = []
    name_layer_list = []
    name_layer_list_internal = []
    i_w = 0
    name_layer = 'Relu0_' + optional
    os.system('rm -rf logs/Layer_custom_tiling.log')
    logging.basicConfig(filename='logs/Layer_custom_tiling.log',
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
    fileh = logging.FileHandler('logs/Layer_custom_tiling.log', 'a')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fileh.setFormatter(formatter)
    fileh.setLevel(logging.DEBUG)

    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    log.addHandler(fileh)
    print("Creating tiling profiling in Tiling_profling.log")
    Input_compressed = []
    i = 0
    import copy
    if np.asarray(Input_act.shape).shape[0]==3:
        Loop_over = copy.deepcopy(Input_act.permute(0, 2, 1).flatten())
    else:
        Loop_over = copy.deepcopy(Input_act.permute(0, 2, 3, 1).flatten())
    for _, i_x in enumerate(Loop_over):
        if (i % int(8 / BitIn)) == 0:
            Input_compressed.append(int(i_x.item()))
        else:
            Input_compressed[-1] += int(i_x.item()) << (BitIn * (i % int(8 / BitIn)))
        i += 1
    Output_compressed = []
    i = 0
    if np.asarray(Input_act.shape).shape[0]==3:
        Loop_over = copy.deepcopy(Output_act.transpose(0, 2, 1).flatten())
    else:
        Loop_over = copy.deepcopy(Output_act.transpose(0, 2, 3, 1).flatten())
    for _, i_x in enumerate(Loop_over):
        if (i % int(8 / BitOut)) == 0:
            Output_compressed.append(int(i_x.item()))
        else:
            Output_compressed[-1] += int(i_x.item()) << (BitOut * (i % int(8 / BitOut)))
        i += 1
    Weights_compressed = []
    i = 0
    if groups == 1:
        DW = 0
    else:
        DW = 1    
    if DW == 0:
        if np.asarray(Input_act.shape).shape[0]==3:
            Loop_over = copy.deepcopy(Weights.permute(0, 2, 1).flatten())
        else:
            Loop_over = copy.deepcopy(Weights.permute(0, 2, 3, 1).flatten())
        for _, i_x in enumerate(Loop_over):
            if BitW == 2:
                i_x = int(i_x) & 0x00000003
            elif BitW == 4:
                i_x = int(i_x) & 0x0000000F
            elif BitW == 8:
                i_x = int(i_x) & 0x000000FF
            if (i % int(8 / BitW)) == 0:
                Weights_compressed.append(i_x)
            else:
                Weights_compressed[-1] += i_x << (BitW * (i % int(8 / BitW)))
            i += 1
    else:
        if np.asarray(Input_act.shape).shape[0]==3:
            Loop_over = copy.deepcopy(Weights.permute(0, 2, 1).flatten())
        else:
            Loop_over = copy.deepcopy(Weights.permute(0, 2, 3, 1).flatten())
        for c in np.arange(int(Weights.shape[0]/(int(8/BitW)))):
            for i in np.arange(Weights.shape[2]):
                for j in np.arange(Weights.shape[3]):
                    for z in np.arange(int(8/BitW)):
                        if BitW == 2:
                            i_x = int(Weights[c*4+z,0,i,j]) & 0x00000003
                        elif BitW == 4:
                            i_x = int(Weights[c*2+z,0,i,j]) & 0x0000000F
                        elif BitW == 8:
                            i_x = int(Weights[c,0,i,j]) & 0x000000FF
                        if z == 0:
                            Weights_compressed.append(i_x)
                        else:
                            Weights_compressed[-1] += i_x << (BitW * z)

    for i_x in range(Output_act.shape[1]):
        if np.asarray(Output_act.shape).shape[0]==3:
            val = np.uint32(mult[0, i_x, 0])
        else:
            val = np.uint32(mult[0, i_x, 0, 0])
        Weights_compressed.append(np.uint8(val & 0x000000FF))
        Weights_compressed.append(np.uint8((val >> 8) & 0x000000FF))
        Weights_compressed.append(np.uint8((val >> 16) & 0x000000FF))
        Weights_compressed.append(np.uint8((val >> 24) & 0x000000FF))
        if BitActivation == 64 and val >= 0:
            Weights_compressed.append(np.uint8(0))
            Weights_compressed.append(np.uint8(0))
            Weights_compressed.append(np.uint8(0))
            Weights_compressed.append(np.uint8(0))
        elif BitActivation == 64:
            Weights_compressed.append(np.uint8(0xFF))
            Weights_compressed.append(np.uint8(0xFF))
            Weights_compressed.append(np.uint8(0xFF))
            Weights_compressed.append(np.uint8(0xFF))

    for i_x in range(Output_act.shape[1]):
        if np.asarray(Output_act.shape).shape[0]==3:
            val = np.uint32(summ[0, i_x, 0])
        else:
            val = np.uint32(summ[0, i_x, 0, 0])
        Weights_compressed.append(np.uint8(val & 0x000000FF))
        Weights_compressed.append(np.uint8((val >> 8) & 0x000000FF))
        Weights_compressed.append(np.uint8((val >> 16) & 0x000000FF))
        Weights_compressed.append(np.uint8((val >> 24) & 0x000000FF))
        if BitActivation == 64 and val >= 0:
            Weights_compressed.append(np.uint8(0))
            Weights_compressed.append(np.uint8(0))
            Weights_compressed.append(np.uint8(0))
            Weights_compressed.append(np.uint8(0))
        elif BitActivation == 64:
            Weights_compressed.append(np.uint8(0xFF))
            Weights_compressed.append(np.uint8(0xFF))
            Weights_compressed.append(np.uint8(0xFF))
            Weights_compressed.append(np.uint8(0xFF))
    relu = 1
    BN = 1

    while len(Weights_compressed) % 4 != 0:
        Weights_compressed = np.concatenate((Weights_compressed, np.asarray([0])))
    Weights_compressed = np.asarray(Weights_compressed)
    string_layer = name_layer[5:] + "_weights.hex"
    save_s = './application/DORY_network/' + string_layer    
    with open(save_s, 'wb') as f:
        for l in Weights_compressed.astype('uint8').flatten():
            f.write(bytes((l,)))
    if np.asarray(Output_act.shape).shape[0]==3:
        layer = 'Conv1d'
        tile_gen = Tiling(layer,
                  Output_act.shape[1],
                  Weights.shape[2],
                  s,
                  p[0],
                  groups,
                  [Input_act.shape[1], Input_act.shape[2], 1],
                  L1_buffer=L1_dimension,
                  L2_buffer=L2_dim,
                  platform='GAP8',
                  chip=chip,
                  test_location='L2+performance+partial',
                  BitIn=BitIn,
                  BitW=BitW,
                  BitOut=BitOut,
                  BitActivation = BitActivation,
                  optional_type=optional)
        in_dim, out_dim, weights_dim, L1_tiles_size = tile_gen.get_tiling(X=np.asarray(Input_compressed),
                                                                    Y=np.asarray(Output_compressed),
                                                                    W=np.asarray(Weights_compressed),
                                                                    relu=relu, BN=BN, dilation = dilation,
                                                                    has_bias=0,
                                                                    out_mul=np.uint32(mult[0, 0, 0]),
                                                                    out_shift=np.uint32(shift)[0],
                                                                    name=name_layer,
                                                                    forcing = forcing)
    else:
        layer = 'Conv'
        tile_gen = Tiling(layer,
                          Output_act.shape[1],
                          [Weights.shape[2], Weights.shape[3]],
                          s,
                          [p[0],p[1],p[0],p[1]],
                          groups,
                          [Input_act.shape[1], Input_act.shape[2], Input_act.shape[3]],
                          L1_buffer=L1_dimension,
                          L2_buffer=L2_dim,
                          platform='GAP8',
                          chip=chip,
                          test_location='L2+performance+partial',
                          BitIn=BitIn,
                          BitW=BitW,
                          BitOut=BitOut,
                          BitActivation = BitActivation,
                          optional_type=optional)
    in_dim2, out_dim2, weights_dim, l1_dim2, L3_tiling, factor_ch_out, factor_h_out, factor_h_in = tile_gen.get_tiling(X=np.asarray(Input_compressed),
                                                                Y=np.asarray(Output_compressed),
                                                                W=np.asarray(Weights_compressed),
                                                                relu=relu, BN=BN, DW=DW,
                                                                has_bias=0,
                                                                out_mul=np.uint32(mult[0, 0, 0, 0]),
                                                                out_shift=np.uint32(shift)[0],
                                                                name=name_layer,
                                                                input_L3 = input_L3,
                                                                input_dim_constraint = 400000,
                                                                output_weights_dim_constraint = 400000,
                                                                weight_constraint = 0)
   


def clip8(conv, bits):
    conv[conv >= +(2**(bits) - 1)] = +(2**(bits) - 1)
    conv[conv <= 0] = 0
    out = np.uint8(conv)
    return out


class BatchNorm_DORY_2D(nn.Module):
    def __init__(self, Cin=8, Kh=3, Kw=3, BitA=8, BitW=8, BitO=8, groups=1, inplace=True):
        super(BatchNorm_DORY_2D, self).__init__()
        self.BitO = BitO
        self.k = torch.Tensor(1, Cin, 1, 1).uniform_(0, (2**(8)))
        self.k = torch.round(self.k)
        th = int(
            (2**(BitA + BitW + np.log2(int(Cin / groups) * Kh * Kw) + 8 - 2 - 1)))
        if th > 2**30:
            th = 2**30
        self.l = torch.Tensor(1, Cin, 1, 1).random_(-th, th)
        self.d = torch.Tensor(1).fill_(
            int(BitA + BitW + np.log2(int(Cin / groups) * Kh * Kw) + 8 - BitO))

    def forward(self, input):
        output = input * self.k + self.l
        x = output >> self.d
        out = clip8(x, self.BitO)
        return out

class BatchNorm_DORY_1D(nn.Module):
    def __init__(self, Cin=8, Kh=3, BitA=8, BitW=8, BitO=8, inplace=True):
        super(BatchNorm_DORY_1D, self).__init__()
        self.BitO = BitO
        self.k = torch.Tensor(1, Cin, 1).uniform_(0, (2**(8)))
        self.k = torch.round(self.k)
        th = int(
            (2**(BitA + BitW + np.log2(int(Cin) * Kh) + 8 - 2 - 1)))
        if th > 2**30:
            th = 2**30
        self.l = torch.Tensor(1, Cin, 1).random_(-th, th)
        self.d = torch.Tensor(1).fill_(
            int(BitA + BitW + np.log2(int(Cin)* Kh) + 8 - BitO))

    def forward(self, input):
        output = input * self.k + self.l
        x = output >> self.d
        out = clip8(x, self.BitO)
        return out


def mixed_tests_generator_subbyte(Cin, h, w, Cout, Kh, Kw, p, s, dr, groups, BitA, BitW, BitO):
    # activations
    if w == 1:
        x = torch.Tensor(1, Cin, h).uniform_(0, (2**(BitA + 1)))
    else:
        x = torch.Tensor(1, Cin, h, w).uniform_(0, (2**(BitA + 1)))
    x[x > (2**BitA - 1)] = 0
    x = torch.round(x)
    # network
    if w == 1:
        net = nn.Sequential(nn.Conv1d(Cin, Cout, kernel_size=Kh, stride=s, dilation=dr, padding=p[0], bias=False),
                            BatchNorm_DORY_1D(Cin=Cout, Kh=Kh, BitA=BitA, BitW=BitW, BitO=BitO,))
    else:
        net = nn.Sequential(nn.Conv2d(Cin, Cout, kernel_size=(Kh,Kw), stride=s, padding=p, groups=groups, bias=False),
                        BatchNorm_DORY_2D(Cin=Cout, Kh=Kh, Kw=Kw, BitA=BitA, BitW=BitW, BitO=BitO, groups=groups))
    # weights
    net[0].weight.data.random_(-(2**(BitW - 1)), (2**(BitW - 1)))
    layer = net[0]
    weights = net[0].weight.data
    k = net[1].k
    l = net[1].l
    d = net[1].d

    y = net(x)
    y_no_quant = layer(x)

    return x, weights, y, k, l, d


def main():
    """
    Create an instance of ONNX_management.
    If it is a test, create the onnx model.
    Then, extract infos and generate all the application folder.
    Give the folder on which you have 1 onnx file and the intermediate results
    """
    # ch_in, h, w, ch_out
    # dim = [512, 8, 8, 512]
    # dim = [3, 128, 128, 32]
    # dim = [32, 64, 64, 32]
    from_file = 0
    if from_file == 1:
        df = pd.read_excel('../../TCN_PULP/Paper_results.xlsx', sheet_name = 'New_figure_DAC')


        parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
        parser.add_argument('--layer',  type=int, default = 0)
        args = parser.parse_args()
        #layer = 11
        row = df[df['layer'] == f'l{args.layer}']
        dim = [int(row['chin'].values[0]), int(row['out_dim'].values[0]), 1, int(row['chout'].values[0])]
        # dim = [256, 16, 16, 256]
        dr = int(row['dil'].values[0])
        fs = [int((row['ker_dim'].values[0]-1)*dr+1), 1]
        p = (int(row['padding'].values[0]),0)
        s = 1#int(row['stride'].values[0])
        groups = 1
        BitA = 8
        BitW = 8
        BitO = 8
    else:
        dim = [32, 32, 32, 32]
        fs = [3, 3]
        # bits encoding activation and weights
        BitA = 8
        BitW = 8
        BitO = 8
        p = (1,1)
        s = 1
        groups = 1
        dr = 1
        forcing = 'None'
        # dim = [32, 32, 32, 32]
        # fs = [3, 3]
        # # bits encoding activation and weights
        # BitA = 8
        # BitW = 8
        # BitO = 8
        # p = (1,1)
        # s = 1
        # groups = 1
        # dr = 1
        # forcing = 'None'
    forcing = 'None'   
    # forcing = 'indirect'
    # forcing = 'nodilation'
    # forcing = 'normal'
    input_L3 = 0
    if dim[2]==1:
        name = '1D-8bit'
    else:
        name = '8bit'

    chip = 'GAP8v3'
    torch.manual_seed(3)
    import random
    x, weights, y, k, l, d = mixed_tests_generator_subbyte(
        dim[0], dim[1], dim[2], dim[3], fs[0], fs[1], p, s, dr, groups, BitA, BitW, BitO)
    # Extract the parameters from the onnx model
    print_model_layer(chip, name, Input_act=x, Weights=weights, Output_act=y, mult=k,
                      summ=l, shift=d, L1_dim=44000,  L2_dim=400000,  s=s, p=p, dilation = dr, groups=groups, BitIn=BitA, BitW=BitW, BitOut=BitO, BitActivation = 32, input_L3 = input_L3, forcing = forcing)

if __name__ == '__main__':
    main()
