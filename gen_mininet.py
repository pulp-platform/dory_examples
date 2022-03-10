import os
import quantlib.backends as qb
import typing
from typing import Union, Tuple, List, Dict
import torch
import torch.nn as nn
import argparse
from argparse import RawTextHelpFormatter
import quantlib.editing.lightweight as qlw

class MiniNet(nn.Module):
    def __init__(self, Kin: int, Kout: int, Fs: int, Pad: int) -> None:
        super(MiniNet, self).__init__()
        self.lay = nn.Conv2d(Kin, Kout, kernel_size=Fs, padding=Pad, bias=False)
        self.bn  = nn.BatchNorm2d(Kout)
        self.act = nn.ReLU(inplace=True)
        self.lay2 = nn.Conv2d(Kout, 1, stride=2, kernel_size=1, padding=0, bias=False)
        self.bn2  = nn.BatchNorm2d(1)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lay(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.lay2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

def get_intermediate_activations(net, test_fn, *test_args, **test_kwargs):
    l = len(list(net.named_modules()))
    buffer_in  = OrderedDict([])
    buffer_out = OrderedDict([])
    hooks = OrderedDict([])
    def get_hk(n):
        def hk(module, input, output):
            buffer_in  [n] = input
            buffer_out [n] = output
        return hk
    for i,(n,l) in enumerate(net.named_modules()):
        hk = get_hk(n)
        hooks[n] = l.register_forward_hook(hk)
    ret = test_fn(*test_args, **test_kwargs)
    for n,l in net.named_modules():
        hooks[n].remove()
    return buffer_in, buffer_out, ret

def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--Kin', type=int, default=16, help='No. of input channels')
    parser.add_argument('--Kout', type=int, default=32, help='No. of output channels')
    parser.add_argument('--Hin', type=int, default=3, help='Height of input')
    parser.add_argument('--Win', type=int, default=3, help='Width of layer')
    parser.add_argument('--Fs', type=int, default=1, help='Filter size')
    parser.add_argument('--Pad', type=int, default=0, help='Padding')
    args = parser.parse_args()

    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')

    # create the network
    network = MiniNet(Kin=args.Kin, Kout=args.Kout, Fs=args.Fs, Pad=args.Pad)
    network = network.to(device=device)  # REMEMBER: place the parameters of the 'Module' on the device that guarantees the best performance
    print(network)
    print()

    # verify that it can process tensor data
    network.eval()  # REMEMBER: before evaluating a network, freeze the batch-normalisation and dropout parameters
    dummy_x = torch.randn(1, args.Kin, args.Hin, args.Win).to(device=device)
    dummy_y = network(dummy_x)

    def all_pact_f2f_recipe(network: nn.Module, name2config: Dict[str, Dict]) -> nn.Module:

        lwg = qlw.LightweightGraph(network)
        name2type = {n.name: n.module.__class__.__name__ for n in lwg.nodes_list}

        # generate lightweight (i.e., atomic) replacement rules
        assert set(name2config.keys()).issubset(set(name2type.keys()))
        type2rule = \
        {
            'Conv2d': qlw.rules.pact.ReplaceConvLinearPACTRule,
            'Linear': qlw.rules.pact.ReplaceConvLinearPACTRule,
            'ReLU':   qlw.rules.pact.ReplaceActPACTRule
        }
        rhos = list(map(lambda n: type2rule[name2type[n]](qlw.rules.NameFilter(n), **name2config[n]), name2config.keys()))
        
        # boot lightweight editor and apply atomic rules
        lwe = qlw.LightweightEditor(lwg)
        lwe.startup()
        for rho in rhos:
            lwe.set_lwr(rho)
            lwe.apply()
        lwe.shutdown()
        
        return lwe.graph.net

    """To simplify the exploration of different quantisation policies, we aim at defining the configurations of quantised nodes in a programmatic way.

    """

    from collections import defaultdict


    def all_pact_create_configs(network: nn.Module, patches: Dict[str, Dict]) -> Dict[str, Dict]:

        lwg = qlw.LightweightGraph(network)
        conv2d_nodes = set([n.name for n in lwg.nodes_list if n.module.__class__.__name__ == 'Conv2d'])
        linear_nodes = set([n.name for n in lwg.nodes_list if n.module.__class__.__name__ == 'Linear'])
        relu_nodes   = set([n.name for n in lwg.nodes_list if n.module.__class__.__name__ == 'ReLU'])
        assert set(patches.keys()).issubset(conv2d_nodes | linear_nodes | relu_nodes)

        # configure convolutional nodes
        conv2d_default = \
        {
            'quantize':   'per_channel',
            'init_clip':  'sawb_asymm',
            'learn_clip': False,
            'symm_wts':   True,
            'tqt':        False,
            'n_levels':   256
        }

        conv2d_config = defaultdict(lambda: conv2d_default.copy())  # it is EXTREMELY important that we return a copy of the dictionary and not the dictionary itself; otherwise, all configs would point to the same (possibly updated) object
        for n in conv2d_nodes:
            conv2d_config[n].update(patches[n] if n in patches.keys() else {})  # patches have higher priority than default configurations

        # configure linear nodes
        linear_default = \
        {
            'quantize':   'per_layer',
            'init_clip':  'sawb_asymm',
            'learn_clip': False,
            'symm_wts':   True,
            'tqt':        False,
            'n_levels':   256
        }

        linear_config = defaultdict(lambda: linear_default.copy())
        for n in linear_nodes:
            linear_config[n].update(patches[n] if n in patches.keys() else {})  # patches have higher priority than default configurations

        # configure ReLU nodes
        relu_default = \
        {
            'init_clip':  'std',
            'learn_clip': True,
            'nb_std':     3,
            'rounding':   False,
            'tqt':        False,
            'n_levels':   256
        }

        relu_config = defaultdict(lambda: relu_default.copy())
        for n in relu_nodes:
            relu_config[n].update(patches[n] if n in patches.keys() else {})  # patches have higher priority than default configurations

        # create complete configuration
        config = {**conv2d_config, **linear_config, **relu_config}

        return config

    # create configuration for PACT float-to-fake conversion
    name2config = all_pact_create_configs(network, {})

    # apply PACT float-to-fake conversion
    pact_network = all_pact_f2f_recipe(network, name2config)
    pact_network.to(device=device)

    import quantlib.editing.fx as qfx


    def f2t_convert(dataloader: torch.utils.data.DataLoader,
                    input_eps:  float,
                    network:    nn.Module) -> nn.Module:

        network.eval()
        network = network.to(device=torch.device('cpu'))
        
        x = torch.randn(args.Kin, args.Hin, args.Win).to(device=device)
        x  = x.unsqueeze(0)
        
        fake2true_converter = qfx.passes.pact.IntegerizePACTNetPass(shape_in=x.shape, eps_in=input_eps, D=2**12)
        
        return fake2true_converter(pact_network)

    tq_pact_network = f2t_convert(None, 1., pact_network)

    x = torch.randn(args.Kin, args.Hin, args.Win).to(device='cpu')

    x = x.unsqueeze(0)
    y_pr = tq_pact_network(x)
    y_pr_int = y_pr.argmax(axis=1)

    def export_network(data_loader: torch.utils.data.DataLoader,
                    input_eps:   float,
                    network:     nn.Module,
                    filename:    str,
                    dir_export:  os.PathLike) -> None:
        
        network.eval()
        
        x = torch.FloatTensor(args.Kin, args.Hin, args.Win).uniform_(0, 256)
        # randn(args.Kin, args.Hin, args.Win)
        # x = torch.zeros(args.Kin, args.Hin, args.Win)
        # x = 255-torch.arange(0,144).reshape(args.Kin, args.Hin, args.Win)
        x = x.unsqueeze(0).floor()
        
        qb.dory.export_net(network,
                        name=filename,
                        out_dir=dir_export,
                        eps_in=input_eps,
                        integerize=False,
                        D=2**12,
                        in_data=x)

    acts = []
    def dump_hook(self, inp, outp, name):
        # DORY wants HWC tensors
        acts.append((name, outp[0]))

    ls_modules = list(tq_pact_network.modules())
    ls_modules[1].weight.data[:] = torch.FloatTensor(*ls_modules[1].weight.data.shape).uniform_(-32,32).floor()
    ls_modules[3].mul.data[:] = 32
    
    # ls_modules[1].weight.data[1:,:] = 0
    # ls_modules[1].weight.data[:,2:] = 0
    # ls_modules[1].weight.data[0,:,0,0] = +127

    export_network(None, 1., tq_pact_network, 'MiniNet', 'MiniNet')

if __name__ == '__main__':
    main()
