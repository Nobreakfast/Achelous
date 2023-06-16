import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_pruning as tp
from typing import Sequence
from backbone.attention_modules.shuffle_attention import ShuffleAttention

class ShuffleAttnPruner(tp.pruner.BasePruningFunc):
    def prune_out_channels(self, layer: ShuffleAttention, idxs: Sequence[int]) -> nn.Module: 
        # keep_idxs = list(set(range(layer.in_dim)) - set(idxs))
        # keep_idxs.sort()
        # layer.in_dim = layer.in_dim-len(idxs)
        # layer.scale = torch.nn.Parameter(layer.scale.data.clone()[keep_idxs])
        # layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
        # tp.prune_linear_in_channels(layer.fc, idxs)
        # tp.prune_linear_out_channels(layer.fc, idxs)
        return layer

    def get_out_channels(self):
        return 0
        # return self.in_dim
    
    # identical functions
    prune_in_channels = prune_out_channels
    get_in_channels = get_out_channels

def get_sparsity(model, amount):
    """
    Get the sparsity dict for a model.
    """
    sparsity_dict = {}
    sum = 0.0
    for k, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            # print("Conv2d layer", k)
            in_ch = m.in_channels
            out_ch = m.out_channels
            strides = m.stride
            featio = out_ch/strides[0]/strides[1]/in_ch/(torch.sum(m.weight.data).item()+1e-8)
        elif isinstance(m, nn.Linear):
            # print("Linear layer", k)
            in_ch = m.in_features
            out_ch = m.out_features
            featio = out_ch/in_ch/(torch.sum(m.weight.data).item()+1e-8)
        else: 
            continue
        sparsity_dict[k] = 1/featio
        sum += sparsity_dict[k]
    for k in sparsity_dict.keys():
        sparsity_dict[k] = amount*(1-sparsity_dict[k]/sum)
    return sparsity_dict

def prune_model(model, sparsity_dict):
    """
    Prune the model according to the sparsity dict.
    """
    print("Building Dependency Graph...")
    example_inputs = {'x': torch.randn((1, 3, 320, 320)), 'x_radar': torch.randn((1, 3, 320, 320))}
    # ignore_layers = [k for k, m in model.named_modules() if hasattr(m, 'cweight')]
    DG = tp.DependencyGraph()
    DG.register_customized_layer(ShuffleAttention, ShuffleAttnPruner)
    DG.build_dependency(model, example_inputs=example_inputs)
    print("Building L1 Norm...")
    for k, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            print(f"Processing Conv layer {k}")
            amount = sparsity_dict[k]
            # if m.kernel_size == (1,1):
            #     continue
            # calculate the l1 norm for weight dim 0
            weight = m.weight.data.cpu().numpy()
            weight = np.sum(np.abs(weight), axis=(1,2,3))
            # get the index of the smallest "amount" weight
            index = np.argsort(weight)[:int(amount*len(weight))]
            # prune the weight
            try:
                group = DG.get_pruning_group( m, tp.prune_conv_out_channels, idxs=index )
            except:
                print("error conv")
                continue
        # elif isinstance(m, nn.Linear):
        #     print(f"Processing Linear layer {k}")
        #     amount = sparsity_dict[k]
        #     # calculate the l1 norm for weight dim 0
        #     weight = m.weight.data.cpu().numpy()
        #     weight = np.sum(np.abs(weight), axis=1)
        #     # get the index of the smallest "amount" weight
        #     index = np.argsort(weight)[:int(amount*len(weight))]
        #     # prune the weight
        #     try:
        #         group = DG.get_pruning_group( m, tp.prune_linear_out_channels, idxs=index )
        #     except:
        #         print("error linear")
        #         continue
        else: 
            continue
        if DG.check_pruning_group(group): 
            group.prune()