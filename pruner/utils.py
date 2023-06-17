import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence

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
    pass
