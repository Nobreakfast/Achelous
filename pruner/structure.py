import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

class Pruner:
    def __init__(self, model) -> None:
        pass


def structure_conv(conv: nn.Module, index: list, dim: int)->nn.Module:
    """
    Prune the Convolution Layer:
    return a new conv layer with pruned weight
    :param conv: nn.Conv2d
    :param index: list of index to be pruned
    :param dim: 0 for prune output channel, 1 for prune input channel
    :return new_conv: nn.Conv2d
    """
    assert dim in [0, 1], "dim must be 0 or 1"
    weight_shape = list(conv.weight.shape)
    # invert the index
    saved_index = [i for i in range(weight_shape[dim]) if i not in index]
    weight_shape[dim] -= len(index)
    new_conv = nn.Conv2d(weight_shape[1], weight_shape[0], weight_shape[2:], stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups, bias=conv.bias is not None)
    new_conv.weight = nn.Parameter(torch.index_select(conv.weight, dim, torch.tensor(saved_index)))
    if conv.bias is not None:
        new_conv.bias = nn.Parameter(torch.index_select(conv.bias, dim, torch.tensor(saved_index)))
    return new_conv

def structure_fc(fc: nn.Module, index: list, dim: int)->nn.Module:
    """
    Prune the Fully Connected Layer:
    return a new fc layer with pruned weight
    :param fc: nn.Linear
    :param index: list of index to be pruned
    :param dim: 0 for prune output channel, 1 for prune input channel
    :return new_fc: nn.Linear
    """
    assert dim in [0, 1], "dim must be 0 or 1"
    weight_shape = list(fc.weight.shape)
    # invert the index
    saved_index = [i for i in range(weight_shape[dim]) if i not in index]
    weight_shape[dim] -= len(index)
    new_fc = nn.Linear(weight_shape[1], weight_shape[0], bias=fc.bias is not None)
    new_fc.weight = nn.Parameter(torch.index_select(fc.weight, dim, torch.tensor(saved_index)))
    if fc.bias is not None:
        new_fc.bias = nn.Parameter(torch.index_select(fc.bias, dim, torch.tensor(saved_index)))
    return new_fc

if __name__ == "__main__":
    conv = nn.Conv2d(3, 6, 3, bias=False)
    print(conv.weight.shape)
    print(conv.weight[2,0,0,0])
    print(conv.weight[3,0,0,0])
    new_conv = structure_conv(conv, [0, 1], 0)
    print(new_conv.weight.shape)
    print(new_conv.weight[0,0,0,0])
    print(new_conv.weight[1,0,0,0])
