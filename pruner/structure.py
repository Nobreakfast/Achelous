import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import backbone.attention_modules.shuffle_attention as sa


class Pruner:
    def __init__(self, model) -> None:
        pass


def structure_conv(conv: nn.Module, index: list, dim: int) -> nn.Module:
    """
    Prune the Convolution Layer:
    return a new conv layer with pruned weight
    :param conv: nn.Conv2d
    :param index: list of index to be pruned
    :param dim: 0 for prune output channel, 1 for prune input channel
    :return new_conv: nn.Conv2d
    """
    assert dim in [0, 1], "dim must be 0 or 1"
    # print(len(index), index)
    if len(index) == 0:
        # print("index is empty")
        return conv
    chs = [conv.out_channels, conv.in_channels]
    weight_shape = list(conv.weight.shape)
    # invert the index
    saved_index = [i for i in range(chs[dim]) if i not in index]
    chs[dim] -= len(index)
    new_conv = nn.Conv2d(
        chs[1],
        chs[0],
        weight_shape[2:],
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
    )
    new_conv.weight = nn.Parameter(
        torch.index_select(conv.weight, dim, torch.tensor(saved_index))
    )
    if conv.bias is not None:
        if dim == 0:
            # new_conv.bias = nn.Parameter(torch.index_select(conv.bias, dim, torch.tensor(saved_index)))
            new_conv.bias.data = conv.bias[saved_index]
        else:
            new_conv.bias.data = conv.bias.data
    # print("new_conv", new_conv)
    return new_conv

def structure_group_conv(conv: nn.Module, index: list) -> nn.Module:
    """
    Prune the Convolution Layer:
    return a new conv layer with pruned weight
    :param conv: nn.Conv2d
    :param index: list of index to be pruned
    :param dim: 0 for prune output channel, 1 for prune input channel
    :return new_conv: nn.Conv2d
    """
    # print(len(index), index)
    if len(index) == 0:
        # print("index is empty")
        return conv
    chs = conv.out_channels
    weight_shape = list(conv.weight.shape)
    # invert the index
    saved_index = [i for i in range(chs) if i not in index]
    chs -= len(index)
    new_conv = nn.Conv2d(
        chs,
        chs,
        weight_shape[2:],
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=chs,
        bias=conv.bias is not None,
    )
    new_conv.weight = nn.Parameter(
        torch.index_select(conv.weight, 0, torch.tensor(saved_index))
    )
    if conv.bias is not None:
        # new_conv.bias = nn.Parameter(torch.index_select(conv.bias, dim, torch.tensor(saved_index)))
        new_conv.bias.data = conv.bias[saved_index]
    # print("new_conv", new_conv)
    return new_conv

def structure_fc(fc: nn.Module, index: list, dim: int) -> nn.Module:
    """
    Prune the Fully Connected Layer:
    return a new fc layer with pruned weight
    :param fc: nn.Linear
    :param index: list of index to be pruned
    :param dim: 0 for prune output channel, 1 for prune input channel
    :return new_fc: nn.Linear
    """
    assert dim in [0, 1], "dim must be 0 or 1"
    if len(index) == 0:
        return fc
    weight_shape = list(fc.weight.shape)
    # invert the index
    saved_index = [i for i in range(weight_shape[dim]) if i not in index]
    weight_shape[dim] -= len(index)
    new_fc = nn.Linear(weight_shape[1], weight_shape[0], bias=fc.bias is not None)
    new_fc.weight.data = nn.Parameter(
        torch.index_select(fc.weight, dim, torch.tensor(saved_index))
    )
    if fc.bias is not None:
        if dim == 0:
            # new_fc.bias = nn.Parameter(torch.index_select(fc.bias, dim, torch.tensor(saved_index)))
            new_fc.bias.data = fc.bias[saved_index]
        else:
            new_fc.bias.data = fc.bias.data
    return new_fc


def structure_bn(bn: nn.Module, index):
    """
    Prune the BatchNorm Layer:
    return a new bn layer with pruned weight
    :param bn: nn.BatchNorm2d
    :param index: list of index to be pruned
    :return new_bn: nn.BatchNorm2d
    """
    if len(index) == 0:
        return bn
    weight_shape = list(bn.weight.shape)
    saved_index = [i for i in range(weight_shape[0]) if i not in index]
    weight_shape[0] -= len(index)
    new_bn = nn.BatchNorm2d(
        weight_shape[0], bn.eps, bn.momentum, bn.affine, bn.track_running_stats
    )
    new_bn.weight.data = bn.weight.data[saved_index]
    if bn.bias is not None:
        new_bn.bias.data = bn.bias.data[saved_index]
    # print("new_bn", new_bn)
    return new_bn


def structure_ln(ln: nn.Module, index):
    """
    Prune the LayerNorm Layer:
    return a new ln layer with pruned weight
    :param ln: nn.LayerNorm
    :param index: list of index to be pruned
    :return new_ln: nn.LayerNorm
    """
    if len(index) == 0:
        return ln
    weight_shape = list(ln.weight.shape)
    saved_index = [i for i in range(weight_shape[0]) if i not in index]
    weight_shape[0] -= len(index)
    new_ln = nn.LayerNorm(weight_shape[0], ln.eps, ln.elementwise_affine)
    new_ln.weight.data = ln.weight.data[saved_index]
    if ln.bias is not None:
        new_ln.bias.data = ln.bias.data[saved_index]
    return new_ln


def structure_gn(gn: nn.Module, index):
    """
    Prune the GroupNorm Layer:
    return a new gn layer with pruned weight
    :param gn: nn.GroupNorm
    :param index: list of index to be pruned
    :return new_gn: nn.GroupNorm
    """
    if len(index) == 0:
        return gn
    weight_shape = list(gn.weight.shape)
    saved_index = [i for i in range(weight_shape[0]) if i not in index]
    weight_shape[0] -= len(index)
    new_gn = nn.GroupNorm(weight_shape[0], weight_shape[0])
    new_gn.weight.data = gn.weight.data[saved_index]
    if gn.bias is not None:
        new_gn.bias.data = gn.bias.data[saved_index]
    return new_gn


def structure_shuffleAttn(shuffleAttn: sa.ShuffleAttention, index):
    """
    Prune the ShuffleAttn Layer:
    return a new shuffleAttn layer with pruned weight
    :param shuffleAttn: nn.ShuffleAttn
    :param index: list of index to be pruned
    :return new_shuffleAttn: nn.ShuffleAttn
    """
    if len(index) == 0:
        return shuffleAttn
    in_channels = shuffleAttn.channel
    saved_index = [i for i in range(in_channels) if i not in index]
    saved_index = [i // shuffleAttn.groups for i in saved_index]
    saved_index = list(set(saved_index))[: len(saved_index) // shuffleAttn.groups]
    channels = in_channels - len(index)
    new_shuffleAttn = sa.ShuffleAttention(channels, G=8)
    fake_index = [i for i in range(shuffleAttn.gn.num_groups) if i not in saved_index]
    setattr(
        new_shuffleAttn,
        "gn",
        structure_gn(shuffleAttn.gn, fake_index),
    )
    new_shuffleAttn.cweight.data = shuffleAttn.cweight.data[:, saved_index, :, :]
    new_shuffleAttn.cbias.data = shuffleAttn.cbias.data[:, saved_index, :, :]
    new_shuffleAttn.sweight.data = shuffleAttn.sweight.data[:, saved_index, :, :]
    new_shuffleAttn.sbias.data = shuffleAttn.sbias.data[:, saved_index, :, :]
    return new_shuffleAttn


if __name__ == "__main__":
    conv = nn.Conv2d(3, 6, 3, bias=False)
    print(conv.weight.shape)
    print(conv.weight[2, 0, 0, 0])
    print(conv.weight[3, 0, 0, 0])
    new_conv = structure_conv(conv, [0, 1], 0)
    print(new_conv.weight.shape)
    print(new_conv.weight[0, 0, 0, 0])
    print(new_conv.weight[1, 0, 0, 0])
