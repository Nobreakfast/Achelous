import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence
from pruner.structure import *
import backbone.attention_modules.shuffle_attention as sa
import math
from thop import profile
import backbone.vision.mobilevit_modules.mobilevit as mv
import backbone.radar.RadarEncoder as re

from nets.Achelous import Achelous3T
from torchvision.models import resnet18

from pruner.utils import *
import torch_pruning as tp

import backbone.attention_modules.shuffle_attention as sa


def test2():
    # import torch
    # from torchvision.models import resnet18

    # model = resnet18(pretrained=True)
    model = Achelous3T(
        resolution=320,
        num_det=7,
        num_seg=9,
        phi="S0",
        backbone="mv",
        neck="gdf",
        spp=True,
        nano_head=False,
    )
    for k, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            print(k, m)
    example_inputs = [torch.randn(2, 3, 320, 320), torch.randn(2, 3, 320, 320)]
    example_inputs2 = {
        "x": torch.randn(2, 3, 320, 320),
        "x_radar": torch.randn(2, 3, 320, 320),
    }

    # Importance criteria
    # example_inputs = torch.randn(1, 3, 224, 224)
    imp = tp.importance.TaylorImportance()

    ignored_layers = []
    ignored_layers_type = [sa.ShuffleAttention]
    for m in model.modules():
        if isinstance(m, tuple(ignored_layers_type)):
            ignored_layers.append(m)

    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 1000:
            ignored_layers.append(m)  # DO NOT prune the final classifier!

    iterative_steps = 5  # progressive pruning
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs2,
        importance=imp,
        iterative_steps=iterative_steps,
        ch_sparsity=0.5,  # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        ignored_layers=ignored_layers,
    )

    def __sum(x: list):
        sum = 0.0
        for i in x:
            try:
                sum += i.sum()
            except:
                sum += __sum(i)
        return sum

    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    for i in range(iterative_steps):
        if isinstance(imp, tp.importance.TaylorImportance):
            # Taylor expansion requires gradients for importance estimation
            output = model(**example_inputs2)
            outsum = __sum(output)
            loss = outsum.sum()
            loss.backward()  # before pruner.step()
        pruner.step()
        macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print(
            f"[Step {i+1}/{iterative_steps}] MACs: {macs/base_macs:.3f}, Params: {nparams/base_nparams:.3f}"
        )
        # finetune your model here
        # finetune(model)
        # ...


def __print_shape(output):
    if isinstance(output, torch.Tensor):
        print(output.shape)
    else:
        for i in output:
            __print_shape(i)


def test1():
    model = Achelous3T(
        resolution=320,
        num_det=7,
        num_seg=9,
        phi="S0",
        backbone="mv",
        neck="gdf",
        spp=True,
        nano_head=False,
    )
    # for k, m in model.named_modules():
    #     if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
    #         print(k, m)
    example_input = [torch.randn(2, 3, 320, 320), torch.randn(2, 3, 320, 320)]
    # model = resnet18()
    # example_input = (torch.randn(2, 3, 224, 224),)

    output = model(*list(example_input))
    __print_shape(output)
    macs, params = profile(model, inputs=example_input)
    print(
        "macs:",
        macs,
        "params:",
        params,
    )

    ignore_Block = [sa.ShuffleAttention]
    tl = [
        nn.Conv2d,
        nn.Linear,
        nn.BatchNorm2d,
        nn.LayerNorm,
        sa.ShuffleAttention,
        # nn.Conv1d,
        # nn.Upsample,
    ]

    il = [[], []]  # ignore_list
    for k, m in model.named_modules():
        if isinstance(m, mv.Attention):
            il[0].append(k + ".to_qkv")
            if m.to_out[0]:
                il[1].append(k + ".to_out.0")
        if isinstance(m, tuple(ignore_Block)):
            il[0].append(k + ".gn")
            il[1].append(k + ".gn")
        if k[:45] == "image_radar_encoder.radar_encoder.rc_blocks.0" and isinstance(
            m, (nn.Conv2d, nn.BatchNorm2d)
        ):
            il[0].append(k)
            il[1].append(k)

    model_new = prune_model(
        model,
        ratio=0.7,
        example_input=example_input,
        type_list=tl,
        ignore_list=il,
        debug=False,
    )

    # for k, m in model.named_modules():
    #     if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
    #         print(k, m)
    output = model_new(*list(example_input))
    macs, params = profile(model_new, inputs=example_input)
    print(
        "macs:",
        macs,
        "params:",
        params,
    )

    __print_shape(output)
    # print(model_new)
    return model_new


if __name__ == "__main__":
    test1()
