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

from nets.Achelous import Achelous3T
from torchvision.models import resnet18

from pruner.utils import *

if __name__ == "__main__":
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
    example_input = [torch.randn(2, 3, 320, 320), torch.randn(2, 3, 320, 320)]
    # model = resnet18()
    # example_input = (torch.randn(2, 3, 224, 224),)

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
        nn.Upsample,
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

    model_new = prune_model(
        model,
        ratio=0.5,
        example_input=example_input,
        type_list=tl,
        ignore_list=il,
        debug=True,
    )
    # for k, m in model_new.named_modules():
    #     if isinstance(m, tl):
    #         print(k, m)

    output = model_new(*list(example_input))
    macs, params = profile(model_new, inputs=example_input)
    print(
        "macs:",
        macs,
        "params:",
        params,
    )
    try:
        print(output.shape)
    except:
        for i in output:
            print(i.shape)
    # print(model_new)
