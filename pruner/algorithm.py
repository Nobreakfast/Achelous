import torch
import torch.nn as nn

import tqdm
from thop import profile, clever_format
from torch import profiler

from .backward2node import __backward2node, __get_groups
from .node import ConvNode


@torch.no_grad()
def test_speed(model, example_input, epoch=30, device="cpu"):
    epoch = epoch
    for i in range(epoch):
        model(*example_input)
    import time

    activity = (
        profiler.ProfilerActivity.CPU
        if device == "cpu"
        else profiler.ProfilerActivity.CUDA
    )
    sort_by = "cpu_time_total" if device == "cpu" else "cuda_time_total"
    with profiler.profile(activities=[activity], record_shapes=True) as prof:
        with profiler.record_function("model_inference"):
            model(*example_input)
    print(prof.key_averages().table(sort_by=sort_by, row_limit=20))
    start = time.time()
    for i in tqdm.trange(epoch):
        model(*example_input)
    end = time.time()
    inf_time = (end - start) / epoch / 1e-3
    print(f"time: {inf_time} ms, FPS: {1000/inf_time}")


def __get_flops(model, example_input, device):
    model.to(device)
    if isinstance(example_input, torch.Tensor):
        example_input = [example_input.to(device)]
    elif isinstance(example_input, dict):
        example_input = [v.to(device) for v in example_input.values()]
    elif isinstance(example_input, (list, tuple)):
        example_input = [v.to(device) for v in example_input]
    flops, params = profile(model, inputs=example_input)
    print(clever_format([flops, params], "%.3f"))
    return flops, params


def prune_model(
    model,
    example_input,
    sparsity=0.7,
    algorithm="uniform",
    device="cpu",
    imt_dict={},
    imk=[],
):
    device = torch.device(device)
    model.eval()
    print("=" * 20, "Original Model", "=" * 20)
    __get_flops(model, example_input, device)
    print("=" * 20, "Original Model", "=" * 20)

    node_dict, ignore_nodes = __backward2node(model, example_input, imt_dict)
    imk.extend([n.name for n in ignore_nodes])
    groups = __get_groups(node_dict)
    prune_fn = globals()[algorithm]
    prune_fn(groups, sparsity, imk)

    groups_hascat = []
    for g in groups:
        if g.hascat():
            groups_hascat.append(g)
            continue
        if g.nodes[0].name == "image_radar_encoder.fpn.spp.cv1.conv":
            print("hello")
        g.prune(g.sparsity)
    for g in groups_hascat:
        g.prune(g.sparsity)

    for node in node_dict.values():
        node.execute()

    print("=" * 20, "Pruned Model", "=" * 20)
    __get_flops(model, example_input, device)
    print("=" * 20, "Pruned Model", "=" * 20)


def uniform(groups, sparsity, imk):
    for g in groups:
        if g.haskey(imk):
            continue
        g.sparsity = sparsity


def erk(groups, sparsity, imk):
    for g in groups:
        if g.haskey(imk):
            continue
        erk_list = []
        for n in g.nodes:
            if isinstance(n, ConvNode):
                erk = 1 - (
                    n.in_ch
                    + n.out_ch
                    + n.module.kernel_size[0]
                    + n.module.kernel_size[1]
                ) / (
                    n.in_ch
                    * n.out_ch
                    * n.module.kernel_size[0]
                    * n.module.kernel_size[1]
                )
            else:
                erk = 1 - (n.in_ch + n.out_ch) / (n.in_ch * n.out_ch)
            erk_list.append(erk)
        g.sparsity = sparsity * min(erk_list)


def featio(groups, sparsity, imk):
    for g in groups:
        if g.haskey(imk):
            continue
        featio_list = [1]
        for n in g.nodes:
            if not hasattr(n.module, "weight"):
                continue
            if isinstance(n, ConvNode):
                strides = n.module.stride
                featio = (n.in_ch * strides[0] * strides[1]) / (
                    n.out_ch * n.module.weight.data.nelement()
                )
            else:
                featio = (n.in_ch) / (n.out_ch * n.module.weight.data.nelement())
            featio_list.append(1 - 1 / featio)
        g.sparsity = sparsity * min(featio_list)
