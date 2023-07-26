import torch
import torch.nn as nn

import tqdm
from tqdm import trange
from thop import profile, clever_format
from torch import profiler

from .backward2node import __backward2node, __get_groups
from .node import InOutNode, CustomNode, ConvNode, LinearNode


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
    groups_haspool = []
    for g in groups:
        if g.hascat():
            groups_hascat.append(g)
            continue
        if g.haspool():
            groups_haspool.append(g)
            continue
        # for n in g.nodes:
        #     if n.name in [
        #         "image_radar_encoder.fpn.spp.cv1.conv",
        #         "image_radar_encoder.fpn.spp.cv2.conv",
        #         "image_radar_encoder.fpn.spp.m.0",
        #     ]:
        #         print("hello")
        g.prune(g.sparsity)
    for g in groups_haspool:
        # for n in g.nodes:
        #     if n.name in [
        #         "image_radar_encoder.fpn.spp.cv1.conv",
        #         "image_radar_encoder.fpn.spp.cv2.conv",
        #         "image_radar_encoder.fpn.spp.m.0",
        #     ]:
        #         print("hello")
        g.prune(g.sparsity)
    for g in groups_hascat:
        # for n in g.nodes:
        #     if n.name in ["image_radar_encoder.fpn.spp.cv2.conv.CatB"]:
        #         print("hello")
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
    """
    erk = 1 - (in_ch + out_ch + k_h + k_w) / (in_ch * out_ch * k_h * k_w) if convolution
    erk = 1 - (in_ch + out_ch) / (in_ch * out_ch) if linear
    """
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
    """
    featio = (in_ch * strides[0] * strides[1]) / (out_ch * nonzeros_count)
    featio = sum(in_ch * strides[0] * strides[1] / out_ch) / sum(nonzeros_count)
    """
    score_dict = {}
    for g in groups:
        if g.haskey(imk):
            continue
        score_dict[g] = {}
        score_dict[g]["score"] = torch.tensor([])
        score_dict[g]["pruned_out_ch"] = 0
        feati = 0
        feato = 0
        N = 0
        score_dict[g]["out_ch"] = g.channel - score_dict[g]["pruned_out_ch"]
        for n in g.nodes:
            if isinstance(n, ConvNode):
                feati += n.in_ch * n.module.stride[0] * n.module.stride[1]
                feato += score_dict[g]["out_ch"]
                N += (
                    n.in_ch
                    * score_dict[g]["out_ch"]
                    * n.module.kernel_size[0]
                    * n.module.kernel_size[1]
                )
            elif isinstance(n, LinearNode):
                feati += n.in_ch
                feato += score_dict[g]["out_ch"]
                N += n.in_ch * score_dict[g]["out_ch"]
            else:
                continue
        if N == 0:
            g.sparsity = sparsity
            score_dict.pop(g)
            continue
        score_dict[g]["N"] = N
        featio = feati / feato / N
        score_dict[g]["score"] = featio * torch.randn(N).abs()
    score_list = torch.cat(
        tuple([score_dict[key]["score"] for key in score_dict.keys()]), dim=0
    )
    number = int(sparsity * len(score_list))
    kth = torch.kthvalue(score_list, number)[0]
    for key in score_dict.keys():
        score_dict[key]["pruned_count"] = torch.sum(score_dict[key]["score"] < kth)
        score_dict[key]["pruned_out_ch"] = int(
            score_dict[key]["out_ch"]
            * (score_dict[key]["pruned_count"] / score_dict[key]["N"])
        )
        key.sparsity = score_dict[key]["pruned_count"] / score_dict[key]["N"]


def featio50(groups, sparsity, imk):
    epoch = 50
    # init score_dict
    score_dict = {}
    num_all = 0
    for g in groups:
        if g.haskey(imk):
            continue
        score_dict[g] = {}
        score_dict[g]["pruned_out_ch"] = 0
        N = 0
        for n in g.nodes:
            if isinstance(n, InOutNode):
                if isinstance(n, CustomNode):
                    N += n.prunable_weight_count()
                else:
                    N += n.module.weight.numel()
        if N == 0:
            g.sparsity = sparsity
            score_dict.pop(g)
            continue
        else:
            num_all += N

    num_toprune = int(sparsity * num_all)
    num_toprune_each = int(num_toprune / epoch)
    # iterative featio
    for i in trange(epoch):
        for g in score_dict.keys():
            score_dict[g]["score"] = torch.tensor([])
            feati = 0
            feato = 0
            N = 0
            score_dict[g]["out_ch"] = g.channel - score_dict[g]["pruned_out_ch"]
            for n in g.nodes:
                if not isinstance(n, InOutNode):
                    continue
                if isinstance(n, ConvNode):
                    feati += n.in_ch * n.module.stride[0] * n.module.stride[1]
                    feato += score_dict[g]["out_ch"]
                    N += (
                        n.in_ch
                        * score_dict[g]["out_ch"]
                        * n.module.kernel_size[0]
                        * n.module.kernel_size[1]
                    )
                elif isinstance(n, LinearNode):
                    feati += n.in_ch
                    feato += score_dict[g]["out_ch"]
                    N += n.in_ch * score_dict[g]["out_ch"]
                elif isinstance(n, CustomNode):
                    feati += n.in_ch
                    feato += score_dict[g]["out_ch"]
                    N += n.prunable_weight_count() * score_dict[g]["out_ch"] / n.out_ch
                else:
                    continue
            N = int(N)
            score_dict[g]["N"] = N
            featio = feati / feato / N
            score_dict[g]["score"] = featio * torch.randn(N).abs()
        score_list = torch.cat(
            tuple([score_dict[key]["score"] for key in score_dict.keys()]), dim=0
        )
        kth = torch.kthvalue(score_list, num_toprune_each)[0]
        for key in score_dict.keys():
            score_dict[key]["pruned_count"] = torch.sum(score_dict[key]["score"] < kth)
            score_dict[key]["pruned_out_ch"] += int(
                score_dict[key]["out_ch"]
                * (score_dict[key]["pruned_count"] / score_dict[key]["N"])
            )
    for key in score_dict.keys():
        key.sparsity = score_dict[key]["pruned_out_ch"] / key.channel


def random(groups, sparsity, imk):
    """
    This aimed to test the different sparsity applied to different groups.
    """
    for g in groups:
        if g.haskey(imk):
            continue
        g.sparsity = sparsity * torch.rand(1)
