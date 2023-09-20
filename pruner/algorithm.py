import torch
import torch.nn as nn

import tqdm
from tqdm import trange
from thop import profile, clever_format
from torch import profiler

from .backward2node import (
    __backward2node,
    __get_groups,
    __skl2imk,
    __sum_output,
    __find_prev_keynode,
)
from .node import InOutNode, CustomNode, ConvNode, LinearNode, ConcatNode, AddNode


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
    bmt_dict={},
    imk=[],
    skl=[],
):
    device = torch.device(device)
    model.eval()
    print("=" * 20, "Original Model", "=" * 20)
    __get_flops(model, example_input, device)
    print("=" * 20, "Original Model", "=" * 20)

    node_dict, ignore_nodes = __backward2node(model, example_input, imt_dict, bmt_dict)

    imk.extend([n.name for n in ignore_nodes])
    imk.extend(__skl2imk(skl, node_dict))
    groups = __get_groups(node_dict)
    prune_fn = globals()[algorithm]
    prune_fn(groups, sparsity, imk, model, example_input, node_dict)

    groups_hascat = []
    groups_haspool = []
    for g in groups:
        if g.hascat():
            groups_hascat.append(g)
            continue
        if g.haspool():
            groups_haspool.append(g)
            continue
        g.prune(g.sparsity)
    for g in groups_haspool:
        g.prune(g.sparsity)
    for g in groups_hascat:
        g.prune(g.sparsity)

    for node in node_dict.values():
        node.execute()

    print("=" * 20, "Pruned Model", "=" * 20)
    __get_flops(model, example_input, device)
    print("=" * 20, "Pruned Model", "=" * 20)


def uniform(groups, sparsity, imk, model=None, example_input=None, node_dict=None):
    for g in groups:
        if g.haskey(imk):
            continue
        g.sparsity = sparsity


def __get_tag2nodes(node_dict):
    tag2nodes = {}
    tags_in = set()
    fusion_nodes = []
    for k, node in node_dict.items():
        if not isinstance(node, (ConcatNode, AddNode)):
            continue
        if len(node.tags) <= 1:
            continue
        tags_in.update(node.tags)
        if len(node.prev[0].tags) <= 1:
            fusion_nodes.append(node)
    for tag in tags_in:
        tag2nodes[tag] = set()
    for n in fusion_nodes:
        for prev in __find_prev_keynode(n.prev):
            if prev.group is None:
                continue
            for prev_g_node in prev.group.nodes:
                if not isinstance(prev_g_node, InOutNode) or isinstance(
                    prev_g_node, CustomNode
                ):
                    continue
                if prev_g_node.tags[0] in tags_in:
                    tag2nodes[prev_g_node.tags[0]].add(prev_g_node)
    return tag2nodes


def mmu(groups, sparsity, imk, model, example_input, node_dict):
    tag2nodes = __get_tag2nodes(node_dict)
    synflow_input = [
        torch.ones_like(example_input[0]),
        torch.ones_like(example_input[1]),
    ]
    out = model(*synflow_input)
    out, _ = __sum_output(out, 0)
    out.backward()

    tag2score = {}
    for tag, nodes in tag2nodes.items():
        tmp_score = 0.0
        count = 0
        for node in nodes:
            count += node.module.weight.numel()
            tmp_score = (
                tmp_score
                + (
                    node.module.weight.data.abs() * node.module.weight.grad.data.abs()
                ).sum()
            )
        tag2score[tag] = tmp_score / count
    min_value = min(tag2score.values())
    for k, v in tag2score.items():
        tag2score[k] = max(1 / (torch.log(v / min_value) + 1), 0.8)

    for g in groups:
        if g.haskey(imk):
            continue
        tags = set()
        for n in g.nodes:
            tags.update(n.tags)
        tags = list(tags)
        if len(tags) == 1:
            g.sparsity = sparsity * tag2score[tags[0]]
        elif len(tags) == 2:
            g.sparsity = sparsity * (tag2score[tags[0]] + tag2score[tags[1]]) / 2
        else:
            assert False, f"tags: {tags}"


def mmu2(groups, sparsity, imk, model, example_input, node_dict):
    tag2nodes = __get_tag2nodes(node_dict)
    synflow_input = [
        torch.ones_like(example_input[0]),
        torch.ones_like(example_input[1]),
    ]
    out = model(*synflow_input)
    out, _ = __sum_output(out, 0)
    out.backward()

    tag2score = {}
    for tag, nodes in tag2nodes.items():
        tmp_score = 0.0
        count = 0
        for node in nodes:
            count += node.module.weight.numel()
            tmp_score = (
                tmp_score
                + (
                    node.module.weight.data.abs() * node.module.weight.grad.data.abs()
                ).sum()
            )
        tag2score[tag] = tmp_score / count
    min_value = min(tag2score.values())
    for k, v in tag2score.items():
        tag2score[k] = max(1 / (torch.log(v / min_value) + 1), 0.8)

    for g in groups:
        if g.haskey(imk):
            continue
        tags = set()
        for n in g.nodes:
            tags.update(n.tags)
        tags = list(tags)
        if len(tags) == 1:
            g.sparsity = sparsity * tag2score[tags[0]]
        elif len(tags) == 2:
            g.sparsity = sparsity * (tag2score[tags[0]] + tag2score[tags[1]]) / 2
        else:
            assert False, f"tags: {tags}"


def erk(groups, sparsity, imk, model=None, example_input=None, node_dict=None):
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


def featio(groups, sparsity, imk, model=None, example_input=None, node_dict=None):
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


def featio50(groups, sparsity, imk, model=None, example_input=None, node_dict=None):
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


def random(groups, sparsity, imk, model=None, example_input=None, node_dict=None):
    """
    This aimed to test the different sparsity applied to different groups.
    """
    for g in groups:
        if g.haskey(imk):
            continue
        g.sparsity = sparsity * torch.rand(1)
