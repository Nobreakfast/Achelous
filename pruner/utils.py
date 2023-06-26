import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence
from pruner.structure import *
import backbone.attention_modules.shuffle_attention as sa
import math


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
            featio = (
                out_ch
                / strides[0]
                / strides[1]
                / in_ch
                / (torch.sum(m.weight.data).item() + 1e-8)
            )
        elif isinstance(m, nn.Linear):
            # print("Linear layer", k)
            in_ch = m.in_features
            out_ch = m.out_features
            featio = out_ch / in_ch / (torch.sum(m.weight.data).item() + 1e-8)
        else:
            continue
        sparsity_dict[k] = 1 / featio
        sum += sparsity_dict[k]
    for k in sparsity_dict.keys():
        sparsity_dict[k] = amount * (1 - sparsity_dict[k] / sum)
    return sparsity_dict


# def prune_model(model, sparsity_dict):
#    """
#    Prune the model according to the sparsity dict.
#    """
#    pass


def __list_to_dict(list: Sequence[str]) -> dict:
    """
    Convert a list to a dict.
    """
    dict = {}
    for k1 in list:
        dict[k1] = {}
        for k2 in list:
            dict[k1][k2] = 0
    return dict


def get_dep_dcit(
    model,
    example_input=torch.randn(2, 3, 224, 224),
    type_list=[nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm],
    debug=False,
):
    """
    Get the dependency dict for a model.
    """
    # get the white list and module dict
    if debug:
        print("Get white list and module dict...")
    white_list = ["output"]
    module_list = {}
    for k, m in model.named_modules():
        if not isinstance(m, tuple(type_list)):
            continue
        white_list.append(k)
        module_list[k] = m

    # get the dependency dict
    if debug:
        print("Get dependency dict...")
    dep_dict = __list_to_dict(white_list)

    # get the name dict for id address to key
    if debug:
        print("Get name dict...")
    name_dict = {}
    for k, m in model.named_modules():
        name_dict[id(m)] = k

    # get module forward and backward dict
    forward_dict, backward_dict = {}, {}

    # ==================== Upsample layer image_radar_encoder.fpn.upsample_5_to_4.upsample.1
    # ==================== Upsample layer image_radar_encoder.fpn.upsample_4_to_3.upsample.1
    # ==================== Upsample layer image_radar_encoder.fpn.lane_seg_3_to_2.upsample.1
    # ==================== Upsample layer image_radar_encoder.fpn.lane_seg_2_to_1.upsample.1
    # ==================== Upsample layer image_radar_encoder.fpn.lane_seg_1_to_0.upsample.1
    # ==================== Upsample layer image_radar_encoder.fpn.se_seg_3_to_2.upsample.1
    # ==================== Upsample layer image_radar_encoder.fpn.se_seg_2_to_1.upsample.1
    # ==================== Upsample layer image_radar_encoder.fpn.se_seg_1_to_0.upsample.1
    def __forward_hook_callback(module, input, output):
        forward_dict[name_dict[id(module)]] = {}
        forward_dict[name_dict[id(module)]]["output"] = output
        forward_dict[name_dict[id(module)]]["input"] = input[0]

    def __backward_hook_callback(module, grad_input, grad_output):
        backward_dict[name_dict[id(module)]] = {}
        backward_dict[name_dict[id(module)]]["output"] = grad_output[0]
        backward_dict[name_dict[id(module)]]["input"] = grad_input

    hooks = []
    if debug:
        print("Get forward and backward dict...")
    for k, m in module_list.items():
        hooks.append(m.register_forward_hook(__forward_hook_callback))
        hooks.append(m.register_backward_hook(__backward_hook_callback))
    try:
        out = model(example_input)
    except:
        out = model(*example_input)
    try:
        out.sum().backward()
    except:
        outs = out[-1].sum()
        # outs = 0
        # for o in out:
        #     outs += o.sum()
        outs.backward()

    # fill the dependency dict
    def __check_id(g):
        for k, m in forward_dict.items():
            if m["output"].grad_fn == g:
                return k
        return "none"

    checked_list = []
    cat_list = []

    def __fill_dep(parent, g, level=0):  # TODO level: useless params
        if (parent, g) in checked_list:
            return
        if g == None:
            return
        if g.__class__.__name__ == "CatBackward0":
            cat_list.append(parent)
        elif g.__class__.__name__ == "SplitBackward0":
            try:
                cat_list.remove(parent)
            except:
                pass
        id = __check_id(g)
        if id != "none":
            dep_dict[parent][id] = 1
            dep_dict[id][parent] = -1
        else:
            # print('Warning: {} not in white list'.format(g.__class__.__name__))
            id = parent
        checked_list.append((parent, g))
        for subg in g.next_functions:
            if parent == "image_radar_encoder.fpn.spp.cv2.conv":
                print("=" * 20, "image_radar_encoder.fpn.spp.cv2.conv", "=" * 20)
                print(g)
            __fill_dep(id, subg[0], level + 1)

    if debug:
        print("Fill dependency dict...")
    try:
        __fill_dep("output", out.grad_fn, 0)
    except:
        __fill_dep("output", outs.grad_fn, 0)
    for hook in hooks:
        hook.remove()
    if debug:
        print("get dependency dict finished.")
    return dep_dict, module_list, cat_list


def get_group_list(dep_dict, module_list, debug=False):
    related_dict = {}
    if debug:
        print("Get related dict...")

    for k1, v1 in dep_dict.items():
        related_dict[k1] = {"in": [], "out": []}
        for k2, v2 in v1.items():
            if v2 == 1:
                related_dict[k1]["in"].append(k2)
            elif v2 == -1:
                related_dict[k1]["out"].append(k2)
            else:
                continue
    for k, v in related_dict.items():
        if k[-10:] == "upsample.1":
            print("*" * 20, k)
            print("in", related_dict[k]["in"])
            print("out", related_dict[k]["out"])
    # get the group dict
    group_list = []
    layers = []
    # layers_next = []

    def __add_layer(m_list, layer_type):
        tmp_list = []
        for i in m_list:
            if i == "output":
                continue
            if isinstance(
                module_list[i], (nn.BatchNorm2d, nn.LayerNorm, sa.ShuffleAttention)
            ):
                tmp_list += __add_layer(related_dict[i][layer_type], layer_type)
            else:
                if i[-10:] == "upsample.1":
                    print("*" * 20, i)
                tmp_list.append(i)
        return tmp_list

    if debug:
        print("add all input layer to list...")
    for k, v in related_dict.items():
        layer = __add_layer(v["in"], "in")
        layers.append(layer)
        # layer_next = __add_layer(v['out'], 'out')
        # layers_next.append(layer_next)

    def __get_merged_data(layers):
        merged_data = []
        # iterate over each sublist in the layers
        for sublist in layers:
            sublist_set = set(sublist)
            merged = False

            temp = []
            for merged_sublist in merged_data:
                if sublist_set.intersection(set(merged_sublist)):
                    merged_sublist.extend(temp)
                    merged_sublist.extend(sublist)
                    merged_sublist[:] = list(set(merged_sublist))
                    merged = True
                    if temp is not []:
                        try:
                            merged_data.remove(temp)
                        except:
                            pass
                    temp = merged_sublist

            if not merged:
                merged_data.append(sublist)
        return list(filter(None, merged_data))

    if debug:
        print("get merged data...")
    layers = __get_merged_data(layers)
    # layers_next = __get_merged_data(layers_next)
    for i in group_list:
        for j in i["modules"]:
            if j[-10:] == "upsample.1":
                print("*" * 20, k)
                print("module", j["modules"])
                print("next", j["out"])
    if debug:
        print("get group list...")

    def __get_next_layer(layer):
        next_layer = []
        if layer == "output":
            return next_layer
        elif isinstance(
            module_list[layer], (nn.BatchNorm2d, nn.LayerNorm, sa.ShuffleAttention)
        ):
            next_layer.extend(related_dict[layer]["out"])
            for k in related_dict[layer]["out"]:
                next_layer.extend(__get_next_layer(k))
        return next_layer

    for i in layers:
        group = {"modules": [], "next": []}
        for j in i:
            group["modules"].append(j)
            group["next"].extend(related_dict[j]["out"])
            for k in related_dict[j]["out"]:
                group["next"].extend(__get_next_layer(k))
        group["next"] = list(set(group["next"]))
        group_list.append(group)

    if debug:
        print("get group list finished.")
    return group_list, related_dict


def prune_model(
    model,
    ratio=0.5,
    prune_type="random",
    example_input=torch.randn(2, 3, 32, 32),
    type_list=[nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm],
    ignore_list=[[], []],
    debug=False,
):
    if debug:
        print("start getting dep dict")
    dep_dict, module_list, cat_list = get_dep_dcit(
        model, example_input, type_list, debug=debug
    )
    if debug:
        print("cat_list:", cat_list)
        print("start getting group list")
    group_list, related_dict = get_group_list(dep_dict, module_list, debug=debug)
    # for group in group_list:
    #     print("modules:", group["modules"])
    #     print("next_modules:", group["next"])

    def __set_module(name, new_module):
        name_list = name.split(".")
        module = model
        for i in name_list[:-1]:
            if i.isdigit():
                module = module[int(i)]
            else:
                module = getattr(module, i)
        setattr(module, name_list[-1], new_module)

    if debug:
        print("start pruning")

    for i in group_list:
        # if i['next'] == ['output']: # TODO: delete this and add in the get_group_list
        #    continue
        if isinstance(module_list[i["modules"][0]], nn.Conv2d):
            out_ch = module_list[i["modules"][0]].out_channels
        elif isinstance(module_list[i["modules"][0]], nn.Linear):
            out_ch = module_list[i["modules"][0]].out_features
        elif isinstance(module_list[i["modules"][0]], (nn.Upsample, nn.BatchNorm2d)):
            continue
        else:
            out_ch = module_list[i["modules"][0]].channel

        # calculate round_to
        round_to = [1]
        for j in i["next"] + i["modules"]:
            if j == "output":
                continue
            # if isinstance(module_list[j], nn.Conv2d):
            try:
                round_to.append(module_list[j].groups)
            except:
                continue
        round_to = max(round_to)
        if prune_type == "random":
            prune_num = int(math.floor(out_ch * ratio / round_to) * round_to)
            prune_idx = torch.randperm(out_ch)[:prune_num]
        if debug:
            print(
                "round_to:",
                round_to,
                "prune_num:",
                prune_num,
            )
        for j in i["modules"]:
            if debug:
                print("pruning modules:", j, module_list[j])
            if j in ignore_list[0]:  # or module_list[j] in ignore_list[1]:
                if debug:
                    print("ignore:", j, "in modules")
                continue
            if isinstance(module_list[j], nn.Conv2d):
                pruned_module = structure_conv(module_list[j], prune_idx, 0)
            elif isinstance(module_list[j], nn.Linear):
                pruned_module = structure_fc(module_list[j], prune_idx, 0)
            elif isinstance(module_list[j], sa.ShuffleAttention):
                pruned_module = structure_shuffleAttn(module_list[j], prune_idx)
            # elif isinstance(module_list[j], nn.Upsample):
            #     module_list[related_dict[j]["in"][0]].groups = 2
            module_list[j] = pruned_module
            __set_module(j, pruned_module)

        intersection = list(set(i["next"]).intersection(cat_list))
        intersection = [k for k in cat_list if k in intersection]
        if "image_radar_encoder.fpn.spp.cv2.conv" in intersection:
            intersection.append("image_radar_encoder.fpn.spp.cv2.conv")
            intersection.append("image_radar_encoder.fpn.spp.cv2.conv")
        for j in i["next"] + intersection:
            if j == "output":
                continue
            if debug:
                print("pruning next:", j, module_list[j])
            if j in ignore_list[1]:
                if debug:
                    print("ignore:", j, "in next")
                continue
            if isinstance(module_list[j], nn.Conv2d):
                pruned_module = structure_conv(module_list[j], prune_idx, 1)
            elif isinstance(module_list[j], nn.Linear):
                pruned_module = structure_fc(module_list[j], prune_idx, 1)
            elif isinstance(module_list[j], nn.BatchNorm2d):
                pruned_module = structure_bn(module_list[j], prune_idx)
            elif isinstance(module_list[j], nn.LayerNorm):
                pruned_module = structure_ln(module_list[j], prune_idx)
            elif isinstance(module_list[j], nn.GroupNorm):
                pruned_module = structure_gn(module_list[j], prune_idx)
            elif isinstance(module_list[j], sa.ShuffleAttention):
                pruned_module = structure_shuffleAttn(module_list[j], prune_idx)

            module_list[j] = pruned_module
            __set_module(
                j, pruned_module
            )  # structure_conv(module_list[j], prune_idx, 1))

    # for k, m in model.named_modules():
    #     if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
    #         print(k, m)
    if debug:
        print("finish pruning")
    return model


# def __test_get_dep_dict():
#     from torchvision.models import resnet18
#     from thop import profile
#     from mvit import mobilevit_s, Attention, FeedForward

#     # model = resnet18()
#     model = mobilevit_s(320)
#     example_input = torch.randn(2, 3, 320, 320)
#     macs, params = profile(model, inputs=(example_input,))
#     print("macs:", macs, "params:", params)

#     il = [[], []]  # ignore_list
#     for k, m in model.named_modules():
#         if isinstance(m, Attention):
#             il[0].append(k + ".to_qkv")
#             if m.to_out[0]:
#                 il[1].append(k + ".to_out.0")
#     dep_dict, module_list, name_dict = get_dep_dcit(model, example_input, debug=False)
#     # convert dict to pd
#     import pandas as pd

#     dep_df = pd.DataFrame(dep_dict)
#     dep_df.to_csv("/Users/allen/Downloads/dep.csv")


# def __test_get_group_list():
#     from torchvision.models import resnet18

#     # model = conv5()
#     model = resnet18()
#     dep_dict, module_list, name_dict = get_dep_dcit(model)
#     group_list = get_group_list(dep_dict, module_list, name_dict)
#     for group in group_list:
#         print("modules:", group["modules"])
#         print("next_modules:", group["next"])


def __test_pruning_model():
    from torchvision.models import resnet18
    from thop import profile
    from mvit import mobilevit_s, Attention, FeedForward
    from shuffle_attention import ShuffleAttention

    class masked_shuffleAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 512, 3, 1, 1)
            self.module = ShuffleAttention(512, 16, 8)

        def forward(self, x):
            return self.module(self.conv(x))

    # model = resnet18()
    # model = mobilevit_s(320)
    model = masked_shuffleAttn()
    example_input = torch.randn(2, 3, 320, 320)
    # example_input = torch.randn(2, 3, 32, 32)
    macs, params = profile(model, inputs=(example_input,))
    print(
        "macs:",
        macs,
        "params:",
        params,
    )

    ignore_Block = [ShuffleAttention]
    tl = [nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.LayerNorm, ShuffleAttention]

    il = [[], []]  # ignore_list
    for k, m in model.named_modules():
        if isinstance(m, Attention):
            il[0].append(k + ".to_qkv")
            if m.to_out[0]:
                il[1].append(k + ".to_out.0")
        if isinstance(m, tuple(ignore_Block)):
            il[0].append(k + ".gn")
            il[1].append(k + ".gn")

    model_new = prune_model(
        model,
        ratio=0.7,
        example_input=example_input,
        type_list=tl,
        ignore_list=il,
        debug=False,
    )
    # for k, m in model_new.named_modules():
    #     if isinstance(m, tl):
    #         print(k, m)

    macs, params = profile(model_new, inputs=(example_input,))
    print(
        "macs:",
        macs,
        "params:",
        params,
    )
    output = model_new(example_input)
    try:
        print(output.shape)
    except:
        for i in output:
            print(i.shape)
    # print(model_new)


if __name__ == "__main__":
    # __test_get_dep_dict()
    # __test_get_group_list()
    __test_pruning_model()
