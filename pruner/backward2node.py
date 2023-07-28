import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .node import *
from .group import *
import torchvision.models as models
from . import mvit
from .test_model import *

from thop import profile, clever_format
from torch import profiler

# from nets.Achelous import Achelous3T
# import backbone.attention_modules.shuffle_attention as sa
# import backbone.radar.RadarEncoder as re
import torch_pruning as tp
import tqdm

CONV_TYPE = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
)
NORM_TYPE = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)
POOLING_BACKWARD_TYPE = [
    "MaxPool2DWithIndicesBackward0",
    "AvgPool2DBackward0",
]
ACTIVITION_BACKWARD_TYPE = [
    "ReluBackward0",
    "SiluBackward0",
    "GeluBackward0",
    "HardswishBackward0",
    "SigmoidBackward0",
    "TanhBackward0",
    "SoftmaxBackward0",
    "LogSoftmaxBackward0",
    # not activation, but plays the same role
    "CloneBackward0",
    # "UnsafeViewBackward0",
    # "ViewBackward0",
    # "BmmBackward0",
]

IGNORE_BACKWARD_TYPE = (
    # "AccumulateGrad",
    # "TBackward0",
    "NoneType",
)
DEBUG = True

"""
START
https://github.com/pytorch/pytorch/issues/9922#issuecomment-686421899
trace module from grad_fn
"""


def forward_hook(module, input, output):
    if torch.is_tensor(output):
        if not hasattr(output.grad_fn, "metadata"):
            print(module, input[0].shape, output.shape)
            return
        if "module" not in output.grad_fn.metadata:
            output.grad_fn.metadata["module"] = module
        if "output" not in output.grad_fn.metadata:
            output.grad_fn.metadata["output"] = output
        if "input" not in output.grad_fn.metadata:
            output.grad_fn.metadata["input"] = input[0]


def ig_forward_hook(module, input, output):
    if torch.is_tensor(output):
        if "module" not in output.grad_fn.metadata:
            output.grad_fn.metadata["ig_module"] = module
        if "output" not in output.grad_fn.metadata:
            output.grad_fn.metadata["output"] = output
        if "input" not in output.grad_fn.metadata:
            output.grad_fn.metadata["input"] = input[0]


def register_forward_hooks(model, imt):
    hooks = []
    imk = []
    for k, m in model.named_modules():
        if isinstance(m, tuple(imt)):
            for sub_k, _ in m.named_modules():
                imk.append(k + "." + sub_k)
    for name, mod in model.named_modules():
        if isinstance(mod, tuple(imt)):
            hooks.append(mod.register_forward_hook(ig_forward_hook))
        elif not mod._modules and name not in imk:  # is a leaf module
            hooks.append(mod.register_forward_hook(forward_hook))
    return hooks


# model = nn.Conv2d(10, 10, 3)
# register_forward_hooks(model, hook)
# inputs = (torch.randn(2,10,32,32),)
# output = model(*inputs)
# print(output.grad_fn.metadata['module'].kernel_size)
# print(output.grad_fn.metadata['output'].shape)
"""
END
https://github.com/pytorch/pytorch/issues/9922#issuecomment-686421899
"""


def __get_groupname(group):
    return [n.name + str(n.level) for n in group]


def __key2module(model, key):
    key_list = key.split(".")
    module = model
    for attr in key_list[:-1]:
        if attr.isdigit():
            module = module[int(attr)]
        else:
            module = getattr(module, attr)
    return module, ".".join(key_list[:-1])


def __find_next_keynode(node_list):
    keynode = []
    for node in node_list:
        if node.node_type in ["in_out", "in_in", "remap", "reshape", "dummy"]:
            keynode.append(node)
        elif node.node_type in ["out_out", "activation", "pool"]:
            keynode.append(node)
            keynode.extend(__find_next_keynode(node.next))
    return list(set(keynode))


def __find_prev_keynode(node_list):
    keynode = []
    for node in node_list:
        if node.node_type in ["in_out", "in_in", "remap", "reshape", "dummy"]:
            keynode.append(node)
        elif node.node_type in ["out_out", "activation", "pool"]:
            keynode.append(node)
            keynode.extend(__find_prev_keynode(node.prev))
    return list(set(keynode))


def __sum_output(output, count):
    total_sum = 0
    if isinstance(output, (list, tuple)):
        for o in output:
            o_total_sum, count = __sum_output(o, count)
            total_sum += o_total_sum
    elif isinstance(output, torch.Tensor):
        total_sum += output.sum()
        count += 1
    return total_sum, count


def __init_dict_and_list(output):
    total_sum, count = __sum_output(output, 0)
    node_dict = {}
    grad_list = []
    grad = total_sum.grad_fn
    check_list = [grad]
    i = 0
    while i < count:
        grad = check_list[0].next_functions
        check_list.remove(check_list[0])
        for sub_g in grad:
            if sub_g[0].__class__.__name__ == "SumBackward0":
                node_dict[f"output_{i}"] = OutputNode(f"output_{i}")
                grad_list.append(
                    [node_dict[f"output_{i}"], sub_g[0].next_functions[0][0]]
                )
                i += 1
            elif sub_g[0].__class__.__name__ == "AddBackward0":
                check_list.append(sub_g[0])
    for i in grad_list:
        print(i[0].name, i[1])
    return node_dict, grad_list, total_sum


def __backward2node(model, example_input, imt_dict):
    # get module2key
    module2key = {}
    for name, mod in model.named_modules():
        if isinstance(mod, tuple(imt_dict.keys())):
            module2key[mod] = name
        if not mod._modules:  # is a leaf module
            module2key[mod] = name

    # register forward hook
    hooks = register_forward_hooks(model, imt_dict.keys())

    if isinstance(example_input, torch.Tensor):
        output = model(example_input)
    elif isinstance(example_input, dict):
        output = model(**example_input)
    elif isinstance(example_input, (list, tuple)):
        output = model(*example_input)

    # init node_dict
    # node_dict = {"output": DummyNode("output")}
    # grad_list = [[node_dict["output"], output.grad_fn, 0]]
    node_dict, grad_list, output_sum = __init_dict_and_list(output)
    output_sum.backward()
    input_list = []
    if isinstance(example_input, torch.Tensor):
        node_dict["input"] = InputNode("input", example_input)
        input_list.append(node_dict["input"])
    elif isinstance(example_input, dict):
        for k in example_input.keys():
            node_dict["input_" + k] = InputNode("input_" + k, example_input[k])
            input_list.append(node_dict["input_" + k])
    elif isinstance(example_input, (list, tuple)):
        for i in range(len(example_input)):
            node_dict["input_" + str(i)] = InputNode(
                "input_" + str(i), example_input[i]
            )
            input_list.append(node_dict["input_" + str(i)])
    # remove hook
    for h in hooks:
        h.remove()

    print("=" * 10, "Insert Relation", "=" * 10) if DEBUG else None
    checked_list = []
    backward2key_dict = {}
    while len(grad_list) > 0:
        [last, grad] = grad_list[0]
        grad_list.remove([last, grad])
        if [last, grad] in checked_list:
            continue
        else:
            checked_list.append([last, grad])

        g_name = grad.__class__.__name__

        if g_name in IGNORE_BACKWARD_TYPE:
            continue

        if "ig_module" in grad.metadata.keys():
            # the ignored module
            if grad in backward2key_dict.keys():
                g_key = backward2key_dict[grad]
            else:
                g_key = module2key[grad.metadata["ig_module"]]
                node_dict[g_key] = imt_dict[type(grad.metadata["ig_module"])](
                    g_key, grad.metadata["ig_module"]
                )
                backward2key_dict[grad] = g_key
            node_dict[g_key].add_next(last)
            # node_dict[g_key].add_level(level)
            # if not hasattr(grad.metadata["input"].grad_fn, "next_functions"):
            #     continue
            grad_list.append(
                [node_dict[g_key], grad.metadata["input"].grad_fn]  # , level + 1]
            ) if hasattr(grad.metadata["input"], "grad_fn") else None
            # grad_next = grad.metadata["input"].grad_fn.next_functions
            # for sub_g in grad_next:
            #     grad_list.append([node_dict[g_key], sub_g[0], level + 1])
            continue

        if grad in backward2key_dict.keys():
            g_key = backward2key_dict[grad]
        else:
            try:
                g_key = module2key[grad.metadata["module"]]
            except:
                g_key = last.name + "." + g_name[:4]
                count = 0
                tmp_g_key = g_key + str(count)
                while tmp_g_key in node_dict.keys():
                    count += 1
                    tmp_g_key = g_key + str(count)
                g_key = tmp_g_key

            if g_name == "ConvolutionBackward0":
                if (
                    grad.metadata["module"].groups
                    == grad.metadata["module"].in_channels
                ) and (
                    grad.metadata["module"].groups
                    == grad.metadata["module"].out_channels
                ):
                    node_dict[g_key] = GroupConvNode(g_key, grad.metadata["module"])
                else:
                    node_dict[g_key] = ConvNode(g_key, grad.metadata["module"])
            elif g_name == "AddmmBackward0":
                node_dict[g_key] = LinearNode(g_key, grad.metadata["module"])
            # elif g_name == "ReshapeAliasBackward0":
            #     node_dict[g_key] = FlattenNode(g_key)
            # elif g_name == "PermuteBackward0":
            #     node_dict[g_key] = PermuteNode(g_key)
            # elif g_name == "ExpandBackward0":
            #     node_dict[g_key] = ExpandNode(g_key)
            # elif g_name == "TransposeBackward0":
            #     node_dict[g_key] = TransposeNode(g_key)
            elif g_name == "MeanBackward1":
                node_dict[g_key] = AdaptiveAvgPoolNode(g_key)
            elif g_name == "NativeBatchNormBackward0":
                node_dict[g_key] = NormNode(g_key, grad.metadata["module"])
            elif g_name == "NativeLayerNormBackward0":
                node_dict[g_key] = LayerNormNode(g_key, grad.metadata["module"])
            elif g_name == "NativeGroupNormBackward0":
                node_dict[g_key] = GroupNormNode(g_key, grad.metadata["module"])
            elif g_name == "CatBackward0":
                node_dict[g_key] = ConcatNode(g_key)
            elif g_name == "SplitBackward0":
                node_dict[g_key] = SplitNode(g_key)
                node_dict[g_key].update_info(grad)
            elif g_name == "AddBackward0":
                if "module" in grad.metadata.keys():
                    node_dict[g_key] = LinearNode(g_key, grad.metadata["module"])
                else:
                    node_dict[g_key] = AddNode(g_key)
            elif g_name == "SubBackward0":
                node_dict[g_key] = SubNode(g_key)
            elif g_name in ACTIVITION_BACKWARD_TYPE:
                node_dict[g_key] = ActiNode(g_key)
            elif g_name == "CppBackward0":
                pass
            # elif g_name == "MulBackward0":
            #     node_dict[g_key] = MulNode(g_key)
            elif g_name == "MmBackward0":
                node_dict[g_key] = MMNode(g_key)
            elif g_name == "MaxPool2DWithIndicesBackward0":
                node_dict[g_key] = MaxPoolNode(g_key)
            elif g_name == "AvgPool2DBackward0":
                node_dict[g_key] = AvgPoolNode(g_key)
            else:
                print(f"Not supported {g_name}, please add patches") if DEBUG else None
                grad_next = grad.next_functions
                for sub_g in grad_next:
                    grad_list.append([last, sub_g[0]])  # , level + 1])
                continue
            backward2key_dict[grad] = g_key
        # declare the relation
        node_dict[g_key].add_next(last)
        # node_dict[g_key].add_level(level)

        # add next grad to the search list
        grad_next = grad.next_functions
        for sub_g in grad_next:
            grad_list.append([node_dict[g_key], sub_g[0]])  # , level + 1])

        # find input
        if "input" in grad.metadata.keys():
            for i in input_list:
                if i.input is grad.metadata["input"]:
                    i.add_next(node_dict[g_key])
                    # i.add_level(level)
    ignore_nodes = []
    for node in node_dict.values():
        node.next = __find_next_keynode(node.next)
        node.prev = __find_prev_keynode(node.prev)
        if isinstance(node, OutputNode):
            for sub_n in node.prev:
                if sub_n.node_type == "in_out":
                    ignore_nodes.append(sub_n)
                else:
                    ignore_nodes.extend(sub_n.prev)

    print("=" * 10, "Insert Relation", "=" * 10) if DEBUG else None
    return node_dict, ignore_nodes


def __get_all_next(node, cl=[]):
    all_next = [node]
    cl.append(node)
    for n in node.next + node.prev:
        if not n.key:
            continue
        if n.node_type == "in_in" and n not in cl:
            tmp_all_next, tmp_cl = __get_all_next(n, cl)
            all_next.extend(tmp_all_next)
            cl.extend(tmp_cl)

    for prev_node in node.prev:
        if not prev_node.key:
            continue
        for prev_next_node in prev_node.next:
            if prev_next_node.node_type != "in_in" or prev_next_node in cl:
                continue
            tmp_all_next, tmp_cl = __get_all_next(prev_next_node, cl)
            all_next.extend(tmp_all_next)
            cl.extend(tmp_cl)
    all_next = list(set(all_next))
    cl = list(set(cl))
    return all_next, cl


def __get_group(node):
    group = [node]
    if node.node_type != "in_in":
        return group
    all_next, _ = __get_all_next(node, cl=[])
    all_next = list(set(all_next))
    group.extend(all_next)
    for next_node in all_next:
        if next_node.node_type != "in_in":
            continue
        group.extend([n for n in next_node.prev if n.key])
    return group


def __get_groups(node_dict):
    checked_list = list(node_dict.keys())
    AddB_list = []
    groups = []
    for name in node_dict.keys():
        if name[-5:] == "AddB0":
            checked_list.remove(name)
            AddB_list.append(name)
    checked_list = AddB_list + checked_list

    for node_name in AddB_list:
        if node_name not in checked_list:
            continue
        node = node_dict[node_name]
        group = __get_group(node)
        group = CurrentGroup(list(set(group)))
        print("=" * 5, "group", [g.name for g in group.nodes]) if DEBUG else None
        for n in group.nodes:
            print("remove", n.name) if DEBUG else None
            checked_list.remove(n.name)
        groups.append(group)

    while checked_list:
        node_name = checked_list[0]
        node = node_dict[node_name]
        if not node.key:
            checked_list.remove(node_name)
            continue
        group = __get_group(node)
        group = CurrentGroup(list(set(group)))
        print("=" * 5, "group", [g.name for g in group.nodes]) if DEBUG else None
        for n in group.nodes:
            print("remove", n.name) if DEBUG else None
            checked_list.remove(n.name)
            # try:
            #     checked_list.remove(n.name)
            # except:
            #     pass
        groups.append(group)
    return groups


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = test_model.ExampleModel().eval()
    # model = models.resnet18().eval()
    model = mvit.mobilevit_xxs().to(device).eval()
    example_input = torch.randn(1, 3, 320, 320).to(device)
    flops, params = profile(model, inputs=(example_input,))
    print(clever_format([flops, params], "%.3f"))
    # example_input = torch.randn(128, 3, 320, 320).to(device)
    # __test_speed(model, example_input)
    model.cpu()
    example_input = example_input.cpu()

    # ignored module type
    imt_dict = {mvit.Transformer: MVitNode}

    node_dict = __backward2node(model, example_input, imt_dict)

    groups = __get_groups(node_dict)
    # print groups
    if DEBUG:
        print("=" * 10, "Groups & Next", "=" * 10)
        for g in groups:
            g.print_info()
            g.print_next_info()
        print("=" * 10, "Groups & Next", "=" * 10)

    print("=" * 10, "Prune Groups", "=" * 10) if DEBUG else None
    groups.reverse()
    for g in groups:
        g.prune(0.7)
    print("=" * 10, "Prune Groups", "=" * 10) if DEBUG else None

    print("=" * 10, "Pruning take effect", "=" * 10) if DEBUG else None
    for node in node_dict.values():
        node.execute()
    print("=" * 10, "Pruning take effect", "=" * 10) if DEBUG else None

    model.to(device)
    example_input = torch.randn(1, 3, 320, 320).to(device)
    model(example_input)
    flops, params = profile(model, inputs=(example_input,))
    print(clever_format([flops, params], "%.3f"))
    # example_input = torch.randn(128, 3, 320, 320).to(device)
    # __test_speed(model, example_input)


if __name__ == "__main__":
    main()
