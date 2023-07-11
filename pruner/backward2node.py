import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from node import *
from group import *
import torchvision.models as models
import mvit

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
    "SigmoidBackward0",
    "TanhBackward0",
    "SoftmaxBackward0",
    "LogSoftmaxBackward0",
    # not activation, but plays the same role
    "CloneBackward0",
    "UnsafeViewBackward0",
    "ViewBackward0",
    "BmmBackward0",
]

IGNORE_BACKWARD_TYPE = (
    "AccumulateGrad",
    "TBackward0",
    "NoneType",
)
DEBUG = True

"""
START
https://github.com/pytorch/pytorch/issues/9922#issuecomment-686421899
trace module from grad_fn
"""


def hook(module, inputs, output):
    if torch.is_tensor(output):
        if "module" not in output.grad_fn.metadata:
            output.grad_fn.metadata["module"] = module
        if "output" not in output.grad_fn.metadata:
            output.grad_fn.metadata["output"] = output


def register_forward_hooks(model, hook):
    hooks = []
    for name, mod in model.named_modules():
        if not mod._modules:  # is a leaf module
            hooks.append(mod.register_forward_hook(hook))
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
    return keynode


def __find_prev_keynode(node_list):
    keynode = []
    for node in node_list:
        if node.node_type in ["in_out", "in_in", "remap", "reshape", "dummy"]:
            keynode.append(node)
        elif node.node_type in ["out_out", "activation", "pool"]:
            keynode.append(node)
            keynode.extend(__find_prev_keynode(node.prev))
    return keynode


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
                node_dict[f"output_{i}"] = DummyNode(f"output_{i}")
                grad_list.append(
                    [node_dict[f"output_{i}"], sub_g[0].next_functions[0][0], 0]
                )
                i += 1
            elif sub_g[0].__class__.__name__ == "AddBackward0":
                check_list.append(sub_g[0])
    for i in grad_list:
        print(i[0].name, i[1])
    return node_dict, grad_list, total_sum


def __backward2node(
    model,
    example_input,
):
    # get module2key
    module2key = {}
    for name, mod in model.named_modules():
        if not mod._modules:  # is a leaf module
            module2key[mod] = name

    # register forward hook
    hooks = register_forward_hooks(model, hook)

    output = model(example_input)

    # init node_dict
    # node_dict = {"output": DummyNode("output")}
    # grad_list = [[node_dict["output"], output.grad_fn, 0]]
    node_dict, grad_list, output_sum = __init_dict_and_list(output)
    output_sum.backward()

    # remove hook
    for h in hooks:
        h.remove()

    print("=" * 10, "Insert Relation", "=" * 10) if DEBUG else None
    checked_list = []
    backward2key_dict = {}
    while len(grad_list) > 0:
        [last, grad, level] = grad_list[0]
        grad_list.remove([last, grad, level])
        if [last, grad] in checked_list:
            continue
        else:
            checked_list.append([last, grad])
        g_name = grad.__class__.__name__
        if g_name in IGNORE_BACKWARD_TYPE:
            continue
        if grad in backward2key_dict.keys():
            g_key = backward2key_dict[grad]
        else:
            try:
                g_key = module2key[grad.metadata["module"]]
            except:
                g_key = last.name + "." + g_name[:4]
            if g_name == "ConvolutionBackward0":
                if grad.metadata["module"].groups == 1:
                    node_dict[g_key] = ConvNode(g_key, grad.metadata["module"])
                else:
                    node_dict[g_key] = GroupConvNode(g_key, grad.metadata["module"])
            elif g_name == "AddmmBackward0":
                node_dict[g_key] = LinearNode(g_key, grad.metadata["module"])
            elif g_name == "ReshapeAliasBackward0":
                node_dict[g_key] = FlattenNode(g_key)
            elif g_name == "PermuteBackward0":
                node_dict[g_key] = PermuteNode(g_key)
            elif g_name == "ExpandBackward0":
                node_dict[g_key] = ExpandNode(g_key)
            elif g_name == "TransposeBackward0":
                node_dict[g_key] = TransposeNode(g_key)
            elif g_name == "MeanBackward1":
                node_dict[g_key] = AdaptiveAvgPoolNode(g_key)
            elif g_name == "NativeBatchNormBackward0":
                node_dict[g_key] = NormNode(g_key, grad.metadata["module"])
            elif g_name == "NativeLayerNormBackward0":
                node_dict[g_key] = LayerNormNode(g_key, grad.metadata["module"])
            elif g_name == "CatBackward0":
                node_dict[g_key] = ConcatNode(g_key)
            elif g_name == "SplitBackward0":
                node_dict[g_key] = SplitNode(g_key)
                print(grad)
            elif g_name == "AddBackward0":
                node_dict[g_key] = AddNode(g_key)
            elif g_name == "SubBackward0":
                node_dict[g_key] = SubNode(g_key)
            elif g_name in ACTIVITION_BACKWARD_TYPE:
                node_dict[g_key] = ActiNode(g_key)
            elif g_name == "CppBackward0":
                pass
            elif g_name == "MulBackward0":
                node_dict[g_key] = MulNode(g_key)
            elif g_name == "MmBackward0":
                node_dict[g_key] = MMNode(g_key)
            elif g_name == "MaxPool2DWithIndicesBackward0":
                node_dict[g_key] = MaxPoolNode(g_key)
            elif g_name == "AvgPool2DBackward0":
                node_dict[g_key] = AvgPoolNode(g_key)
            else:
                print(f"Not supported {g_name}, please add patches") if DEBUG else None
                continue
            backward2key_dict[grad] = g_key
        # declare the relation
        node_dict[g_key].next.append(last)
        node_dict[g_key].add_level(level)
        last.prev.append(node_dict[g_key])

        # add next grad to the search list
        grad_next = grad.next_functions
        for sub_g in grad_next:
            grad_list.append([node_dict[g_key], sub_g[0], level + 1])
    for node in node_dict.values():
        if node.node_type in ["in_out", "remap", "reshape", "in_in"]:
            node.set_key()
    for node in node_dict.values():
        node.next = __find_next_keynode(node.next)
        node.prev = __find_prev_keynode(node.prev)
    print("=" * 10, "Insert Relation", "=" * 10) if DEBUG else None
    return node_dict, module2key


def __get_groups(node_dict):
    checked_list = list(node_dict.keys())
    groups = []
    while checked_list:
        print("=" * 10, "checked_list", checked_list)
        node_name = checked_list[0]
        node = node_dict[node_name]
        if not node.key:
            checked_list.remove(node_name)
            continue
        group = [node]
        if node.node_type in ["in_in"]:
            for kn in node.prev:
                if kn.key:
                    group.append(kn)
        group = CurrentGroup(list(set(group)))
        print("=" * 5, "group", [g.name for g in group.nodes])
        for n in group.nodes:
            print("remove", n.name) if DEBUG else None
            checked_list.remove(n.name)
        groups.append(group)
    return groups


def main():
    class ExampleModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, 1, 1)
            self.bn1 = nn.BatchNorm2d(8)
            self.conv2 = nn.Conv2d(4, 8, 3, 1, 1)
            self.conv3 = nn.Conv2d(4, 8, 3, 1, 1)
            self.bn23 = nn.BatchNorm2d(16)
            self.conv4 = nn.Conv2d(16, 16, 3, 1, 1, groups=16)
            self.bn4 = nn.BatchNorm2d(16)
            self.conv5_1 = nn.Conv2d(16, 16, 3, 1, 1)
            self.conv5_2 = nn.Conv2d(16, 16, 3, 1, 1)
            self.identity = nn.Identity()
            self.bn5i = nn.BatchNorm2d(16)
            self.bn45 = nn.BatchNorm2d(32)
            self.conv6 = nn.Conv2d(32, 1, 3, 1, 1)
            self.bn6 = nn.BatchNorm2d(1)
            # self.pool = nn.MaxPool2d(2, 2)
            self.pool = nn.AvgPool2d(2, 2)
            self.flat = nn.Flatten()
            self.fc = nn.Linear(4, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x1, x2 = torch.chunk(x, 2, dim=1)
            x1 = self.conv2(x1)
            x2 = self.conv3(x2)
            x = torch.cat([x1, x2], dim=1)
            x = self.bn23(x)
            identity = self.identity(x)
            x1 = self.conv4(x)
            x1 = self.bn4(x1)
            x2 = self.conv5_1(x)
            x2 = self.conv5_2(x2 + identity)
            x2 = self.bn5i(x2)
            x = torch.cat([x1, x2], dim=1)
            x = self.bn45(x)
            x = self.conv6(x)
            x = self.bn6(x)
            x = self.pool(x)
            x = self.flat(x)
            x = self.fc(x)
            return x

    # model = ExampleModel().eval()
    # model = models.resnet18().eval()
    model = mvit.mobilevit_xxs().eval()
    example_input = torch.randn(1, 3, 320, 320)

    node_dict, module2key = __backward2node(model, example_input)

    groups = __get_groups(node_dict)
    # print groups
    print("=" * 10, "Groups & Next", "=" * 10)
    for g in groups:
        g.print_info()
        g.print_next_info()
    print("=" * 10, "Groups & Next", "=" * 10)

    print("=" * 10, "Prune Groups", "=" * 10)
    groups.reverse()
    for g in groups:
        g.prune()
    print("=" * 10, "Prune Groups", "=" * 10)

    # print("=" * 10, "Prune Index", "=" * 10)
    # for node in node_dict.values():
    #     print(node.name, node.prune_idx)
    # print("=" * 10, "Prune Index", "=" * 10)

    print("=" * 10, "Pruning take effect", "=" * 10)
    for node in node_dict.values():
        node.execute()
    print("=" * 10, "Pruning take effect", "=" * 10)

    print("=" * 10, "Print Nodes", "=" * 10)
    for node in node_dict.values():
        if isinstance(node, (InOutNode, OutOutNode)):
            print(node.name, node.module)
    print("=" * 10, "Print Nodes", "=" * 10)

    print("=" * 10, "Print Model", "=" * 10)
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
            print(name, module)
    print("=" * 10, "Print Model", "=" * 10)

    example_input = torch.randn(1, 3, 4, 4)
    model(example_input)


if __name__ == "__main__":
    main()
