import torch
import torch.nn as nn
import torch.nn.functional as F
import abc

"""
params:
    name: str
    module: nn.Module, Customize Module
    node_type: str
        "in_out": has in_channels and out_channels such as conv, fc
        "in_in": both side are same layer, such as Add, Sub
        "out_out": attach to a layer, such as norm, groups=channel, relu
        "remap": remap the feature map, such as concat, split, pooling, add (special case), mul, mm
        "reshape": reshape a layer, such as flatten
        "dummy": dummy node, such as output1, output2
        "activation": activation function, such as relu, sigmoid, tanh
        "pool": pooling function, such as maxpool, avgpool
"""


#########################
####### BaseNode ########
#########################
class BaseNode(abc.ABC):
    """
    BaseNode
    """

    def __init__(self, name: str, module, node_type: str) -> None:
        self.name = name
        self.module = module
        self.module_type = type(module)
        self.node_type = node_type
        self.prev = []
        self.next = []
        self.prune_idx = [[], []]
        self.in_count, self.out_count = 0, 0
        self.in_ch, self.out_ch = 0, 0
        self.key = False
        self.level = 0

    @abc.abstractmethod
    def execute(self):
        pass

    def add_prev(self, prev):
        self.prev.append(prev)

    def add_next(self, next):
        self.next.append(next)

    def add_idx(self, index, dim):
        if dim == 0:
            self.prune_idx[dim] = index
            self.in_count += 1
        else:
            if self.out_count == 0:
                self.prune_idx[dim] = index
            elif self.out_count > 0:
                self.prune_idx[dim] = torch.cat(
                    [
                        self.prune_idx[dim],
                        index + torch.tensor(self.out_ch * self.out_count),
                    ]
                )
                print("=" * 6, self.prune_idx[dim])
                print("=" * 6, index)
            self.out_count += 1

    def add_level(self, level):
        self.level = max(level, self.level)

    def get_channels(self):
        self.in_ch = self.prev[0].out_ch
        self.out_ch = self.next[0].in_ch
        return self.in_ch, self.out_ch

    def prune(self, prune_idx, dim):
        self.prune_idx[dim] = prune_idx

    def _get_saved_idx(self, param, prune_idx, dim):
        len_param = param.shape[dim]
        saved_idx = torch.LongTensor(
            [i for i in range(len_param) if i not in prune_idx]
        )
        return saved_idx

    def _prune_param(self, param, saved_idx, dim):
        param.data = torch.index_select(param.data, dim, saved_idx)
        if param.grad is not None:
            param.grad.data = torch.index_select(param.grad.data, dim, saved_idx)


#########################
####### InOutNode #######
#########################
class InOutNode(BaseNode):
    def __init__(self, name: str, module) -> None:
        super().__init__(name, module, "in_out")
        self.in_ch, self.out_ch = self.get_channels()
        self.saved_idx = [[], []]
        self.key = True

    @abc.abstractmethod
    def get_channels(self):
        pass

    @abc.abstractmethod
    def execute(self):
        pass

    def _prune_weight(self):
        self.saved_idx[1] = self._get_saved_idx(
            self.module.weight, self.prune_idx[1], 0
        )
        self.saved_idx[0] = self._get_saved_idx(
            self.module.weight, self.prune_idx[0], 1
        )
        self._prune_param(self.module.weight, self.saved_idx[1], 0)
        self._prune_param(self.module.weight, self.saved_idx[0], 1)
        if self.module.bias is not None:
            self._prune_param(self.module.bias, self.prune_idx[1], 0)


####### ConvNode #######
class ConvNode(InOutNode):
    def __init__(self, name: str, module) -> None:
        super().__init__(name, module)

    def get_channels(self):
        return self.module.in_channels, self.module.out_channels

    def execute(self):
        self.module.in_channels = self.in_ch - len(self.prune_idx[0])
        self.module.out_channels = self.out_ch - len(self.prune_idx[1])
        self._prune_weight()


###### LienarNode ######
class LinearNode(InOutNode):
    def __init__(self, name: str, module) -> None:
        super().__init__(name, module)

    def get_channels(self):
        return self.module.in_features, self.module.out_features

    def execute(self):
        self.module.in_features = self.in_ch - len(self.prune_idx[0])
        self.module.out_features = self.out_ch - len(self.prune_idx[1])
        self._prune_weight()


#########################
####### InInNode ########
#########################
class InInNode(BaseNode):
    def __init__(self, name: str) -> None:
        super().__init__(name, None, "in_in")
        self.key = True

    def execute(self):
        pass


###### AddNode ########
class AddNode(InInNode):
    def __init__(self, name: str) -> None:
        super().__init__(name)


###### SubNode ########
class SubNode(InInNode):
    def __init__(self, name: str) -> None:
        super().__init__(name)


###### MulNode ########
class MulNode(InInNode):
    def __init__(self, name: str) -> None:
        super().__init__(name)


###### MMNode ########
class MMNode(InInNode):
    def __init__(self, name: str) -> None:
        super().__init__(name)


#########################
###### OutOutNode #######
#########################
class OutOutNode(BaseNode):
    def __init__(self, name: str, module) -> None:
        super().__init__(name, module, "out_out")
        self.in_ch, self.out_ch = self.get_channels()
        self.saved_idx = []

    @abc.abstractmethod
    def execute(self):
        pass

    def _prune_weight(self):
        self.saved_idx = self._get_saved_idx(self.module.weight, self.prune_idx[0], 0)
        self._prune_param(self.module.weight, self.saved_idx, 0)
        if self.module.bias is not None:
            self._prune_param(self.module.bias, self.saved_idx, 0)


####### NormNode ########
class NormNode(OutOutNode):
    def __init__(self, name: str, module) -> None:
        super().__init__(name, module)

    def get_channels(self):
        return self.module.num_features, self.module.num_features

    def execute(self):
        self._prune_weight()
        self.module.num_features = len(self.saved_idx)
        self.module.running_mean = self.module.running_mean.data[self.saved_idx]
        self.module.running_var = self.module.running_var.data[self.saved_idx]


####### LayerNormNode ########
class LayerNormNode(OutOutNode):
    def __init__(self, name: str, module) -> None:
        super().__init__(name, module)

    def get_channels(self):
        return self.module.normalized_shape[0], self.module.normalized_shape[0]

    def execute(self):
        self._prune_weight()


####### GroupConvNode ########
class GroupConvNode(OutOutNode):
    def __init__(self, name: str, module) -> None:
        super().__init__(name, module)

    def get_channels(self):
        return self.module.groups, self.module.groups

    def execute(self):
        self.module.groups = self.module.groups - len(self.prune_idx[0])
        self.module.in_ch = self.module.groups
        self.module.out_ch = self.module.groups
        self._prune_weight()


#########################
###### DummyNode ########
#########################
class DummyNode(BaseNode):
    def __init__(self, name: str) -> None:
        super().__init__(name, None, "dummy")

    def get_channels(self):
        self.in_ch = self.prev[0].out_ch
        self.out_ch = -1
        return self.in_ch, self.out_ch

    def execute(self):
        pass


#########################
####### ActiNode ########
#########################
class ActiNode(BaseNode):
    def __init__(self, name: str) -> None:
        super().__init__(name, None, "activation")

    def execute(self):
        pass


#########################
###### RemapNode ########
#########################
class RemapNode(BaseNode):
    def __init__(self, name: str) -> None:
        super().__init__(name, None, "remap")
        self.ratio = 1  # record the concat and split ratio
        self.key = True

    def add_idx(self, index, dim):
        if self.in_ch == 0 and self.out_ch == 0:
            self.in_ch, self.out_ch = self.get_channels()
        return super().add_idx(index, dim)

    def execute(self):
        pass


###### ConcatNode ########
class ConcatNode(RemapNode):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def get_channels(self):
        if self.in_ch != 0 and self.out_ch != 0:
            return self.in_ch, self.out_ch
        for prev in self.prev:
            if prev.out_ch == 0:
                prev.get_channels()
            if prev.key:
                self.in_ch += prev.out_ch
        self.ratio = self.in_ch // prev.out_ch
        self.out_ch = self.in_ch
        assert self.in_ch == prev.out_ch * self.ratio
        assert (
            self.out_ch == self.next[0].in_ch
        ), f"self.next[0].in_ch: {self.next[0].in_ch}, self.out_ch: {self.out_ch}"
        return self.in_ch, self.out_ch

    def prune(self, prune_idx, dim):
        if dim == 0:
            self.prune_idx[0] = torch.cat(
                [
                    torch.Tensor(self.prune_idx[0]),
                    prune_idx + self.in_count * self.in_ch / self.ratio,
                ],
                dim=0,
            )
            self.prune_idx[1] = self.prune_idx[0]
            self.in_count += 1


###### SplitNode ########
class SplitNode(RemapNode):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def get_channels(self):
        if self.in_ch != 0 and self.out_ch != 0:
            return self.in_ch, self.out_ch
        for next in self.next:
            if next.in_ch == 0:
                next.get_channels()
            if next.key:
                self.in_ch += next.in_ch
        self.ratio = self.in_ch // next.in_ch
        self.out_ch = next.in_ch

        assert self.in_ch == next.in_ch * self.ratio
        assert (
            self.out_ch == self.next[0].in_ch
        ), f"self.next[0].in_ch: {self.next[0].in_ch}, self.out_ch: {self.out_ch}"
        return self.in_ch, self.out_ch

    def prune(self, prune_idx, dim):
        if dim == 0:
            self.prune_idx[0] = prune_idx
            len_idx_unit = len(prune_idx) // self.ratio
            for i in range(0, self.ratio):
                start = i * len_idx_unit
                end = (i + 1) * len_idx_unit
                self.prune_idx[1].append(prune_idx[start:end] - i * self.out_ch)


#########################
######## PoolNode########
#########################
class PoolNode(BaseNode):
    def __init__(self, name: str) -> None:
        super().__init__(name, None, "pool")

    def execute(self):
        pass


###### AvgPoolNode ########
class AvgPoolNode(PoolNode):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def get_channels(self):
        self.in_ch = self.prev[0].out_ch
        return self.in_ch, self.in_ch


###### AdaptiveAvgPoolNode ########
class AdaptiveAvgPoolNode(PoolNode):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def get_channels(self):
        self.in_ch = self.prev[0].out_ch
        return self.in_ch, self.in_ch


###### MaxPoolNode ########
class MaxPoolNode(PoolNode):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def get_channels(self):
        self.in_ch = self.prev[0].out_ch
        return self.in_ch, self.in_ch


########################
###### ReshapeNode #####
########################
class ReshapeNode(BaseNode):
    def __init__(self, name: str) -> None:
        super().__init__(name, None, "reshape")
        self.key = True

    def execute(self):
        pass


###### FlattenNode ######
class FlattenNode(ReshapeNode):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def prune(self, prune_idx, dim):
        if dim == 0:
            self.prune_idx[0] = prune_idx
            self.prune_idx[1] = prune_idx


###### PermuteNode ######
class PermuteNode(ReshapeNode):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def prune(self, prune_idx, dim):
        if dim == 0:
            self.prune_idx[0] = prune_idx
            self.prune_idx[1] = prune_idx


###### ExpandNode #######
class ExpandNode(ReshapeNode):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def prune(self, prune_idx, dim):
        if dim == 0:
            self.prune_idx[0] = prune_idx
            self.prune_idx[1] = prune_idx


###### TransposeNode #######
class TransposeNode(ReshapeNode):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def prune(self, prune_idx, dim):
        if dim == 0:
            self.prune_idx[0] = prune_idx
            self.prune_idx[1] = prune_idx


########################
###### CustomNode ######
########################


###### DeformableNode ######
class DeformableNode(InOutNode):
    def __init__(self, name: str, module) -> None:
        super().__init__(name, module)

    def get_channels(self):
        return self.module.in_channels, self.module.out_channels

    def execute(self):
        pass


######## MVitNode ########
class MVitNode(OutOutNode):
    def __init__(self, name: str, module) -> None:
        super().__init__(name, module)

    def get_channels(self):
        return (
            self.module.layers[0][0].norm.normalized_shape[0],
            self.module.layers[0][1].fn.net[3].out_features,
        )

    def execute(self):
        self.saved_idx = self._get_saved_idx(
            self.module.layers[0][0].fn.to_qkv.weight, self.prune_idx[0], 1
        )
        for attn, ff in self.module.layers:
            # prune norm and to_qkv input
            self._prune_param(attn.norm.weight, self.saved_idx, 0)
            self._prune_param(attn.norm.bias, self.saved_idx, 0)
            attn.norm.normalized_shape = (len(self.saved_idx),)

            self._prune_param(attn.fn.to_qkv.weight, self.saved_idx, 1)
            attn.fn.to_qkv.in_features = len(self.saved_idx)
            # prune_to_out output
            if isinstance(attn.fn.to_out, nn.Identity):
                pass
            else:
                self._prune_param(attn.fn.to_out[0].weight, self.saved_idx, 0)
                if attn.fn.to_out[0].bias is not None:
                    self._prune_param(attn.fn.to_out[0].bias, self.saved_idx, 0)
                attn.fn.to_out[0].out_features = len(self.saved_idx)
            # prune norm ff input
            self._prune_param(ff.norm.weight, self.saved_idx, 0)
            self._prune_param(ff.norm.bias, self.saved_idx, 0)
            ff.norm.normalized_shape = (len(self.saved_idx),)

            self._prune_param(ff.fn.net[0].weight, self.saved_idx, 1)
            ff.fn.net[0].in_features = len(self.saved_idx)
            # prune ff output
            self._prune_param(ff.fn.net[3].weight, self.saved_idx, 0)
            if ff.fn.net[3].bias is not None:
                self._prune_param(ff.fn.net[3].bias, self.saved_idx, 0)
            ff.fn.net[3].out_features = len(self.saved_idx)
