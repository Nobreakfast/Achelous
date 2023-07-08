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


#########################
####### InOutNode #######
#########################
class InOutNode(BaseNode):
    def __init__(self, name: str, module) -> None:
        super().__init__(name, module, "in_out")
        self.in_ch, self.out_ch = self.get_channels()

    @abc.abstractmethod
    def get_channels(self):
        pass


####### ConvNode #######
class ConvNode(InOutNode):
    def __init__(self, name: str, module) -> None:
        super().__init__(name, module)

    def get_channels(self):
        return self.module.in_channels, self.module.out_channels


###### LienarNode ######
class LinearNode(InOutNode):
    def __init__(self, name: str, module) -> None:
        super().__init__(name, module)

    def get_channels(self):
        return self.module.in_features, self.module.out_features


#########################
####### InInNode ########
#########################
class InInNode(BaseNode):
    def __init__(self, name: str) -> None:
        super().__init__(name, None, "in_in")


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

    @abc.abstractmethod
    def get_channels(self):
        pass


####### NormNode ########
class NormNode(OutOutNode):
    def __init__(self, name: str, module) -> None:
        super().__init__(name, module)

    def get_channels(self):
        return self.module.num_features, self.module.num_features


####### GroupConvNode ########
class GroupConvNode(OutOutNode):
    def __init__(self, name: str, module) -> None:
        super().__init__(name, module)

    def get_channels(self):
        return self.module.groups, self.module.groups


#########################
###### DummyNode ########
#########################
class DummyNode(BaseNode):
    def __init__(self, name: str) -> None:
        super().__init__(name, None, "dummy")


#########################
###### RemapNode ########
#########################
class RemapNode(BaseNode):
    def __init__(self, name: str) -> None:
        super().__init__(name, None, "remap")

    def add_idx(self, index, dim):
        if self.in_ch == 0 and self.out_ch == 0:
            self.in_ch, self.out_ch = self.get_channels()
        return super().add_idx(index, dim)

    def get_channels(self):
        pass


###### ConcatNode ########
class ConcatNode(RemapNode):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def get_channels(self):
        for prev in self.prev:
            self.in_ch += prev.out_ch
        self.out_ch = prev.out_ch
        return self.in_ch, self.out_ch


###### SplitNode ########
class SplitNode(RemapNode):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def get_channels(self):
        for next in self.next:
            self.in_ch += next.in_ch
        self.out_ch = next.in_ch
        return self.in_ch, self.out_ch


###### AvgPoolNode ########
class AvgPoolNode(RemapNode):
    def __init__(self, name: str) -> None:
        super().__init__(name)

    def get_channels(self):
        self.in_ch = self.prev[0].out_ch
        return self.in_ch, self.in_ch


###### MaxPoolNode ########
class MaxPoolNode(RemapNode):
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


###### FlattenNode ######
class FlattenNode(ReshapeNode):
    def __init__(self, name: str) -> None:
        super().__init__(name)


########################
###### CustomNode ######
########################


###### DeformableNode ######
class DeformableNode(InOutNode):
    def __init__(self, name: str, module) -> None:
        super().__init__(name, module)

    def get_channels(self):
        return self.module.in_channels, self.module.out_channels
