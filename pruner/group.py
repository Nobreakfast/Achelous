import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import abc
from .node import *

"""
params:
    nodes: list, list of nodes
    type: str, current or next
"""


#########################
####### BaseGroup #######
#########################
class BaseGroup(abc.ABC):
    """
    BaseGroup
    """

    def __init__(self, nodes: list, type: str) -> None:
        self.nodes = nodes
        self.type = type
        self.next_group = None
        self.channel = self.get_info()

    def hascat(self):
        for n in self.nodes:
            if isinstance(n, ConcatNode):
                return True
        return False

    def haskey(self, key):
        for n in self.nodes:
            if n.name == key:
                return True
        return False

    @abc.abstractmethod
    def get_info(self):
        pass

    def print_info(self):
        print("=" * 20, self.type, "=" * 20)
        print(f"group nodes: {[node.name for node in self.nodes]}")
        print(f"group channels: {self.channel}")

    def prune(self, prune_idx, dim):
        for node in self.nodes:
            if isinstance(node, ConcatNode) and dim == 0:
                node.prune(prune_idx, dim, self.channel)
                continue
            node.prune(prune_idx, dim)


#########################
##### CurrentGroup ######
#########################
class CurrentGroup(BaseGroup):
    def __init__(self, nodes: list) -> None:
        super().__init__(nodes, "current")
        self.__get_next_group()
        self.round_to = self.next_group.get_round_to()

    def __get_next_group(self):
        next_group = []
        for node in self.nodes:
            next_group.extend(node.next)
        next_group = NextGroup(list(set(next_group)))
        self.next_group = next_group

    def get_info(self):
        out_ch = []
        for node in self.nodes:
            out_ch.append(node.get_channels()[1])
        # assert len(set(out_ch)) == 1, f"out_ch: {out_ch}"
        return max(out_ch)

    def print_next_info(self):
        self.next_group.print_info()

    def prune(self, prune_ratio):
        round_to = max(self.round_to)
        split = 1
        cat = 1
        for node in self.next_group.nodes:
            if node.name[:6] == "output":
                return
            if hasattr(node, "split"):
                round_to *= node.split
                split = node.split
                break
        for node in self.nodes:
            if hasattr(node, "cat_idx1"):
                cat = node.cat_idx1
                round_to *= cat
                break

        prune_num = (
            int(math.floor(self.channel * prune_ratio / round_to) * round_to)
            // split
            // cat
        )
        tmp_prune_idx = torch.cat(
            [
                torch.randperm(self.channel // split // cat)[:prune_num]
                + i * self.channel // split // cat
                for i in range(cat)
            ]
        )
        prune_idx = torch.cat(
            [tmp_prune_idx + i * self.channel / split for i in range(split)]
        )
        for node in self.nodes:
            if node.prune_idx[1] != []:
                # print("=" * 20, "node:", node.name)
                prune_idx = node.prune_idx[1]
                break
        super().prune(prune_idx, 1)
        self.next_group.prune(prune_idx)


#########################
####### NextGroup #######
#########################
class NextGroup(BaseGroup):
    def __init__(self, nodes: list) -> None:
        super().__init__(nodes, "next")

    def get_round_to(self):
        round_to = []
        for node in self.nodes:
            round_to.append(node.round_to)
        return round_to

    def get_info(self):
        in_ch = []
        for node in self.nodes:
            if isinstance(node, ConcatNode):
                in_ch.append(node.in_ch // node.ratio)
                continue
            in_ch.append(node.get_channels()[0])
        in_ch = list(set(in_ch))
        # assert len(in_ch) == 1, f"in_ch: {in_ch}"
        return max(in_ch)

    def prune(self, prune_idx):
        if isinstance(prune_idx, list):
            for node, count in zip(self.nodes, range(len(prune_idx))):
                node.prune(prune_idx[count], 0)
        else:
            super().prune(prune_idx, 0)
