import torch
import torch.nn as nn
import torch.nn.functional as F
import abc

class BaseNode(abc.ABC):
    def __init__(self, name: str, next: list, type: str) -> None:
        self.name = name
        self.next = next
        self.type = type
        self.in_ch, self.out_ch = self.get_channels()

    @abc.abstractmethod
    def get_channels(self):
        pass

    def prune_table(self):
        pass

class ConvNode(BaseNode):
    def __init__(self, name: str, next: list, type: str) -> None:
        super().__init__(name, next, type)
        pass

    def get_channels(self):
        print("conv node")
        return 3, 16


if __name__ == '__main__':
    print("="*10, "Test Node class", "="*10)
    conv = ConvNode('conv1', [], 'conv')
    print(conv.in_ch, conv.out_ch)