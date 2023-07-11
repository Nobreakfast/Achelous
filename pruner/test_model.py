import torch
import torch.nn as nn
import torch.nn.functional as F


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
