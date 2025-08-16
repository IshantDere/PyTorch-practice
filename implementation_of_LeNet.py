import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernal_size = (5,5),
                 stride = 1,
                 padding = 0):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernal_size, stride, padding)

        self.norm = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU()

    def forward(self, x):

        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class LinearBloack(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class LeNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.ds = nn.MaxPool2d((2,2), 2)
        self.conv1 = CNN(in_channels, 6)
        self.conv2 = CNN(6, 16)
        self.conv3 = CNN(16, 120)

        self.l1 = LinearBloack(120, 84)
        self.l2 = LinearBloack(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ds(x)
        x = self.conv2(x)
        x = self.ds(x)
        x = self.conv3(x)

        x = x.squeeze(-1).squeeze(-1)
        x = self.l1(x)
        x = self.l2(x)
        return x

x = torch.randn(2, 1, 32, 32)
y = LeNet(1, 6)
z = y(x)
z.shape