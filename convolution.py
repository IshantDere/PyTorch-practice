import torch
from torch import nn

class Conv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(5,5),
                 stride=1,
                 padding=0):
        super().__init__()

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()


    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x