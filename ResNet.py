# imports 
import torch
from torch import nn

# created convolution block
class Conv(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernal_size = (3,3),
                 stride = (1, 1),
                 padding = 1):
        super().__init__()

        self.conv = Conv(in_channels, 
                         out_channels,
                         kernal_size,
                         stride,
                         padding)
        self.norm = nn.BatchNorm2d(in_channels)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.act(x)
        return x
    
    # created a ResNet architecture
    
    class ResnetBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels):
        super().__init__()

        self.conv = Conv(in_channels,
                         out_channels)
        
    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = x + residual
        return x