import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        identity = x 
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity
        # OR #
        # out = identity + out

        out = F.relu(out)
        
        return out
    
x = torch.randn(1, 64, 32, 32)
block = ResidualBlock(64)
y = block(x)

print(y.shape)