import torch
from torch import nn

class Inception_block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.Conv2d(16, 24, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=1),
            nn.Conv2d(4, 8, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 8, kernel_size=1)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)
    
x = torch.randn(1, 3, 32, 32)  
block = Inception_block(3)
y = block(x)
print(y.shape)