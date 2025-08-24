import torch
from torch import nn

base_mode = [
    # expand_ratio, channels, repeats = no_of_layers, stride, kernal_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3 ],

]

phi_values = {
    # tuple of: (phi_value, resolution, drop_rate)
    "b0":(0, 224, 0.2),
    "b1":(0.5, 240, 0.2),
    "b2":(1, 260, 0.3),
    "b3":(2, 300, 0.3),
    "b4":(3, 380, 0.4),
    "b5":(4, 456, 0.4),
    "b6":(5, 528, 0.5),
    "b7":(6, 600, 0.5),
}
class CNN(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups = 1):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    
    class SqueezeExcitation(nn.Module):
    def __init__(self,
                 in_channels,
                 redused_dim):
        super().__init__()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # C x H x W --> C x 1 x 1
            nn.Conv2d(in_channels, redused_dim, 1),
            nn.SiLU(),
            nn.Conv2d(redused_dim, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)
    
    class InvertedResidualBlock(nn.Module):
        def __init__(self,
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    expand_ratio,
                    reduction = 4,
                    survival_prob = 0.8 # for stochastic depth
                    ):
            super().__init__()
            self.survival_prob = 0.8
            self.use_residual = in_channels == out_channels and stride == 1
            hidden_dim = in_channels * expand_ratio
            self.expand = in_channels != hidden_dim


