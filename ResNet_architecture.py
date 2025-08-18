import torch
from torch import nn

class Conv(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernal_size = (3,3),
                 stride = (1, 1),
                 padding = 1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels,
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
    
    class Encoder(nn.Module):
    def __init__(self, in_channels,
                 hidden_dim = 32,
                 n_layers = 4):
        super().__init__()

        self.layer = []
        old_hidden_dim = in_channels

        for i in range(n_layers):
            self.layer.append(
                Conv(
                    old_hidden_dim,
                    hidden_dim,
                    (2,2),
                    (2,2),
                    0
                )
            )

            old_hidden_dim = hidden_dim
            hidden_dim = hidden_dim * 2
            self.seq = nn.Sequential(*self.layer)
    def forward(self, x):
        x = self.seq(x)
        return x
    
x = torch.randn(2, 3, 512, 512)
model = Encoder(3, n_layers=4)
model(x).shape

class Res(nn.Module):
    def __init__(self,
                 in_channels,
                 n_layers = 5):
        super().__init__()

        self.layers = []

        for i in range(n_layers):
            self.layers.append(
                ResnetBlock(in_channels,
                            in_channels)
            )
    def forward(self, x):
        residual = x
        external_residual_connection = x

        for j in self.layers:
            x = j(x)
            x = x + residual
            return x + external_residual_connection
        
x = torch.randn(2, 256, 32, 32)
model = Res(256)
y = model(x)
y.shape
class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_dim=256,
                 n_layers = 5):
        super().__init__()

        self.decoder_layers = []

        old_hidden_dim = in_channels

        for i in range(n_layers):
            self.decoder_layers.append(
                Conv(
                    old_hidden_dim,
                    hidden_dim,
                    (2,2),
                    (2,2),
                    0
                )
            )

            old_hidden_dim = hidden_dim
            hidden_dim = hidden_dim

        self.dec_seq = nn.Sequential(*self.decoder_layers)

        self.linear = nn.Linear(256,out_channels)


    def forward(self,x):
        x = self.dec_seq(x)
        x = x.squeeze(-1).squeeze(-1)

        x = self.linear(x)
        return x
    
x = torch.randn(2,256,32,32)
model = Decoder(256,1000)
y = model(x)
y.shape
class Resnet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()

        self.encoder = Encoder(in_channels)
        self.residual_block = Res(256)
        self.decoder = Decoder(256,1000)

    def forward(self,x):
        x = self.encoder(x)
        x = self.residual_block(x)
        x = self.decoder(x)
        return x

x = torch.randn(2,3,512,512)
model = Resnet(3,1000)
y = model(x)
y.shape