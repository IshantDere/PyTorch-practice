import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
    
G_XtoY = Generator()
G_YtoX = Generator()
D_X = Discriminator()
D_Y = Discriminator()
x = torch.randn(1, 3, 64, 64)
y = torch.randn(1, 3, 64, 64)
fake_y = G_XtoY(x)
fake_x = G_YtoX(y)
cycle_x = G_YtoX(fake_y)
cycle_y = G_XtoY(fake_x)

print(x.shape)
print(fake_y.shape)
print(cycle_x.shape)