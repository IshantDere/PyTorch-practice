import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=1, feature_g=64):
        super().__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, feature_g*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g*8, feature_g*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g*4, feature_g*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g*2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self, img_channels=1, feature_d=64):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, feature_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d, feature_d*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d*2, 1, 4, 2, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)

z_dim = 100
gen = Generator(z_dim)
disc = Discriminator()

noise = torch.randn(1, z_dim, 1, 1)
fake_img = gen(noise)
decision = disc(fake_img)

print("Fake image shape:", fake_img.shape)
print("Disc output:", decision)
