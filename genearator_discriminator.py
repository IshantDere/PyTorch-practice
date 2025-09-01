import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):  
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )
    
    def forward(self, x):
        return self.gen(x)
    
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.disc(x)
    
z_dim = 64       
img_dim = 28*28    

gen = Generator(z_dim, img_dim)
disc = Discriminator(img_dim)

noise = torch.randn(1, z_dim)

fake_image = gen(noise)

pred = disc(fake_image)

print("Fake image shape:", fake_image.shape)
print("Discriminator output:", pred)        