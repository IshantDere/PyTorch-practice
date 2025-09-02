import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, z_dim, num_classes, img_dim):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.gen = nn.Sequential(
            nn.Linear(z_dim + num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, img_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat([noise, self.label_emb(labels)], dim=1)
        return self.gen(x)
    
class Discriminator(nn.Module):
    def __init__(self, num_classes, img_dim):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.disc = nn.Sequential(
            nn.Linear(img_dim + num_classes, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        x = torch.cat([img, self.label_emb(labels)], dim=1)
        return self.disc(x)
    
z_dim = 64
img_dim = 28*28  
num_classes = 10
batch_size = 16

gen = Generator(z_dim, num_classes, img_dim)
disc = Discriminator(num_classes, img_dim)

noise = torch.randn(batch_size, z_dim)
labels = torch.randint(0, num_classes, (batch_size,))
fake_imgs = gen(noise, labels)

print(fake_imgs.shape)