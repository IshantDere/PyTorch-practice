import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()

        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyRelu(0,1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)
    
class generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()

        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64
img_dim = 28*28*1
batch_size = 32
num_epochs = 50

disc = discriminator(img_dim).to(device)
gen = generator(z_dim, img_dim).to(device)

fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081))
])

dataset = datasets.MNIST(root="dataset/", transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
optim_disc = optim.Adam(disc.parameters(), lr = lr)
optim_gen = optim.Adam(gen.parameters(), lr = lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        # Train Discriminator 
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)

        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))

        disc_fake = disc(fake.detach()).view(-1)  
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        lossD = (lossD_real + lossD_fake) / 2

        optim_disc.zero_grad()
        lossD.backward()
        optim_disc.step()

        # Train Generator 
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))

        optim_gen.zero_grad()
        lossG.backward()
        optim_gen.step()

