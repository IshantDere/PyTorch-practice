import torch
from torch import nn
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torch.optim import Adam

class ConvDown(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = [2,2],
                 stride = [2,2],
                 padding = 0):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding)
        self.act = nn.ReLU()
        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.norm(x)
        return x

x = torch.randn(1,2, 128, 128)
model = ConvDown(2, 3)
y = model(x)
y.shape

class Convup(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size = [2,2],
                 stride = [2,2],
                 padding = 0):
        super().__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                              kernel_size, stride, padding)
        self.act = nn.ReLU()
        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.norm(x)
        return x
    
x = torch.randn(1,1, 128, 128)
model = Convup(1, 3)
y = model(x)
y.shape

class Encoder(nn.Module):
    def __init__(self,
                 in_channels=3,
                 hidden_dim=32,
                 n_layers=8):
        super().__init__()


        self.layers = []

        for _ in range(n_layers):
            self.layers.append(
                ConvDown(in_channels,
                         hidden_dim)
            )
            in_channels = hidden_dim
            hidden_dim = hidden_dim * 2

        self.layers = nn.Sequential(*self.layers)

    def forward(self,x):
        x = self.layers(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self,
                 in_channels=4096,
                 n_layers=8):
        super().__init__()

        self.layers = []

        for _ in range(n_layers):
            hidden_dim = in_channels // 2
            self.layers.append(
                ConvUp(in_channels,
                       hidden_dim
                       )
            )
            in_channels = hidden_dim
        self.layers = nn.Sequential(*self.layers)

        self.last = nn.Conv2d(in_channels,3,(3,3),(1,1),1)

    def forward(self,x):
        x = self.layers(x)
        x = self.last(x)
        return x
    
class Generator(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 n_layers=8):
        super().__init__()


        self.encoder = Encoder(in_channels,n_layers=n_layers)
        self.decoder = Decoder(4096,n_layers=n_layers)

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
class Discriminator(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=2,
                 hidden_dim=64,
                 n_layers=8):
        super().__init__()

        self.layers = []

        for _ in range(n_layers):
            self.layers.append(
                ConvDown(in_channels,hidden_dim)
            )
            in_channels = hidden_dim
            hidden_dim = hidden_dim * 2
        self.layers = nn.Sequential(*self.layers)
        self.linear1 = nn.Linear(8192,out_channels)

    def forward(self,x):
        x = self.layers(x)
        x = x[:,:,:1,:1]
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear1(x)
        return x
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # convert PIL → Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # normalize
])

class dataset(Dataset):
    def __init__(self,
                 root_dir="/content/root"):
        super().__init__()


        self.root_dir = root_dir
        self.files_A = os.listdir(os.path.join(
            self.root_dir, "trainA"
        ))
        self.files_B = os.listdir(os.path.join(
            self.root_dir, "trainB"
        ))

    def __len__(self):
        return len(self.files_B)

    def __getitem__(self,idx):
        img_a_selection = self.files_A[idx]

        img_a = Image.open(os.path.join(self.root_dir, 'trainA',img_a_selection)).convert('RGB')
        img_b = Image.open(os.path.join(self.root_dir, "trainB", img_a_selection)).convert('RGB')

        img_a_tensor = transform(img_a)
        img_b_tensor = transform(img_b)
        return img_a_tensor, img_b_tensor

dataset1 = dataset('/content/root')

for x, y in dataset1:
    print(x.shape, y.shape)

dataloader = DataLoader(dataset1,1,True)
generator = Generator()
discriminator = Discriminator()
optim_gen = Adam(generator.parameters())
optim_disc = Adam(discriminator.parameters())

epochs = 50
l1 = torch.nn.L1Loss()
disc_loss = nn.BCELoss()

for epoch in epochs:
    for x,y in dataloader:

        x_pred = generator(x)

        with torch.no_grad():
            x_disc_pred = discriminator(x_pred)

        optim_gen.zero_grad()
        loss_generator = l1(x_pred, y)
        loss_discriminator = disc_loss(x_disc_pred, 0)

        total_generator_loss = 2.0 * loss_generator + 0.5 * loss_discriminator
        total_generator_loss.backward()
        optim_gen.step()

        ######### Discriminator ############

        optim_disc.zero_grad()
        x_pred_disc = discriminator(x_pred)
        x_real_disc = discriminator(y)

        loss_fake = disc_loss(x_pred_disc,0)
        loss_real = disc_loss(x_real_disc,1)

        total_disc_loss = loss_fake + loss_real


        total_disc_loss.backward()
        optim_disc.step()