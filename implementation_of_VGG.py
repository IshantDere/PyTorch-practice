import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self ,in_channels, out_channels, 
                 kernal_size = (3,3),
                 stride = 1,
                 padding = 1):
        super(CNN, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernal_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
    
class LinearBloack(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        return x
    
class VGG(nn.Module):
    def __init__(self, in_channels, out_channels ):
        super().__init__()

        self.ds = nn.MaxPool2d((2, 2), 2)

        self.conv1 = CNN(in_channels , 64)
        self.conv2 = CNN(64, 64)

        self.conv3 = CNN(64, 128)
        self.conv4 = CNN(128, 128)
        
        self.conv5 = CNN(128, 256)
        self.conv6 = CNN(256, 256)
        self.conv7 = CNN(256, 256)
    
        self.conv8 = CNN(256, 512)
        self.conv9 = CNN(512, 512)
        self.conv10 = CNN(512, 512)    

        self.conv11 = CNN(512, 512)
        self.conv12 = CNN(512, 512)
        self.conv13 = CNN(512, 512)
        
        self.l1 = LinearBloack(512, 4096)
        self.l2 = LinearBloack(4096, 4096)
        self.l3 = nn.Linear(4096, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.ds(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.ds(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.ds(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.ds(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ds(x)
        x = x[:,:,0:1,0:1]
        print(x.shape)
        x = x.squeeze(-1).squeeze(-1)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x
    
x = torch.randn(2, 3, 224, 224)
model = VGG(3, 1000)
y = model(x)
y.shape

# BY USING THE LOOP #

class VGG_Loop(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=1000,
                 hidden_dim=64,
                 layers=[2,2,3,3,3]):
        super().__init__()
        self.layers = []


        for i in range(len(layers)):
            for j in range(layers[i]):
                self.layers.append(
                    CNN(
                        in_channels,
                        hidden_dim
                    )
                )
                in_channels = hidden_dim
            
            hidden_dim = hidden_dim * 2
            self.layers.append(nn.MaxPool2d((2,2),2))

        self.l1 = LinearBloack(1024, 4096)
        self.l2 = LinearBloack(4096, 4096)
        self.l3 = nn.Linear(4096, 1000)

        self.seq = nn.Sequential(
            *self.layers
        )

    def forward(self,x):
        x = self.seq(x)
        x = x[:,:,-1,-1]
        
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x
    
x = torch.randn(2, 3, 224, 224)
model = VGG_Loop(3, 1000)
y = model(x)
y.shape