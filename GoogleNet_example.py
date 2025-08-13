import torch
import torch.nn as nn
import torch.nn.functional as F

class Inception(nn.Module):
    def __init__(self, in_channels, 
                 ch1, ch3reduce, 
                 ch3, ch5reduce, 
                 ch5, pool_proj):
        super(Inception, self).__init__()
    
        self.branch1 = nn.Conv2d(in_channels, ch1, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3reduce, kernel_size=1),
            nn.Conv2d(ch3reduce, ch3, kernel_size=3, padding=1)
        )

        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5reduce, kernel_size=1),
            nn.Conv2d(ch5reduce, ch5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        
        fout1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat([out1, out2, out3, out4], dim=1)


class SmallGoogLeNet(nn.Module):
    
    def __init__(self):
        super(SmallGoogLeNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.inception1 = Inception(64, 32, 48, 64, 8, 16, 16)
        self.fc = nn.Linear(32 + 64 + 16 + 16, 10) 

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = self.inception1(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

model = SmallGoogLeNet()
print(model)

x = torch.randn(1, 3, 224, 224)
output = model(x)
print(output.shape)