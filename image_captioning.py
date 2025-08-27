import torch
import torchvision.models as models
from torch import nn

cnn = models.resnet18(pretrained=True)
cnn = nn.Sequential(*list(cnn.children())[:-1])

class CaptionRNN(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256):
        super().__init__()
        self.rnn = nn.LSTM(512, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, img_features):
        h, _ = self.rnn(img_features)
        return self.fc(h)
    
image = torch.randn(1, 3, 224, 224)
features = cnn(image).view(1, 1, 512)
model = CaptionRNN(vocab_size=1000)
output = model(features)