# imports

import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.utils.data import dataloader
import torchvision.datasets
import torchvision.transforms as transforms

# create fully connected network

class nural_network(nn.Module):
  def __init__(self, input_size, num_classes):
    super(nural_network, self).__init__()
    self.layer_1 = nn.Linear(input_size, 50)
    self.layer_2 = nn.Linear(50, num_classes)

  def forward(self, x):
    x = F.relu(self.layer_1(x))
    x = self.layer_2(x)
    return x
