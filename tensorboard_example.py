import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torchvision

writer = SummaryWriter('run/mnist_experiment')

for epoch in range(20):
  loss = 0.21 * epoch
  writer.add_scalar('Loss/train', loss, epoch)

model = nn.Linear(10, 1)
x = torch.randn(1, 10)
writer.add_graph(model, x)

writer.close()