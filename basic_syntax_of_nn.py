import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNet(nn.Module):
  def __init__(self):
    super(NeuralNet, self).__init__()

    self.layer_1 = nn.Linear(10, 20)
    self.layer_2 = nn.Linear(20, 5)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.layer_1(x))
    x = self.layer_2(x)
    return x
  
model = NeuralNet()
print(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

for epoch in range(100):
  inputs = torch.randn(32, 10)
  targets = torch.randn(32, 5)

  outputs = model(inputs)
  loss = criterion(outputs, targets)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  
  print("Epoch" + str(epoch + 1) + ",Loss :" + str(loss.item()))