import torch
import torch.nn as nn
import torch.optim as optim

x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = torch.tensor([[10.0], [20.0], [30.0], [40.0], [50.0]])

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(1, 1) 
    def forward(self, x):
        return self.linear(x)

model = SimpleNN()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(500):
    outputs = model(x)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    if epoch % 50 == 0:
        print(f"Epoch [{epoch}/500], Loss: {loss.item():.4f}")

test_input = torch.tensor([[6.0]])
predicted = model(test_input).item()
print(f"\nPredicted marks for 6 hours of study: {predicted:.2f}")
