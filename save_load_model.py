import torch
from torch import nn
from torch import optim

class NN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.layer_1(x)
        return x

learning_rate = 0.001

model = NN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = learning_rate)

x = torch.randn(100, 10)
y = torch.randn(100, 1)

num_epochs = 5
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    print("Epoch", epoch + 1, "/", num_epochs, "Loss:", round(loss.item(), 4))

    if epoch == 2:
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": loss.item(),
        }, "checkpoint.pth")
        print("Checkpoint saved at epoch {}".format(epoch+1))
        break
checkpoint = torch.load("checkpoint.pth")

model = NN()
optimizer = optim.SGD(model.parameters(), lr=0.01)

model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optimizer_state"])
start_epoch = checkpoint["epoch"] + 1
loss = checkpoint["loss"]

print("Loaded checkpoint from epoch {} with loss {:.4f}".format(start_epoch, loss))

for epoch in range(start_epoch, num_epochs):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, loss.item()))