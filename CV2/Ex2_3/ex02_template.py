import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.max_pool2d(F.relu(self.conv1(xb)), 2)
        xb = F.max_pool2d(F.relu(self.conv2(xb)), 2)
        xb = xb.view(-1, 32*7*7)
        xb = F.relu(self.fc1(xb))
        xb = self.fc2(xb)
        return xb
        
train_data = np.load("x_train.npy")
train_target = np.load("y_train.npy")

xb = torch.from_numpy(train_data)[:16]
yb = torch.from_numpy(train_target)[:16]

net = MNIST_CNN()

pred = net(xb)

loss_fcn = F.cross_entropy

loss = loss_fcn(pred, yb)
print(loss)

opt = torch.optim.SGD(net.parameters(), lr=0.1)

loss.backward()
opt.step()
opt.zero_grad()

new_pred = net(xb)
new_loss = loss_fcn(new_pred, yb)
print(new_loss)

