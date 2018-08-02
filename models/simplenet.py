import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3)
        self.conv3 = nn.Conv2d(24, 48, kernel_size=3)
        self.fcl1 = nn.Linear(1728,600)
        self.fcl2 = nn.Linear(600,100)
        self.fcl3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 1728)
        x = self.fcl1(x)
        x = self.fcl2(x)
        x = self.fcl3(x)
        return F.log_softmax(x, dim=1)
