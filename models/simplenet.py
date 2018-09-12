import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.l1 = nn.Linear(64 * 2 * 2, 32)
        self.l2 = nn.Linear(32, 10)

    def forward(self, x):
        out = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        out = F.max_pool2d(F.relu(self.bn2(self.conv2(out))), 2)
        out = F.max_pool2d(F.relu(self.bn3(self.conv3(out))), 2)
        out = F.max_pool2d(F.relu(self.bn4(self.conv4(out))), 2)
        result = out.view(-1, 64 * 2 * 2)
        result = self.l2(self.l1(result))
        return result, out
