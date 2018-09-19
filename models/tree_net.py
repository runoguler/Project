import torch.nn as nn
import torch.nn.functional as F

img_shape = (3, 32, 32)


class TreeRootNet(nn.Module):
    def __init__(self):
        super(TreeRootNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, x):
        out = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        out = F.max_pool2d(F.relu(self.bn2(self.conv2(out))), 2)
        return out


class TreeBranchNet(nn.Module):
    def __init__(self, branch=6):
        super(TreeBranchNet, self).__init__()

        self.conv1 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.l1 = nn.Linear(64 * 2 * 2, 32)
        self.l2 = nn.Linear(32, branch)

    def forward(self, x):
        out = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        out = F.max_pool2d(F.relu(self.bn2(self.conv2(out))), 2)
        result = out.view(-1, 64 * 2 * 2)
        result = self.l2(self.l1(result))
        return result, out
