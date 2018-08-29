import torch.nn as nn
import torch.nn.functional as F


img_shape = (3, 32, 32)


class TreeRootNet(nn.Module):
    def __init__(self):
        super(TreeRootNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.l1 = nn.Linear(16 * 8 * 8, 32)
        self.l2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), 2))
        out = F.relu(F.max_pool2d(self.bn2(self.conv2(out)), 2))
        result = out.view(-1, 16 * 8 * 8)
        result = self.l2(self.l1(result))
        return F.log_softmax(result, dim=1), out


class TreeBranchNet(nn.Module):
    def __init__(self, branch=5):
        super(TreeBranchNet, self).__init__()

        self.conv1 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.l1 = nn.Linear(64 * 2 * 2, 32)
        self.l2 = nn.Linear(32, branch)


    def forward(self, x):
        out = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), 2))
        out = F.relu(F.max_pool2d(self.bn2(self.conv2(out)), 2))
        result = out.view(-1, 64 * 2 * 2)
        result = self.l2(self.l1(result))
        return F.log_softmax(result, dim=1), out
