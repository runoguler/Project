import torch.nn as nn
import torch.nn.functional as F

img_shape = (3, 32, 32)

class Block(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class StaticTreeRootNet(nn.Module):
    cfg_full = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]
    cfg = [64, (128, 2), 128, (256, 2)]
    cfg_half = [64, (128, 2), 128]

    def __init__(self):
        super(StaticTreeRootNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg_half:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        return out


class StaticTreeBranchNet(nn.Module):
    cfg_full = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]
    cfg = [256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]
    cfg_half = [(128, 2), 128, (256, 2), 256, 256, 256, 256, 256, (512, 2), 512]

    def __init__(self, branch=6):
        super(StaticTreeBranchNet, self).__init__()
        self.layers = self._make_layers(in_planes=128)
        self.linear = nn.Linear(512, branch)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg_half:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, out):
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        result = out.view(out.size(0), -1)
        result = self.linear(result)
        return result, out
