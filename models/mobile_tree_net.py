import torch.nn as nn
import torch.nn.functional as F

img_shape = (3, 32, 32)

cfg_full = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]


def make_layers(in_planes, input):
    layers = []
    for x in input:
        out_planes = x if isinstance(x, int) else x[0]
        stride = 1 if isinstance(x, int) else x[1]
        layers.append(Block(in_planes, out_planes, stride))
        in_planes = out_planes
    return nn.Sequential(*layers)


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


class MobileTreeRootNet(nn.Module):
    cfg = [64, (128, 2), 128]

    def __init__(self, input=cfg, in_planes=32):
        super(MobileTreeRootNet, self).__init__()
        self.conv1 = nn.Conv2d(img_shape[0], in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.layers = make_layers(in_planes=in_planes, input=input)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        return out


class MobileTreeBranchNet(nn.Module):
    cfg = [(256, 2), 256, (512, 2)]

    def __init__(self, input=cfg, in_planes=128):
        super(MobileTreeBranchNet, self).__init__()
        self.layers = make_layers(in_planes=in_planes, input=input)

    def forward(self, out):
        out = self.layers(out)
        return out


class MobileTreeLeafNet(nn.Module):
    cfg = [512, 512, 512, 512, 512, (1024, 2), 1024]

    def __init__(self, branch=6, input=cfg, in_planes=512, fcl=(1024*(img_shape[1]//32)*(img_shape[2]//32))):
        super(MobileTreeLeafNet, self).__init__()
        self.layers = make_layers(in_planes=in_planes, input=input)
        self.linear = nn.Linear(fcl, branch)

    def forward(self, out):
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        result = out.view(out.size(0), -1)
        result = self.linear(result)
        return result