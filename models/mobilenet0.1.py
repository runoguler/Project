'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_dw(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, in_planes, 3, stride, 1, groups=in_planes, bias=False),
        nn.BatchNorm2d(in_planes),
        nn.ReLU(inplace=True),

        nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


def make_layers(in_planes, channels):
    layers = []
    for x in channels:
        out_planes = x if isinstance(x, int) else x[0]
        stride = 1 if isinstance(x, int) else x[1]
        layers.append(conv_dw(in_planes, out_planes, stride))
        in_planes = out_planes
    return nn.Sequential(*layers)


# ImageNet to CIFAR-10 conversion: (3x224x224 size image --> 3x32x32)
# First stride is 2 --> 1
# AvgPool2d is 7 --> 2
class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10, channels=cfg, fcl=1024):
        super(MobileNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ),
            make_layers(in_planes=32, channels=channels),
            nn.AvgPool2d(2)
        )
        self.fc = nn.Linear(fcl, num_classes)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def test():
    net = MobileNet()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

test()
