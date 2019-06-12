import torch.nn as nn
import math


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG16(nn.Module):

    cfg_full = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

    def __init__(self, cfg=cfg_full, num_classes=1000, fcl=512*7*7, hidden_layer=4096, init_weights=True):
        super(VGG16, self).__init__()
        self.features = make_layers(cfg, batch_norm=True)
        if fcl <= 1024:
            # hidden_layer = fcl
            self.classifier = nn.Linear(fcl, num_classes)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(fcl, hidden_layer),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(hidden_layer, hidden_layer),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(hidden_layer, num_classes),
            )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

