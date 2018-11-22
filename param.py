from models.mobilenet import MobileNet
from models.mobile_tree_net import MobileTreeRootNet, MobileTreeLeafNet, MobileTreeBranchNet

import utils
import torch

from torchsummary import summary

def calculate_no_of_params(models, with_names=False):
    length = 0
    if isinstance(models, list):
        conv, bn, linear = 0, 0, 0
        for model in models:
            if not model is None:
                if with_names:
                    print(type(model))
                for name, p in model.named_parameters():
                    if "conv" in name:
                        conv += p.numel()
                    elif "bn" in name:
                        bn += p.numel()
                    else:
                        linear += p.numel()
                    if with_names:
                        print(name + ' - ' + str(p.numel()))
                length += sum(p.numel() for p in model.parameters())
    else:
        conv, bn, linear = 0, 0, 0
        for name, p in models.named_parameters():
            if "conv" in name:
                conv += p.numel()
            elif "bn" in name:
                bn += p.numel()
            else:
                linear += p.numel()
            if with_names:
                print(name + ' - ' + str(p.numel()))
        length = sum(p.numel() for p in models.parameters())
    return conv, bn, linear, length

def generate_model_list(root_node, level, device, fcl_factor, root_step=1, step=3, dividing_factor=2.0):
    leaf_node_labels = []
    cfg_full = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]
    models = [MobileTreeRootNet(cfg_full[:root_step]).to(device)]
    nodes = [(root_node, 0)]
    index = 0
    remaining = 1
    prev_lvl = 0
    while remaining > 0:
        if nodes[index] is None:
            models.append(None)
            models.append(None)
            nodes.append(None)
            nodes.append(None)
            index += 1
            continue

        lvl = nodes[index][1] + 1

        conv_step = (step * (lvl - 1)) + root_step
        in_planes = cfg_full[conv_step - 1] if isinstance(cfg_full[conv_step - 1], int) else cfg_full[conv_step - 1][0]

        if prev_lvl < lvl:
            prev_lvl = lvl
            for i in range(conv_step, len(cfg_full)-1, 2):
                if isinstance(cfg_full[i], int):
                    cfg_full[i] = int(cfg_full[i] // dividing_factor)
                else:
                    cfg_full[i] = (int(cfg_full[i][0] // dividing_factor), cfg_full[i][1])

        # LEFT BRANCH
        left = nodes[index][0].left
        if not isinstance(left, int):
            if left.count > 3 and lvl < level:
                models.append(MobileTreeBranchNet(input=cfg_full[conv_step:conv_step + 3], in_planes=in_planes).to(device))
                nodes.append((left, lvl))
                remaining += 1
            else:
                models.append(MobileTreeLeafNet(branch=(left.count + 1), input=cfg_full[conv_step:], in_planes=in_planes, fcl=cfg_full[-1]*fcl_factor*fcl_factor).to(device))
                nodes.append(None)
                leaf_node_labels.append(left.value)
        else:
            models.append(MobileTreeLeafNet(branch=2, input=cfg_full[conv_step:], in_planes=in_planes, fcl=cfg_full[-1]*fcl_factor*fcl_factor).to(device))
            nodes.append(None)
            leaf_node_labels.append((left,))

        # RIGHT BRANCH
        right = nodes[index][0].right
        if not isinstance(right, int):
            if right.count > 3 and lvl < level:
                models.append(MobileTreeBranchNet(input=cfg_full[conv_step:conv_step + 3], in_planes=in_planes).to(device))
                nodes.append((right, lvl))
                remaining += 1
            else:
                models.append(MobileTreeLeafNet(branch=(right.count + 1), input=cfg_full[conv_step:], in_planes=in_planes, fcl=cfg_full[-1]*fcl_factor*fcl_factor).to(device))
                nodes.append(None)
                leaf_node_labels.append(right.value)
        else:
            models.append(MobileTreeLeafNet(branch=2, input=cfg_full[conv_step:], in_planes=in_planes, fcl=cfg_full[-1]*fcl_factor*fcl_factor).to(device))
            nodes.append(None)
            leaf_node_labels.append((right,))

        index += 1
        remaining -= 1
    return models, leaf_node_labels

size = 64
no_classes = 365
depth = 2

fcl_factor = size // 32

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = MobileNet(num_classes=no_classes, fcl=(fcl_factor*fcl_factor*1024)).to(device)
#print('MobileNet: ' + str(calculate_no_of_params(model)))

root_node = utils.generate(no_classes, no_classes*5, False)
models, leaf_node_labels = generate_model_list(root_node, depth, device, fcl_factor,
                                               root_step=1, step=3, dividing_factor=2)

x = 0
if x == 0:
    conv, bn, linear, length = calculate_no_of_params(model)
    print('MobileNet:')
    print('Conv:\t' + str(conv))
    print('Bn: \t' + str(bn))
    print('Fcl:\t' + str(linear))
    print('Total:\t' + str(length))
    print()

    conv, bn, linear, length = calculate_no_of_params(models)
    print('TreeNet:')
    print('Conv:\t' + str(conv))
    print('Bn: \t' + str(bn))
    print('Fcl:\t' + str(linear))
    print('Total:\t' + str(length))
    print()

    # print(models)
    # summary(model, input_size=(3, size, size))

    for i, model in enumerate(models):
        if not model is None:
            if i == 0:
                summary(model, input_size=(3, size, size))
            if i == 1 or i == 2:
                summary(model, input_size=(64, size, size))
            if 2 < i < 7:
                summary(model, input_size=(128, size//4, size//4))
            if 6 < i < 15:
                summary(model, input_size=(256, size//8, size//8))



elif x == 1:
    conv, bn, linear, length = calculate_no_of_params(model)
    print('MobileNet:')
    print('Conv:\t' + str(conv))
    print('Bn: \t' + str(bn))
    print('Fcl:\t' + str(linear))
    print('Total:\t' + str(length))
else:
    conv, bn, linear, length = calculate_no_of_params(models)
    print('TreeNet:')
    print('Conv:\t' + str(conv))
    print('Bn: \t' + str(bn))
    print('Fcl:\t' + str(linear))
    print('Total:\t' + str(length))