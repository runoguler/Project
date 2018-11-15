import argparse
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

import logging
import time

from models.tree_net import TreeRootNet, TreeBranchNet
from models.mobilenet import MobileNet
from models.mobile_static_tree_net import StaticTreeRootNet, StaticTreeBranchNet
from models.mobile_tree_net import MobileTreeRootNet, MobileTreeLeafNet, MobileTreeBranchNet

import utils
import os


def train_tree(models, train_loader, device, epoch, args):
    models[0].train()
    models[1].train()
    models[2].train()

    lossfn = torch.nn.CrossEntropyLoss()
    lossfn.to(device)

    optim = torch.optim.Adam(list(models[0].parameters()) + list(models[1].parameters()) + list(models[2].parameters()), lr=args.lr,
                                betas=(0.5, 0.999))

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        optim.zero_grad()

        layer = models[0](data)
        out_b1, _ = models[1](layer)
        out_b2, _ = models[2](layer)

        b1_labels = labels.clone()
        b2_labels = labels.clone() - 5

        b1_labels[b1_labels > 4] = 5
        b2_labels[b2_labels < 0] = 5

        loss1 = lossfn(out_b1, b1_labels)
        loss2 = lossfn(out_b2, b2_labels)

        loss = loss1 + loss2
        loss.backward()
        optim.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tB1 Loss: {:.6f}\tB2 Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                       100. * batch_idx / len(train_loader), loss1.item(), loss2.item()))


def test_tree(models, test_loader, device):
    models[0].eval()
    models[1].eval()
    models[2].eval()

    corrects, no_class, both_class, false_in_class, correct_from_both, max_correct_from_both = 0, 0, 0, 0, 0, 0

    for data, label in test_loader:
        data, labels = data.to(device), label.to(device)

        layer = models[0](data)
        out_b1, _ = models[1](layer)
        out_b2, _ = models[2](layer)

        pred_b1 = out_b1.max(1, keepdim=True)[1]
        pred_b2 = out_b2.max(1, keepdim=True)[1]

        for i in range(labels.size(0)):
            if pred_b1[i] == 5:
                if pred_b2[i] == 5:
                    no_class += 1
                else:
                    if labels[i].item() == (pred_b2[i].item() + 5):
                        corrects += 1
                    else:
                        false_in_class += 1
            else:
                if pred_b2[i] == 5:
                    if labels[i].item() == pred_b1[i].item():
                        corrects += 1
                    else:
                        false_in_class += 1
                else:
                    both_class += 1
                    if (labels[i].item() == (pred_b2[i].item() + 5)) or (labels[i].item() == pred_b1[i].item()):
                        correct_from_both += 1
                        if (out_b1[i][pred_b1[i].item()].item() > out_b2[i][pred_b2[i].item()].item()) and (
                                labels[i].item() == pred_b1[i].item()):
                            max_correct_from_both += 1
                        elif (out_b2[i][pred_b2[i].item()].item() > out_b1[i][pred_b1[i].item()].item()) and (
                                labels[i].item() == (pred_b2[i].item() + 5)):
                            max_correct_from_both += 1

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n\t  F_in: {} None: {} Both: ({}/{}/{})\n'.format(
        corrects, len(test_loader.sampler),
        100. * corrects / len(test_loader.sampler),
        false_in_class, no_class, both_class, correct_from_both, max_correct_from_both
    ))


def train_dynamic_tree(models, leaf_node_labels, train_loader, device, epoch, args):
    leaf_node_index = []
    leaf_node_paths = []  # NOT INCLUDING models[0]

    for i in range(len(models)):
        if not models[i] is None:
            models[i].train()
            if isinstance(models[i], MobileTreeLeafNet):
                leaf_node_index.append(i)

    losses = []
    optims = []
    for i in leaf_node_index:
        path = []
        while i > 0:
            path = [i] + path
            i = (i + 1) // 2 - 1
        model_path = list(models[0].parameters())
        for i in path:
            model_path += list(models[i].parameters())

        leaf_node_paths.append(path)
        losses.append(torch.nn.CrossEntropyLoss().to(device))
        optims.append(torch.optim.Adam(model_path, lr=args.lr, betas=(0.5, 0.999)))

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        for i in range(len(optims)):
            optims[i].zero_grad()

        losses_to_print = []
        for i in range(len(leaf_node_paths)):  # for every branch(path) going to a leaf node
            layer = models[0](data)
            for j in range(len(leaf_node_paths[i])):
                k = leaf_node_paths[i][j]
                if j + 1 == len(leaf_node_paths[i]):
                    result, _ = models[k](layer)

                    lbls = labels.clone()
                    for l in range(len(lbls)):
                        if lbls[l].item() in leaf_node_labels[i]:
                            lbls[l] = leaf_node_labels[i].index(lbls[l])
                        else:
                            lbls[l] = len(leaf_node_labels[i])

                    l = losses[i](result, lbls)
                    l.backward(retain_graph=True)
                    optims[i].step()
                    losses_to_print.append(l)
                else:
                    layer = models[k](layer)

        if batch_idx % args.log_interval == 0:
            p_str = 'Train Epoch: {} [{}/{} ({:.0f}%)]'
            for loss in losses_to_print:
                p_str += '\tLoss: {:.6f}'.format(loss.item())

            print(p_str.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader)))


def train_dynamic_tree_old(models, leaf_node_labels, train_loader, device, epoch, args, use_cuda):
    leaf_node_index = []
    leaf_node_paths = []    # NOT INCLUDING models[0]

    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    for i in range(len(models)):
        if not models[i] is None:
            models[i].train()
            if isinstance(models[i], MobileTreeLeafNet):
                leaf_node_index.append(i)

    losses = []
    optims = []
    for j, i in enumerate(leaf_node_index):
        path = []
        while i > 0:
            path = [i] + path
            i = (i+1)//2 - 1
        model_path = list(models[0].parameters())
        for i in path:
            model_path += list(models[i].parameters())
        leaf_node_paths.append(path)

        if args.no_weights:
            losses.append(torch.nn.CrossEntropyLoss().to(device))
        else:
            weights = [1.0] * (len(leaf_node_labels[j]) + 1)
            weights[-1] = args.weight_mult / (args.num_classes - len(leaf_node_labels))
            losses.append(torch.nn.CrossEntropyLoss(weight=FloatTensor(weights)).to(device))

        optims.append(torch.optim.Adam(model_path, lr=args.lr))

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        for i in range(len(optims)):
            optims[i].zero_grad()

        losses_to_print = []
        for i in range(len(leaf_node_paths)):   # for every branch(path) going to a leaf node
            layer = models[0](data)
            for j in range(len(leaf_node_paths[i])):
                k = leaf_node_paths[i][j]
                if j+1 == len(leaf_node_paths[i]):
                    result, _= models[k](layer)

                    lbls = labels.clone()
                    for l in range(len(lbls)):
                        if isinstance(leaf_node_labels[i], int):
                            if lbls[l].item() == leaf_node_labels[i]:
                                lbls[l] = 0
                            else:
                                lbls[l] = 1
                        else:
                            if lbls[l].item() in leaf_node_labels[i]:
                                lbls[l] = leaf_node_labels[i].index(lbls[l])
                            else:
                                lbls[l] = len(leaf_node_labels[i])

                    l = losses[i](result, lbls)
                    l.backward(retain_graph=True)
                    optims[i].step()
                    losses_to_print.append(l)
                else:
                    layer = models[k](layer)

        if batch_idx % args.log_interval == 0:
            p_str = 'Train Epoch: {} [{}/{} ({:.0f}%)]'
            for loss in losses_to_print:
                p_str += '\tLoss: {:.6f}'.format(loss.item())

            print(p_str.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                       100. * batch_idx / len(train_loader)))


def train_hierarchical(models, leaf_node_labels, train_loader, device, epoch, args, use_cuda, min_depth, max_depth):
    leaf_node_index = []
    leaf_node_paths = []

    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    for i in range(len(models)):
        if not models[i] is None:
            if (2 ** (max_depth + 1)) > i + 1 >= (2 ** min_depth):    # Models in the specific depth level
                models[i].train()
                for param in models[i].parameters():
                    param.requires_grad = True
            else:
                models[i].train()
                for param in models[i].parameters():
                    param.requires_grad = False
            if isinstance(models[i], MobileTreeLeafNet):
                leaf_node_index.append(i)

    losses = []
    optims = []
    for j, i in enumerate(leaf_node_index):
        path = []
        while i >= 0:
            path = [i] + path
            i = (i+1)//2 - 1
        model_path = list()
        for i in path:
            if (2 ** (max_depth + 1)) > i + 1 >= (2 ** min_depth):
                model_path += list(models[i].parameters())
        leaf_node_paths.append(path)

        if args.no_weights:
            losses.append(torch.nn.CrossEntropyLoss().to(device))
        else:
            weights = [1.0] * (len(leaf_node_labels[j]) + 1)
            weights[-1] = args.weight_mult / (args.num_classes - len(leaf_node_labels))
            losses.append(torch.nn.CrossEntropyLoss(weight=FloatTensor(weights)).to(device))

        optims.append(torch.optim.Adam(model_path, lr=args.lr))

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        for i in range(len(optims)):
            optims[i].zero_grad()

        losses_to_print = []
        for i in range(len(leaf_node_paths)):   # for every branch(path) going to a leaf node
            layer = models[0](data)
            for j in range(len(leaf_node_paths[i])):
                k = leaf_node_paths[i][j]
                if j+1 == len(leaf_node_paths[i]):
                    result, _= models[k](layer)

                    lbls = labels.clone()
                    for l in range(len(lbls)):
                        if isinstance(leaf_node_labels[i], int):
                            if lbls[l].item() == leaf_node_labels[i]:
                                lbls[l] = 0
                            else:
                                lbls[l] = 1
                        else:
                            if lbls[l].item() in leaf_node_labels[i]:
                                lbls[l] = leaf_node_labels[i].index(lbls[l])
                            else:
                                lbls[l] = len(leaf_node_labels[i])

                    l = losses[i](result, lbls)
                    l.backward(retain_graph=True)
                    optims[i].step()
                    losses_to_print.append(l)
                else:
                    layer = models[k](layer)

        if batch_idx % args.log_interval == 0:
            p_str = 'Train Epoch: {} [{}/{} ({:.0f}%)]'
            for loss in losses_to_print:
                p_str += '\tLoss: {:.6f}'.format(loss.item())

            print(p_str.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                       100. * batch_idx / len(train_loader)))


def test_dynamic_tree(models, leaf_node_labels, test_loader, device, args):
    leaf_node_index = []
    leaf_node_paths = []  # NOT INCLUDING models[0]

    for i in range(len(models)):
        if not models[i] is None:
            models[i].eval()
            if isinstance(models[i], MobileTreeLeafNet):
                leaf_node_index.append(i)

    for i in leaf_node_index:
        path = []
        while i > 0:
            path = [i] + path
            i = (i + 1) // 2 - 1
        leaf_node_paths.append(path)

    definite_correct = 0
    indefinite_correct = 0
    wrong = 0

    for data, label in test_loader:
        data, labels = data.to(device), label.to(device)

        pred = []
        for i in range(len(leaf_node_paths)):  # for every branch(path) going to a leaf node
            layer = models[0](data)
            for j in range(len(leaf_node_paths[i])):
                k = leaf_node_paths[i][j]
                if j + 1 == len(leaf_node_paths[i]):
                    result, _ = models[k](layer)
                    pred.append(result.max(1, keepdim=True)[1])
                else:
                    layer = models[k](layer)

        for i in range(labels.size(0)):
            lbl = labels[i].item()
            ln_index = -1
            for j in range(len(leaf_node_labels)):
                if lbl in leaf_node_labels[j]:
                    k = leaf_node_labels[j].index(lbl)
                    ln_index = (j, k)
                    break
            if pred[ln_index[0]][i] == ln_index[1]:
                definite = True
                for j in range(len(leaf_node_index)):
                    if j != ln_index[0]:
                        if pred[j][i] != len(leaf_node_labels[j]):
                            definite = False
                if definite:
                    definite_correct += 1
                else:
                    indefinite_correct += 1
            else:
                wrong += 1

    if args.log:
        logging.info('Test set: Accuracy: {}/{} ({:.0f}%)\tDefinite Corrects: {}/{} ({:.0f}%)'.format(
            (definite_correct + indefinite_correct), len(test_loader.sampler),
            100. * (definite_correct + indefinite_correct) / len(test_loader.sampler),
            definite_correct, len(test_loader.sampler), 100. * definite_correct / len(test_loader.sampler)
        ))
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\tDefinite Corrects: {}/{} ({:.0f}%)\n'.format(
        (definite_correct + indefinite_correct), len(test_loader.sampler),
        100. * (definite_correct + indefinite_correct) / len(test_loader.sampler),
        definite_correct, len(test_loader.sampler), 100. * definite_correct / len(test_loader.sampler)
    ))


def train_net(model, train_loader, device, epoch, args):
    model.train()
    loss = torch.nn.CrossEntropyLoss()
    loss.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        optim.zero_grad()
        output = model(data)
        train_loss = loss(output, labels)
        train_loss.backward()
        optim.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                       100. * batch_idx / len(train_loader), train_loss.item()))


def test_net(model, test_loader, device, args):
    model.eval()
    loss = torch.nn.CrossEntropyLoss(size_average=False)
    loss.to(device)

    test_loss = 0
    correct = 0
    for data, label in test_loader:
        data, labels = data.to(device), label.to(device)
        output = model(data)
        test_loss += loss(output, labels).item()
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    if args.log:
        logging.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.sampler),
            100. * correct / len(test_loader.sampler)))
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.sampler),
        100. * correct / len(test_loader.sampler)))


def train_parallel_mobilenet(models, leaf_node_labels, train_loader, device, epoch, args):
    optims = []
    for model in models:
        model.train()
        optims.append(torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999)))

    lossfn = torch.nn.CrossEntropyLoss()
    lossfn.to(device)

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        lss_list = []
        for i in range(len(models)):
            optims[i].zero_grad()
            output = models[i](data)

            lbls = labels.clone()
            for l in range(len(lbls)):
                if lbls[l].item() in leaf_node_labels[i]:
                    lbls[l] = leaf_node_labels[i].index(lbls[l])
                else:
                    lbls[l] = len(leaf_node_labels[i])

            lss = lossfn(output, lbls)
            lss_list.append(lss)
            lss.backward()
            optims[i].step()

        if batch_idx % args.log_interval == 0:
            p_str = 'Train Epoch: {} [{}/{} ({:.0f}%)]'
            for loss in lss_list:
                p_str += '\tLoss: {:.6f}'.format(loss.item())

            print(p_str.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                       100. * batch_idx / len(train_loader)))


def test_parallel_mobilenet(models, leaf_node_labels, test_loader, device):
    for model in models:
        model.eval()

    definite_correct = 0
    indefinite_correct = 0
    wrong = 0

    for data, label in test_loader:
        data, labels = data.to(device), label.to(device)

        pred = []
        for i in range(len(models)):
            output = models[i](data)
            pred.append(output.max(1, keepdim=True)[1])

        for i in range(len(labels)):
            lbl = labels[i].item()
            ln_index = -1
            for j in range(len(leaf_node_labels)):
                if isinstance(leaf_node_labels[j], int):
                    if lbl == leaf_node_labels[j]:
                        ln_index = j
                        break
                else:
                    if lbl in leaf_node_labels[j]:
                        k = leaf_node_labels[j].index(lbl)
                        ln_index = (j, k)
                        break
            if (isinstance(ln_index, int) and pred[ln_index][i] == 0) or pred[ln_index[0]][i] == ln_index[1]:
                definite = True
                for j in range(len(leaf_node_labels)):
                    if (isinstance(ln_index, int) and j != ln_index) or j != ln_index[0]:
                        if (isinstance(leaf_node_labels[j], int) and pred[j][i] != 1) or pred[j][i] != len(
                                leaf_node_labels[j]):
                            definite = False
                if definite:
                    definite_correct += 1
                else:
                    indefinite_correct += 1
            else:
                wrong += 1

    print('Test set: Accuracy: {}/{} ({:.0f}%)\tDefinite Corrects: {}/{} ({:.0f}%)'.format(
        (definite_correct + indefinite_correct), len(test_loader.sampler),
        100. * (definite_correct + indefinite_correct) / len(test_loader.sampler),
        definite_correct, len(test_loader.sampler), 100. * definite_correct / len(test_loader.sampler)
    ))


def generate_model_list(root_node, level, device, fcl_factor):
    leaf_node_labels = []
    cfg_full = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]
    root_step = 1
    step = 3
    models = [MobileTreeRootNet(cfg_full[:root_step]).to(device)]
    nodes = [(root_node, 0)]
    index = 0
    remaining = 1
    # steps = [3,6,6,9,9,9,9,12,12,12,12,12,12,12,12,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
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

        for i in range(conv_step, len(cfg_full)):
            if isinstance(cfg_full[i], int):
                cfg_full[i] //= 2
            else:
                cfg_full[i] = (cfg_full[i][0] // 2, cfg_full[i][1])

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
    print(root_node)
    print(leaf_node_labels)
    for lbls in leaf_node_labels:
        print(len(lbls))
    return models, leaf_node_labels


def find_leaf_node_labels(root_node, level):
    leaf_node_labels = []

    search = [(root_node, 0)]
    while search:
        i = search.pop()
        node = i[0]
        lvl = i[1] + 1
        left = node.left
        right = node.right

        if isinstance(left, int):
            leaf_node_labels.append((left,))
        else:
            if left.count > 3 and lvl < level:
                search.append((left, lvl))
            else:
                leaf_node_labels.append(left.value)

        if isinstance(right, int):
            leaf_node_labels.append((right,))
        else:
            if right.count > 3 and lvl < level:
                search.append((right, lvl))
            else:
                leaf_node_labels.append(right.value)

    return leaf_node_labels


def calculate_indices(data, no_classes, load_indices, train_or_val):
    indices = []
    if load_indices:
        if train_or_val == 0:
            indices = np.load('train_indices.npy')
        elif train_or_val == 1:
            indices = np.load('val_indices.npy')
        print("Indices Loaded")
    else:
        print("Calculating Indices...")
        for i in range(len(data)):
            _, label = data[i]
            if label < no_classes:
                indices.append(i)
            if i % 50000 == 0:
                print('{}/{} ({:.0f}%)'.format(i, len(data), 100. * i / len(data)))
        print("Calculation Done")
        if train_or_val == 0:
            np.save('train_indices.npy', indices)
        elif train_or_val == 1:
            np.save('val_indices.npy', indices)
    return indices


def main():
    batch_size = 64
    test_batch_size = 1000
    epochs = 20
    lr = 0.002
    depth = 4
    resize = 256

    parser = argparse.ArgumentParser(description="Parameters for Training CIFAR-10")
    parser.add_argument('--test', action='store_true', help='enables test mode')
    parser.add_argument('--resume', action='store_true', help='whether to resume training or not (default: 0)')
    parser.add_argument('--log', action='store_true', help='log the events')
    parser.add_argument('--same', action='store_true', help='use same user preference table to generate the tree')
    parser.add_argument('--no-weights', action='store_true', help='train without class weights')
    parser.add_argument('--resize', type=int, default=resize, metavar='rsz', help='resize images in the dataset')
    parser.add_argument('--mobile-net', action='store_true', help='train mobile-net instead of tree-net')
    parser.add_argument('--parallel-mobile-nets', action='store_true', help='train parallel-mobile-net instead of tree-net')
    parser.add_argument('--mobile-static-tree-net', action='store_true', help='train mobile-static-tree-net instead of tree-net')
    parser.add_argument('--mobile-tree-net', action='store_true', help='train mobile-tree-net instead of tree-net')
    parser.add_argument('--mobile-tree-net-old', action='store_true', help='train mobile-tree-net-old instead of tree-net')
    parser.add_argument('--depth', type=int, default=depth, choices=[1, 2, 3, 4], metavar='lvl', help='depth of the tree')
    parser.add_argument('--batch-size', type=int, default=batch_size, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=test_batch_size, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=epochs, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=lr, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--num-workers', type=int, default=1, metavar='N', help='number of workers for cuda')
    parser.add_argument('--weight-mult', type=float, default=1.0, metavar='N', help='class weight multiplier')
    parser.add_argument('--pref-prob', type=float, default=0.3, metavar='N', help='class weight multiplier')
    parser.add_argument('--num-classes', type=int, default=365, metavar='N', help='train for only first n classes')
    parser.add_argument('--load-indices', action='store_true', help='skip the calculation of indices by loading the last calculation')
    parser.add_argument('--fine-tune', action='store_true', help='log the events')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
    args = parser.parse_args()

    test = args.test
    resume = args.resume
    same = args.same

    no_classes = args.num_classes

    if args.log:
        start_time = time.time()
        logging.basicConfig(filename="mainlog.log", level=logging.INFO)
        logging.info("---START---\n")
        logging.info(time.asctime(time.localtime(start_time)))

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cuda_args = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

    traindir = os.path.join('../places365/places365_standard', 'train')
    valdir = os.path.join('../places365/places365_standard', 'val')

    train_data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.4),
        transforms.RandomRotation(20),
        transforms.RandomAffine(45, (0.2, 0.2)),
        transforms.Resize((args.resize ,args.resize)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    val_data_transform = transforms.Compose([
        transforms.Resize((args.resize, args.resize)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    places_training_data = datasets.ImageFolder(traindir, transform=train_data_transform)
    places_validation_data = datasets.ImageFolder(valdir, transform=val_data_transform)
    if no_classes == 365:
        train_loader = torch.utils.data.DataLoader(places_training_data, batch_size=args.batch_size, shuffle=True, **cuda_args)
        val_loader = torch.utils.data.DataLoader(places_validation_data, batch_size=args.test_batch_size, shuffle=True, **cuda_args)
    else:
        train_indices = calculate_indices(places_training_data, no_classes, args.load_indices, train_or_val=0)
        test_indices = calculate_indices(places_validation_data, no_classes, args.load_indices, train_or_val=1)
        train_loader = torch.utils.data.DataLoader(places_training_data, batch_size=args.batch_size,
                                                   sampler=SubsetRandomSampler(train_indices), **cuda_args)
        val_loader = torch.utils.data.DataLoader(places_validation_data, batch_size=args.test_batch_size,
                                                   sampler=SubsetRandomSampler(test_indices), **cuda_args)

    fcl_factor = args.resize // 32

    if args.mobile_net:
        model = MobileNet(num_classes=no_classes, fcl=(fcl_factor*fcl_factor*1024)).to(device)
        if args.log:
            logging.info("Mobile-Net")
            if resume:
                logging.info("resume")
            elif test:
                logging.info("test")
            logging.info("Learning Rate: " + str(args.lr))
            logging.info("Epochs: " + str(args.epochs))
            logging.info("Batch Size: " + str(args.batch_size))
            logging.info("Size of Images: " + str(args.resize))
            logging.info("Number of Classes: " + str(no_classes))
        if not test:
            if resume:
                model.load_state_dict(torch.load('./saved/mobilenet.pth'))
            for epoch in range(1, args.epochs + 1):
                train_net(model, train_loader, device, epoch, args)
                test_net(model, val_loader, device, args)
            torch.save(model.state_dict(), './saved/mobilenet.pth')
        else:
            model.load_state_dict(torch.load('./saved/mobilenet.pth'))
            test_net(model, val_loader, device, args)
    elif args.mobile_static_tree_net:
        models = [StaticTreeRootNet().to(device), StaticTreeBranchNet().to(device), StaticTreeBranchNet().to(device)]
        # LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

        if not test:
            if resume:
                models[0].load_state_dict(torch.load('./saved/root.pth'))
                models[1].load_state_dict(torch.load('./saved/branch1.pth'))
                models[2].load_state_dict(torch.load('./saved/branch2.pth'))
            for epoch in range(1, args.epochs + 1):
                train_tree(models, train_loader, device, epoch, args)
                test_tree(models, val_loader, device)

            torch.save(models[0].state_dict(), './saved/root.pth')
            torch.save(models[1].state_dict(), './saved/branch1.pth')
            torch.save(models[2].state_dict(), './saved/branch2.pth')

        if test:
            models[0].load_state_dict(torch.load('./saved/root.pth'))
            models[1].load_state_dict(torch.load('./saved/branch1.pth'))
            models[2].load_state_dict(torch.load('./saved/branch2.pth'))

            test_tree(models, val_loader, device)
    elif args.mobile_tree_net:
        print("Mobile Tree Net\n")
        load = resume or test or same
        root_node = utils.generate(365, 1000, load, prob=0.3)
        models, leaf_node_labels = generate_model_list(root_node, args.depth, device, fcl_factor)

        if not test:
            if resume:
                for i in range(len(models)):
                    if not models[i] is None:
                        models[i].load_state_dict(torch.load('./saved/treemodel' + str(i) + '.pth'))
            for epoch in range(1, args.epochs + 1):
                train_dynamic_tree(models, leaf_node_labels, train_loader, device, epoch, args)
                test_dynamic_tree(models, leaf_node_labels, val_loader, device, args)

            for i in range(len(models)):
                if not models[i] is None:
                    torch.save(models[i].state_dict(), './saved/treemodel' + str(i) + '.pth')

        if test:
            for i in range(len(models)):
                if not models[i] is None:
                    models[i].load_state_dict(torch.load('./saved/treemodel' + str(i) + '.pth'))

            test_dynamic_tree(models, leaf_node_labels, val_loader, device, args)
    elif args.mobile_tree_net_old:
        print("Mobile Tree Net Old\n")
        load = resume or test or same
        if no_classes == 365:
            root_node = utils.generate(365, 1000, load, prob=args.pref_prob)
        else:
            root_node = utils.generate(no_classes, no_classes*5, load, prob=args.pref_prob)
        models, leaf_node_labels = generate_model_list(root_node, args.depth, device, fcl_factor)
        if args.log:
            logging.info("Mobile Tree Net Old")
            for lbls in leaf_node_labels:
                logging.info(len(lbls))
            if resume:
                logging.info("resume")
            elif test:
                logging.info("test")
            elif same:
                logging.info("same")
            logging.info("Learning Rate: " + str(args.lr))
            logging.info("Depth: " + str(args.depth))
            logging.info("Epochs: " + str(args.epochs))
            logging.info("Batch Size: " + str(args.batch_size))
            logging.info("Size of Images: " + str(args.resize))
            logging.info("Number of Classes: " + str(no_classes))
        if not test:
            if args.fine_tune:
                for i in range(len(models)):
                    if not models[i] is None:
                        models[i].load_state_dict(torch.load('./saved/treemodel' + str(i) + '.pth'))
                for epoch in range(1, args.epochs + 1):
                    train_hierarchical(models, leaf_node_labels, train_loader, device, epoch, args, use_cuda, args.depth, args.depth)
                    test_dynamic_tree(models, leaf_node_labels, val_loader, device, args)
                for i in range(len(models)):
                    if not models[i] is None:
                        torch.save(models[i].state_dict(), './saved/treemodel' + str(i) + '.pth')
            else:
                if resume:
                    for i in range(len(models)):
                        if not models[i] is None:
                            models[i].load_state_dict(torch.load('./saved/treemodel' + str(i) + '.pth'))
                for epoch in range(1, args.epochs + 1):
                    train_dynamic_tree_old(models, leaf_node_labels, train_loader, device, epoch, args, use_cuda)
                    test_dynamic_tree(models, leaf_node_labels, val_loader, device, args)
                for i in range(len(models)):
                    if not models[i] is None:
                        torch.save(models[i].state_dict(), './saved/treemodel' + str(i) + '.pth')

        if test:
            for i in range(len(models)):
                if not models[i] is None:
                    models[i].load_state_dict(torch.load('./saved/treemodel' + str(i) + '.pth'))

            test_dynamic_tree(models, leaf_node_labels, val_loader, device, args)
    elif args.parallel_mobile_nets:
        cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]
        root_node = utils.generate(10, 80, resume)
        leaf_node_labels = find_leaf_node_labels(root_node, args.depth)
        for i in range(len(cfg)):
            cfg[i] = cfg[i] // len(leaf_node_labels) if isinstance(cfg[i], int) else (
            cfg[i][0] // len(leaf_node_labels), cfg[i][1])
        models = []
        for i in leaf_node_labels:
            branches = 2 if isinstance(i, int) else len(i) + 1
            models.append(MobileNet(num_classes=branches, channels=cfg, fcl=((fcl_factor*fcl_factor*1024) // len(leaf_node_labels))).to(device))

        print(root_node)

        if not test:
            if resume:
                for i in range(len(models)):
                    models[i].load_state_dict(torch.load('./saved/parallel_mobilenet' + str(i) + '.pth'))
            for epoch in range(1, args.epochs + 1):
                train_parallel_mobilenet(models, leaf_node_labels, train_loader, device, epoch, args)
                test_parallel_mobilenet(models, leaf_node_labels, val_loader, device)

            for i in range(len(models)):
                torch.save(models[i].state_dict(), './saved/parallel_mobilenet' + str(i) + '.pth')
        else:
            for i in range(len(models)):
                models[i].load_state_dict(torch.load('./saved/parallel_mobilenet' + str(i) + '.pth'))
            test_parallel_mobilenet(models, leaf_node_labels, val_loader, device)
    else:
        models = [TreeRootNet().to(device), TreeBranchNet().to(device), TreeBranchNet().to(device)]
        # LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

        if not test:
            if resume:
                models[0].load_state_dict(torch.load('./saved/root.pth'))
                models[1].load_state_dict(torch.load('./saved/branch1.pth'))
                models[2].load_state_dict(torch.load('./saved/branch2.pth'))
            for epoch in range(1, args.epochs + 1):
                train_tree(models, train_loader, device, epoch, args)
                test_tree(models, val_loader, device)

            torch.save(models[0].state_dict(), './saved/root.pth')
            torch.save(models[1].state_dict(), './saved/branch1.pth')
            torch.save(models[2].state_dict(), './saved/branch2.pth')

        if test:
            models[0].load_state_dict(torch.load('./saved/root.pth'))
            models[1].load_state_dict(torch.load('./saved/branch1.pth'))
            models[2].load_state_dict(torch.load('./saved/branch2.pth'))

            test_tree(models, val_loader, device)

    if args.log:
        end_time = time.time()
        logging.info(time.asctime(time.localtime(end_time)))
        logging.info("--- %s seconds ---" % (end_time - start_time))
        logging.info("---END---\n")


if __name__ == '__main__':
    main()
