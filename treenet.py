import argparse
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

import logging
import time

from models.mobilenet import MobileNet
from models.mobile_tree_net import MobileTreeRootNet, MobileTreeLeafNet, MobileTreeBranchNet
import torchvision.models as Models

import utils
import os

class_labels = [89, 168, 203, 244, 254, 268, 284, 298, 320, 321]
best_acc = 0
vis = 0


def map_labels(labels):
    lbls = labels.clone()
    for i in range(len(lbls)):
        lbls[i] = class_labels.index(lbls[i].item())
    return lbls


def train_tree(models, leaf_node_labels, train_loader, device, epoch, args, use_cuda):
    leaf_node_index = []

    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    list_of_model_params = list()
    for i in range(len(models)):
        if not models[i] is None:
            models[i].train()
            list_of_model_params += models[i].parameters()
            if isinstance(models[i], MobileTreeLeafNet):
                leaf_node_index.append(i)

    losses = []
    for ls in leaf_node_labels:
        if args.no_weights:
            losses.append(torch.nn.CrossEntropyLoss().to(device))
        else:
            weights = [1.0] * (len(ls) + 1)
            weights[-1] = args.weight_mult / (args.num_classes - len(ls))
            losses.append(torch.nn.CrossEntropyLoss(weight=FloatTensor(weights)).to(device))

    if args.adam:
        optim = torch.optim.Adam(list_of_model_params, lr=args.lr)
    else:
        optim = torch.optim.SGD(list_of_model_params, lr=args.lr, momentum=0.9, weight_decay=5e-4)

    definite_correct = 0
    avg_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        optim.zero_grad()

        pred = []
        losses_to_print = []
        leaf_node_results = []
        sum_of_losses = 0
        results = [None] * len(models)
        results[0] = models[0](data)
        for i in range(1, len(models)):
            if not models[i] is None:
                prev = (i + 1) // 2 - 1
                if i in leaf_node_index:
                    res, _ = models[i](results[prev])
                    results[i] = res
                    leaf_node_results.append(res)
                    if not args.fast_train:
                        pred.append(res.max(1, keepdim=True)[1])
                else:
                    results[i] = models[i](results[prev])

        for i in range(len(leaf_node_results)):
            lbls = labels.clone()
            for l in range(len(lbls)):
                if lbls[l].item() in leaf_node_labels[i]:
                    lbls[l] = leaf_node_labels[i].index(lbls[l])
                else:
                    lbls[l] = len(leaf_node_labels[i])

            l = losses[i](leaf_node_results[i], lbls)
            sum_of_losses += l
            losses_to_print.append(l.item())

        sum_of_losses.backward()
        optim.step()

        avg_loss += (sum(losses_to_print) / float(len(losses_to_print)))

        if not args.fast_train:
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

        if batch_idx % args.log_interval == 0:
            p_str = 'Train Epoch: {} [{}/{} ({:.0f}%)]'
            for loss in losses_to_print:
                p_str += '\tLoss: {:.6f}'.format(loss)

            print(p_str.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                       100. * batch_idx / len(train_loader)))

    acc = 100. * definite_correct / len(train_loader.sampler)
    if not args.val_mode and epoch == args.epochs:
        for i in range(len(models)):
            if not models[i] is None:
                saveModel(models[i], acc, epoch, './saved/treemodel' + str(i) + '.pth')
        print("Model Saved!")

    avg_loss /= len(train_loader)
    if args.visdom:
        vis.plot_loss(avg_loss, epoch, name='train_loss')
        vis.plot_acc(acc, epoch, name='train_acc')
    if args.log:
        logging.info('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            avg_loss, definite_correct, len(train_loader.sampler), acc))
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            avg_loss, definite_correct, len(train_loader.sampler), acc))


def train_tree_old(models, leaf_node_labels, train_loader, device, epoch, args, use_cuda):
    leaf_node_index = []
    leaf_node_paths = []

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
            weights[-1] = args.weight_mult / (args.num_classes - len(leaf_node_labels[j]))
            losses.append(torch.nn.CrossEntropyLoss(weight=FloatTensor(weights)).to(device))

        if args.adam:
            optims.append(torch.optim.Adam(model_path, lr=args.lr))
        else:
            optims.append(torch.optim.SGD(model_path, lr=args.lr, momentum=0.9, weight_decay=5e-4))

    definite_correct = 0
    avg_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        if args.use_classes:
            labels = map_labels(labels).to(device)

        pred = []
        losses_to_print = []
        for i in range(len(leaf_node_paths)):
            optims[i].zero_grad()

            lbls = labels.clone()
            for l in range(len(lbls)):
                if lbls[l].item() in leaf_node_labels[i]:
                    lbls[l] = leaf_node_labels[i].index(lbls[l])
                else:
                    lbls[l] = len(leaf_node_labels[i])

            layer = models[0](data)
            for j in range(len(leaf_node_paths[i]) - 1):
                k = leaf_node_paths[i][j]
                layer = models[k](layer)
            k = leaf_node_index[i]
            result, _ = models[k](layer)
            if not args.fast_train:
                pred.append(result.max(1, keepdim=True)[1])

            l = losses[i](result, lbls)
            l.backward(retain_graph=True)
            optims[i].step()
            losses_to_print.append(l.item())

        avg_loss += (sum(losses_to_print) / float(len(losses_to_print)))

        if not args.fast_train:
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

        if batch_idx % args.log_interval == 0:
            p_str = 'Train Epoch: {} [{}/{} ({:.0f}%)]'
            for loss in losses_to_print:
                p_str += '\tLoss: {:.6f}'.format(loss)

            print(p_str.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                       100. * batch_idx / len(train_loader)))

    acc = 100. * definite_correct / len(train_loader.sampler)
    if not args.val_mode and epoch == args.epochs:
        for i in range(len(models)):
            if not models[i] is None:
                saveModel(models[i], acc, epoch, './saved/treemodel' + str(i) + '.pth')
        print("Model Saved!")

    avg_loss /= len(train_loader)
    if args.visdom:
        vis.plot_loss(avg_loss, epoch, name='train_loss')
        vis.plot_acc(acc, epoch, name='train_acc')

    if args.log:
        logging.info('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            avg_loss, definite_correct, len(train_loader.sampler), acc))
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            avg_loss, definite_correct, len(train_loader.sampler), acc))


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
        while i > 0:
            path = [i] + path
            i = (i+1)//2 - 1
        if (2 ** (max_depth + 1)) > 1 >= (2 ** min_depth):
            model_path = list(models[0].parameters())
        else:
            model_path =list()
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

        losses_to_print = []
        for i in range(len(leaf_node_paths)):   # for every branch(path) going to a leaf node
            optims[i].zero_grad()

            lbls = labels.clone()
            for l in range(len(lbls)):
                if lbls[l].item() in leaf_node_labels[i]:
                    lbls[l] = leaf_node_labels[i].index(lbls[l])
                else:
                    lbls[l] = len(leaf_node_labels[i])

            layer = models[0](data)
            for j in range(len(leaf_node_paths[i])-1):
                k = leaf_node_paths[i][j]
                layer = models[k](layer)
            k = leaf_node_index[i]
            result, _ = models[k](layer)

            l = losses[i](result, lbls)
            l.backward(retain_graph=True)
            optims[i].step()
            losses_to_print.append(l)

        if batch_idx % args.log_interval == 0:
            p_str = 'Train Epoch: {} [{}/{} ({:.0f}%)]'
            for loss in losses_to_print:
                p_str += '\tLoss: {:.6f}'.format(loss.item())

            print(p_str.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                       100. * batch_idx / len(train_loader)))


def test_tree(models, leaf_node_labels, test_loader, device, args, epoch=0):
    global best_acc
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
    avg_loss = 0
    for data, label in test_loader:
        data, labels = data.to(device), label.to(device)
        if args.use_classes:
            labels = map_labels(labels).to(device)

        pred = []
        losses_to_print = []
        leaf_node_results = []
        sum_of_losses = 0
        results = [None] * len(models)
        results[0] = models[0](data)
        for i in range(1, len(models)):
            if not models[i] is None:
                prev = (i + 1) // 2 - 1
                if i in leaf_node_index:
                    res, _ = models[i](results[prev])
                    results[i] = res
                    pred.append(res.max(1, keepdim=True)[1])
                    if not args.fast_train:
                        leaf_node_results.append(res)
                else:
                    results[i] = models[i](results[prev])

        if not args.fast_train:
            use_cuda = torch.cuda.is_available()
            FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
            losses = []
            for ls in leaf_node_labels:
                if args.no_weights:
                    losses.append(torch.nn.CrossEntropyLoss().to(device))
                else:
                    weights = [1.0] * (len(ls) + 1)
                    weights[-1] = args.weight_mult / (args.num_classes - len(ls))
                    losses.append(torch.nn.CrossEntropyLoss(weight=FloatTensor(weights)).to(device))
            for i in range(len(leaf_node_results)):
                lbls = labels.clone()
                for l in range(len(lbls)):
                    if lbls[l].item() in leaf_node_labels[i]:
                        lbls[l] = leaf_node_labels[i].index(lbls[l])
                    else:
                        lbls[l] = len(leaf_node_labels[i])

                l = losses[i](leaf_node_results[i], lbls)
                sum_of_losses += l
                losses_to_print.append(l.item())
            avg_loss += (sum(losses_to_print) / float(len(losses_to_print)))

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

    acc = 100. * definite_correct / len(test_loader.sampler)
    if args.val_mode and acc > best_acc:
        best_acc = acc
        for i in range(len(models)):
            if not models[i] is None:
                saveModel(models[i], acc, epoch, './saved/treemodel' + str(i) + '.pth')
        print("Model Saved!")

    avg_loss /= len(test_loader)
    if args.visdom:
        vis.plot_loss(avg_loss, epoch, name='val_loss')
        vis.plot_acc(acc, epoch, name='val_acc')

    if args.log:
        logging.info('Test set: Accuracy: {}/{} ({:.2f}%)\tDefinite Corrects: {}/{} ({:.2f}%)\tAvg loss: {:.4f}'.format(
            (definite_correct + indefinite_correct), len(test_loader.sampler),
            100. * (definite_correct + indefinite_correct) / len(test_loader.sampler),
            definite_correct, len(test_loader.sampler), acc, avg_loss
        ))
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\tDefinite Corrects: {}/{} ({:.2f}%)\tAvg loss: {:.4f}\n'.format(
        (definite_correct + indefinite_correct), len(test_loader.sampler),
        100. * (definite_correct + indefinite_correct) / len(test_loader.sampler),
        definite_correct, len(test_loader.sampler), acc, avg_loss
    ))


def test_tree_personal(models, leaf_node_labels, test_loader, device, args, preferences):
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

    correct = 0
    wrong = 0

    for data, label in test_loader:
        data, labels = data.to(device), label.to(device)
        if args.use_classes:
            labels = map_labels(labels).to(device)

        pred = []   # indices are predictions(vector(size of a single batch)) for each leaf (Note that: None for the leaves not used)
        used_ln_indexes = []    #used paths indexed from left to right in order
        for i in range(len(leaf_node_paths)):  # for every branch(path) going to a leaf node
            if not any(elem in leaf_node_labels[i] for elem in preferences):
                pred.append(None)
                continue
            used_ln_indexes.append(i)
            layer = models[0](data)
            for j in range(len(leaf_node_paths[i])):
                k = leaf_node_paths[i][j]
                if j + 1 == len(leaf_node_paths[i]):
                    result, _ = models[k](layer)
                    pred.append(result.max(1, keepdim=True)[1])
                else:
                    layer = models[k](layer)

        for i in range(len(labels)):
            lbl = labels[i].item()
            ln_index = -1
            for j in range(len(leaf_node_labels)):
                if lbl in leaf_node_labels[j]:
                    k = leaf_node_labels[j].index(lbl)
                    ln_index = (j, k)
                    break
            if ln_index[0] not in used_ln_indexes:
                definite = True
                for j in range(len(leaf_node_index)):
                    if j in used_ln_indexes:
                        if pred[j][i] != len(leaf_node_labels[j]):
                            definite = False
                if definite:
                    correct += 1
                else:
                    wrong += 1
            else:
                if pred[ln_index[0]][i] == ln_index[1] or ((lbl not in preferences) and pred[ln_index[0]][i] not in preferences):
                    definite = True
                    for j in range(len(leaf_node_index)):
                        if j in used_ln_indexes and j != ln_index[0]:
                            if pred[j][i] != len(leaf_node_labels[j]):
                                definite = False
                    if definite:
                        correct += 1
                    else:
                        wrong += 1
                else:
                    wrong += 1

    if args.log:
        logging.info('Test set: Accuracy: {}/{} ({:.2f}%)'.format(
            correct, len(test_loader.sampler),
            100. * correct / len(test_loader.sampler)
        ))
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, len(test_loader.sampler),
        100. * correct / len(test_loader.sampler)
    ))


def test_tree_all_preferences(models, leaf_node_labels, test_loader, device, args, all_prefs):
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

    corrects = [0] * len(all_prefs)

    for data, label in test_loader:
        data, labels = data.to(device), label.to(device)
        if args.use_classes:
            labels = map_labels(labels).to(device)

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

        for p, single_pref in enumerate(all_prefs):
            correct = 0

            for i in range(len(labels)):
                lbl = labels[i].item()
                is_correct = True
                for j, single_leaf_labels in enumerate(leaf_node_labels):
                    if any(elem in single_leaf_labels for elem in single_pref):
                        if lbl in single_pref:
                            if lbl in single_leaf_labels:
                                if not pred[j][i] == single_leaf_labels.index(lbl):
                                    is_correct = False
                            else:
                                if (not pred[j][i] == len(single_leaf_labels)) and (single_leaf_labels[pred[j][i]] in single_pref):
                                    is_correct = False
                        else:
                            if (not pred[j][i] == len(single_leaf_labels)) and (single_leaf_labels[pred[j][i]] in single_pref):
                                is_correct = False
                if is_correct:
                    correct += 1

            corrects[p] += correct

    accuracies = [100. * c / len(test_loader.sampler) for c in corrects]
    avg_acc = sum(accuracies)/float(len(accuracies))

    if args.log:
        logging.info('Test set: Accuracy: {:.2f}%'.format(avg_acc))

    print(accuracies)
    print('\nTest set: Accuracy: {:.2f}%\n'.format(avg_acc))


def train_net(model, train_loader, device, epoch, args, save_name="net"):
    model.train()
    loss = torch.nn.CrossEntropyLoss()
    loss.to(device)

    if args.adam:
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    losses = 0
    correct = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        if args.use_classes:
            labels = map_labels(labels).to(device)

        optim.zero_grad()
        output = model(data)
        train_loss = loss(output, labels)
        train_loss.backward()
        optim.step()

        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).sum().item()
        losses += train_loss.item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                       100. * batch_idx / len(train_loader), train_loss.item()))

    acc = 100. * correct / len(train_loader.sampler)
    losses /= len(train_loader)

    if not args.val_mode and epoch == args.epochs:
        saveModel(model, acc, epoch, './saved/' + save_name + '.pth')
        print("Model Saved!")

    if args.visdom:
        vis.plot_loss(losses, epoch, name='train_loss')
        vis.plot_acc(acc, epoch, name='train_acc')
    if args.log:
        logging.info('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            losses, correct, len(train_loader.sampler), acc))
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        losses, correct, len(train_loader.sampler), acc))


def test_net(model, test_loader, device, args, epoch=0, save_name="net"):
    global best_acc
    model.eval()
    loss = torch.nn.CrossEntropyLoss()
    loss.to(device)

    test_loss = 0
    correct = 0
    for data, label in test_loader:
        data, labels = data.to(device), label.to(device)
        if args.use_classes:
            labels = map_labels(labels).to(device)
        output = model(data)
        test_loss += loss(output, labels).item()
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    acc = 100. * correct / len(test_loader.sampler)
    if args.val_mode and acc > best_acc:
        best_acc = acc
        saveModel(model, acc, epoch, './saved/' + save_name + '.pth')
        print("Model Saved!")

    if args.visdom:
        vis.plot_loss(test_loss, epoch, name='val_loss')
        vis.plot_acc(acc, epoch, name='val_acc')
    if args.log:
        logging.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, len(test_loader.sampler), acc))
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.sampler), acc))


def test_net_all_preferences(model, test_loader, device, args, all_prefs):
    model.eval()

    corrects = [0] * len(all_prefs)
    for data, label in test_loader:
        data, labels = data.to(device), label.to(device)
        if args.use_classes:
            labels = map_labels(labels).to(device)

        output = model(data)
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

        for p, single_pref in enumerate(all_prefs):
            correct = 0
            for i in range(len(labels)):
                lbl = labels[i].item()
                if lbl in single_pref:
                    if pred[i] == lbl:
                        correct += 1
                else:
                    if pred[i] not in single_pref:
                        correct += 1

            corrects[p] += correct

    accuracies = [100. * c / len(test_loader.sampler) for c in corrects]
    avg_acc = sum(accuracies) / float(len(accuracies))

    if args.log:
        logging.info('Test set: Accuracy: {:.2f}%'.format(avg_acc))

    print(accuracies)
    print('\nTest set: Accuracy: {:.2f}%\n'.format(avg_acc))


def train_parallel_mobilenet(models, leaf_node_labels, train_loader, device, epoch, args, use_cuda):
    optims = []
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    losses = []
    for i, model in enumerate(models):
        model.train()
        if args.no_weights:
            losses.append(torch.nn.CrossEntropyLoss().to(device))
        else:
            weights = [1.0] * (len(leaf_node_labels[i]) + 1)
            weights[-1] = args.weight_mult / (args.num_classes - len(leaf_node_labels[i]))
            losses.append(torch.nn.CrossEntropyLoss(weight=FloatTensor(weights)).to(device))
        if args.adam:
            optims.append(torch.optim.Adam(model.parameters(), lr=args.lr))
        else:
            optims.append(torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4))

    definite_correct = 0
    avg_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        if args.use_classes:
            labels = map_labels(labels).to(device)

        lss_list = []
        pred = []
        for i in range(len(models)):
            optims[i].zero_grad()
            output = models[i](data)
            if not args.fast_train:
                pred.append(output.max(1, keepdim=True)[1])

            lbls = labels.clone()
            for l in range(len(lbls)):
                if lbls[l].item() in leaf_node_labels[i]:
                    lbls[l] = leaf_node_labels[i].index(lbls[l])
                else:
                    lbls[l] = len(leaf_node_labels[i])

            lss = losses[i](output, lbls)
            lss_list.append(lss.item())
            lss.backward()
            optims[i].step()

        avg_loss += (sum(lss_list) / float(len(lss_list)))

        if not args.fast_train:
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
                    for j in range(len(models)):
                        if j != ln_index[0]:
                            if pred[j][i] != len(leaf_node_labels[j]):
                                definite = False
                    if definite:
                        definite_correct += 1

        if batch_idx % args.log_interval == 0:
            p_str = 'Train Epoch: {} [{}/{} ({:.0f}%)]'
            for loss in lss_list:
                p_str += '\tLoss: {:.6f}'.format(loss)

            print(p_str.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                       100. * batch_idx / len(train_loader)))

    acc = 100. * definite_correct / len(train_loader.sampler)
    if not args.val_mode and epoch == args.epochs:
        for i in range(len(models)):
            saveModel(models[i], acc, epoch, './saved/parallel_mobilenet' + str(i) + '.pth')
        print("Model Saved!")

    avg_loss /= len(train_loader)
    if args.visdom:
        vis.plot_loss(avg_loss, epoch, name='train_loss')
        vis.plot_acc(acc, epoch, name='train_acc')

    if args.log:
        logging.info('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            avg_loss, definite_correct, len(train_loader.sampler), acc))
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            avg_loss, definite_correct, len(train_loader.sampler), acc))


def test_parallel_mobilenet(models, leaf_node_labels, test_loader, device, args, epoch=0):
    global best_acc
    for model in models:
        model.eval()

    definite_correct = 0
    indefinite_correct = 0
    wrong = 0
    avg_loss = 0
    for data, label in test_loader:
        data, labels = data.to(device), label.to(device)
        if args.use_classes:
            labels = map_labels(labels).to(device)

        pred = []
        losses_to_print = []
        sum_of_losses = 0
        leaf_node_results = []
        for i in range(len(models)):
            output = models[i](data)
            pred.append(output.max(1, keepdim=True)[1])
            if not args.fast_train:
                leaf_node_results.append(output)

        if not args.fast_train:
            use_cuda = torch.cuda.is_available()
            FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
            losses = []
            for ls in leaf_node_labels:
                if args.no_weights:
                    losses.append(torch.nn.CrossEntropyLoss().to(device))
                else:
                    weights = [1.0] * (len(ls) + 1)
                    weights[-1] = args.weight_mult / (args.num_classes - len(ls))
                    losses.append(torch.nn.CrossEntropyLoss(weight=FloatTensor(weights)).to(device))
            for i in range(len(leaf_node_results)):
                lbls = labels.clone()
                for l in range(len(lbls)):
                    if lbls[l].item() in leaf_node_labels[i]:
                        lbls[l] = leaf_node_labels[i].index(lbls[l])
                    else:
                        lbls[l] = len(leaf_node_labels[i])

                l = losses[i](leaf_node_results[i], lbls)
                sum_of_losses += l
                losses_to_print.append(l.item())
            avg_loss += (sum(losses_to_print) / float(len(losses_to_print)))

        for i in range(len(labels)):
            lbl = labels[i].item()
            ln_index = -1
            for j in range(len(leaf_node_labels)):
                if lbl in leaf_node_labels[j]:
                    k = leaf_node_labels[j].index(lbl)
                    ln_index = (j, k)
                    break
            if pred[ln_index[0]][i] == ln_index[1]:
                definite = True
                for j in range(len(leaf_node_labels)):
                    if j != ln_index[0]:
                        if pred[j][i] != len(leaf_node_labels[j]):
                            definite = False
                if definite:
                    definite_correct += 1
                else:
                    indefinite_correct += 1
            else:
                wrong += 1

    acc = 100. * definite_correct / len(test_loader.sampler)
    if args.val_mode and acc > best_acc:
        best_acc = acc
        for i in range(len(models)):
            saveModel(models[i], acc, epoch, './saved/parallel_mobilenet' + str(i) + '.pth')

    avg_loss /= len(test_loader)
    if args.visdom:
        vis.plot_loss(avg_loss, epoch, name='val_loss')
        vis.plot_acc(acc, epoch, name='val_acc')

    if args.log:
        logging.info('Test set: Accuracy: {}/{} ({:.2f}%)\tDefinite Corrects: {}/{} ({:.2f}%)\tAvg loss: {:.4f}'.format(
        (definite_correct + indefinite_correct), len(test_loader.sampler),
        100. * (definite_correct + indefinite_correct) / len(test_loader.sampler),
        definite_correct, len(test_loader.sampler), acc, avg_loss
        ))
    print('Test set: Accuracy: {}/{} ({:.2f}%)\tDefinite Corrects: {}/{} ({:.2f}%)\tAvg loss: {:.4f}\n'.format(
        (definite_correct + indefinite_correct), len(test_loader.sampler),
        100. * (definite_correct + indefinite_correct) / len(test_loader.sampler),
        definite_correct, len(test_loader.sampler), acc, avg_loss
    ))


def test_parallel_personal(models, leaf_node_labels, test_loader, device, args, preferences):
    for model in models:
        model.eval()

    correct = 0
    wrong = 0

    for data, label in test_loader:
        data, labels = data.to(device), label.to(device)
        if args.use_classes:
            labels = map_labels(labels).to(device)

        pred = []
        used_ln_index = []
        for i in range(len(models)):
            if not any(elem in leaf_node_labels[i] for elem in preferences):
                pred.append(None)
            else:
                used_ln_index.append(i)
                output = models[i](data)
                pred.append(output.max(1, keepdim=True)[1])

        for i in range(len(labels)):
            lbl = labels[i].item()
            ln_index = -1
            for j in range(len(leaf_node_labels)):
                if lbl in leaf_node_labels[j]:
                    k = leaf_node_labels[j].index(lbl)
                    ln_index = (j, k)
                    break
            if ln_index[0] not in used_ln_index:
                definite = True
                for j in range(len(leaf_node_labels)):
                    if j in used_ln_index:
                        if pred[j][i] != len(leaf_node_labels[j]):
                            definite = False
                if definite:
                    correct += 1
                else:
                    wrong += 1
            else:
                if pred[ln_index[0]][i] == ln_index[1] or ((lbl not in preferences) and pred[ln_index[0]][i] not in preferences):
                    definite = True
                    for j in range(len(leaf_node_labels)):
                        if j in used_ln_index and j != ln_index[0]:
                            if pred[j][i] != len(leaf_node_labels[j]):
                                definite = False
                    if definite:
                        correct += 1
                    else:
                        wrong += 1
                else:
                    wrong += 1
    if args.log:
        logging.info('Test set: Accuracy: {}/{} ({:.2f}%)'.format(
        correct, len(test_loader.sampler),
        100. * correct / len(test_loader.sampler)
        ))
    print('Test set: Accuracy: {}/{} ({:.2f}%)'.format(
        correct, len(test_loader.sampler),
        100. * correct / len(test_loader.sampler)
    ))


def test_parallel_all_preferences(models, leaf_node_labels, test_loader, device, args, all_prefs):
    for model in models:
        model.eval()

    corrects = [0] * len(all_prefs)
    for data, label in test_loader:
        data, labels = data.to(device), label.to(device)
        if args.use_classes:
            labels = map_labels(labels).to(device)

        pred = []
        for i in range(len(models)):
            output = models[i](data)
            pred.append(output.max(1, keepdim=True)[1])

        for p, single_pref in enumerate(all_prefs):
            correct = 0
            for i in range(len(labels)):
                lbl = labels[i].item()

                is_correct = True
                for j, single_leaf_labels in enumerate(leaf_node_labels):
                    if any(elem in single_leaf_labels for elem in single_pref):
                        if lbl in single_pref:
                            if lbl in single_leaf_labels:
                                if not pred[j][i] == single_leaf_labels.index(lbl):
                                    is_correct = False
                            else:
                                if (not pred[j][i] == len(single_leaf_labels)) and (
                                        single_leaf_labels[pred[j][i]] in single_pref):
                                    is_correct = False
                        else:
                            if (not pred[j][i] == len(single_leaf_labels)) and (
                                    single_leaf_labels[pred[j][i]] in single_pref):
                                is_correct = False
                if is_correct:
                    correct += 1

            corrects[p] += correct

    accuracies = [100. * c / len(test_loader.sampler) for c in corrects]
    avg_acc = sum(accuracies) / float(len(accuracies))

    if args.log:
        logging.info('Test set: Accuracy: {:.2f}%'.format(avg_acc))

    print(accuracies)
    print('\nTest set: Accuracy: {:.2f}%\n'.format(avg_acc))


def generate_model_list(root_node, level, device, fcl_factor, root_step=1, step=3, dividing_factor=2, not_involve=1, log=False):
    leaf_node_labels = []
    cfg_full = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]
    models = [MobileTreeRootNet(cfg_full[:root_step]).to(device)]
    nodes = [(root_node, 0)]
    index = 0
    remaining = 1
    prev_lvl = 0
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

        if prev_lvl < lvl:
            prev_lvl = lvl
            for i in range(conv_step, len(cfg_full) - not_involve, 2):
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
    for lbls in leaf_node_labels:
        print(len(lbls))
    print(cfg_full)
    if log:
        logging.info(cfg_full)
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


def calculate_all_indices(data, train_or_val):
    indices = [[] for _ in range(365)]
    print("Calculating All Indices...")
    for i in range(len(data)):
        _, label = data[i]
        indices[label].append(i)
        if i % 50000 == 0:
            print('{}/{} ({:.0f}%)'.format(i, len(data), 100. * i / len(data)))
    print("Calculation Done")
    if train_or_val == 0:
        np.save('all_train_indices.npy', indices)
    elif train_or_val == 1:
        np.save('all_val_indices.npy', indices)
    return indices


def load_class_indices(data, no_classes, train_or_val, classes=None):
    indices = []
    if train_or_val == 0:
        if os.path.isfile('all_train_indices.npy'):
            all_indices = np.load('all_train_indices.npy')
        else:
            all_indices = calculate_all_indices(data, train_or_val)
        if classes is None:
            for i in range(no_classes):
                indices += all_indices[i]
        else:
            for i in range(len(classes)):
                indices += all_indices[classes[i]]
    elif train_or_val == 1:
        if os.path.isfile('all_val_indices.npy'):
            all_indices = np.load('all_val_indices.npy')
            if classes is None:
                indices = all_indices[:no_classes].reshape(-1)
            else:
                indices = all_indices[classes].reshape(-1)
        else:
            all_indices = calculate_all_indices(data, train_or_val)
            if classes is None:
                for i in range(no_classes):
                    indices += all_indices[i]
            else:
                for i in range(len(classes)):
                    indices += all_indices[classes[i]]
    return indices


def calculate_no_of_params(models):
    length = 0
    if isinstance(models, list):
        for model in models:
            if not model is None:
                length += sum(p.numel() for p in model.parameters())
    else:
        length = sum(p.numel() for p in models.parameters())
    return length


def calculate_no_of_params_in_detail(models, more_detail=False):
    length = 0
    if isinstance(models, list):
        convs, bns, linears = 0, 0, 0
        for model in models:
            if not model is None:
                conv, bn, linear = 0, 0, 0
                if more_detail:
                    if isinstance(model, MobileTreeRootNet):
                        print("Root:")
                    elif isinstance(model, MobileTreeBranchNet):
                        print("Branch:")
                    elif isinstance(model, MobileTreeLeafNet):
                        print("Leaf:")
                for name, p in model.named_parameters():
                    np = p.numel()
                    if "conv" in name:
                        conv += np
                        convs += np
                    elif "bn" in name:
                        bn += np
                        bns += np
                    else:
                        linear += np
                        linears += np
                length += sum(p.numel() for p in model.parameters())
                if more_detail:
                    print('Conv:\t' + str(conv))
                    print('Bn: \t' + str(bn))
                    print('Fcl:\t' + str(linear))
                    print('Total:\t' + str(conv + bn + linear))
                    print()
    else:
        convs, bns, linears = 0, 0, 0
        for name, p in models.named_parameters():
            if "conv" in name:
                convs += p.numel()
            elif "bn" in name:
                bns += p.numel()
            else:
                linears += p.numel()
        length = sum(p.numel() for p in models.parameters())
    return convs, bns, linears, length


def calculate_params_all_preferences_tree(models, all_prefs, leaf_node_labels, log):
    params_of_model = [0] * len(models)

    leaf_node_index = []
    leaf_node_paths = []
    for i in range(len(models)):
        if not models[i] is None:
            params_of_model[i] = sum(p.numel() for p in models[i].parameters())
            models[i].eval()
            if isinstance(models[i], MobileTreeLeafNet):
                leaf_node_index.append(i)
    for i in leaf_node_index:
        path = []
        while i > 0:
            path = [i] + path
            i = (i + 1) // 2 - 1
        leaf_node_paths.append(path)

    no_params = [0] * len(all_prefs)
    for p, single_pref in enumerate(all_prefs):
        models_to_include = [False] * len(models)
        for i in range(len(leaf_node_labels)):
            if any(elem in leaf_node_labels[i] for elem in single_pref):
                models_to_include[0] = True
                for j in leaf_node_paths[i]:
                    models_to_include[j] = True
        params = 0
        for i in range(len(models_to_include)):
            if models_to_include[i]:
                params += params_of_model[i]
        no_params[p] = params

    avg_no_params = sum(no_params) / float(len(no_params))

    print(no_params)
    print("\nAvg # of Params: " + str(avg_no_params))

    if log:
        logging.info("Avg # of Params: " + str(avg_no_params))

    return avg_no_params


def calculate_params_all_preferences_parallel(models, all_prefs, leaf_node_labels, log):
    params_of_model = [0] * len(models)

    for i in range(len(models)):
        if not models[i] is None:
            params_of_model[i] = sum(p.numel() for p in models[i].parameters())

    no_params = [0] * len(all_prefs)
    for p, single_pref in enumerate(all_prefs):
        models_to_include = [False] * len(models)
        for i in range(len(leaf_node_labels)):
            if any(elem in leaf_node_labels[i] for elem in single_pref):
                models_to_include[i] = True
        params = 0
        for i in range(len(models_to_include)):
            if models_to_include[i]:
                params += params_of_model[i]
        no_params[p] = params

    avg_no_params = sum(no_params) / float(len(no_params))

    print(no_params)
    print("\nAvg # of Params: " + str(avg_no_params))

    if log:
        logging.info("Avg # of Params: " + str(avg_no_params))

    return avg_no_params


def pref_table_to_all_prefs(preference_table):
    all_prefs = []
    for i in range(len(preference_table)):
        all_prefs.append([])
        for j in range(len(preference_table[i])):
            if preference_table[i][j] == 1:
                all_prefs[i].append(j)
    return all_prefs


def saveModel(model, acc, epoch, path='./saved/mobilenet.pth'):
    if vis != 0:
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'vis': True,
            'vis-win-loss': vis.win_loss,
            'vis-win-acc': vis.win_acc
        }
    else:
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'vis': False
        }
    torch.save(state, path)


def getArgs():
    batch_size = 64
    test_batch_size = 64
    epochs = 10
    lr = 0.001
    depth = 1
    resize = 256

    parser = argparse.ArgumentParser(description="Parameters for training Tree-Net")
    parser.add_argument('-cf', '--cifar10', action='store_true', help='uses Cifar-10 dataset')
    parser.add_argument('-t', '--test', action='store_true', help='enables test mode')
    parser.add_argument('-jt', '--just-train', action='store_true', help='train only without testing')
    parser.add_argument('-tp', '--test-prefs', action='store_true', help='do not test for all preferences while training')
    parser.add_argument('-r', '--resume', action='store_true', help='whether to resume training or not (default: 0)')
    parser.add_argument('-f', '--fine-tune', action='store_true', help='fine-tune optimization')
    parser.add_argument('-s', '--same', action='store_true', help='use same user preference table to generate the tree')
    parser.add_argument('-l', '--log', action='store_true', help='log the events')
    parser.add_argument('-ll', '--limit-log', action='store_true', help='do not log initial logs')
    parser.add_argument('-ft', '--fast-train', action='store_true', help='does not calculate unnecessary things')
    parser.add_argument('-nw', '--no-weights', action='store_true', help='train without class weights')
    parser.add_argument('-rs', '--resize', type=int, default=resize, metavar='rsz', help='resize images in the dataset (default: 256)')
    parser.add_argument('-p', '--prefs', nargs='+', type=int)
    parser.add_argument('-m', '--model', type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7], help='choose models')
    parser.add_argument('-m0', '--mobile-net', action='store_true', help='train mobile-net instead of tree-net')
    parser.add_argument('-mp', '--parallel-mobile-nets', action='store_true', help='train parallel-mobile-net instead of tree-net')
    parser.add_argument('-mtn', '--mobile-tree-net', action='store_true', help='train mobile-tree-net instead of tree-net')
    parser.add_argument('-mt', '--mobile-tree-net-old', action='store_true', help='train mobile-tree-net-old instead of tree-net')
    parser.add_argument('-d', '--depth', type=int, default=depth, metavar='lvl', help='depth of the tree (default: 1)')
    parser.add_argument('-b', '--batch-size', type=int, default=batch_size, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('-tb', '--test-batch-size', type=int, default=test_batch_size, metavar='N', help='input batch size for testing (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=epochs, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('-lr', '--lr', type=float, default=lr, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('-cw', '--num-workers', type=int, default=0, metavar='N', help='number of workers for cuda')
    parser.add_argument('-w', '--weight-mult', type=float, default=1.0, metavar='N', help='class weight multiplier')
    parser.add_argument('-pp', '--pref-prob', type=float, default=0.3, metavar='N', help='class weight multiplier')
    parser.add_argument('-nc', '--num-classes', type=int, default=365, metavar='N', help='train for only first n classes (default: 365)')
    parser.add_argument('-sm', '--samples', type=int, default=1000, metavar='N', help='number of preferences in the preference table')
    parser.add_argument('-cp', '--calc-params', action='store_true', help='enable calculating parameters of the model')
    parser.add_argument('-cs', '--calc-storage', action='store_true', help='enable calculating storage of the models for all preferences')
    parser.add_argument('-li', '--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status (default: 100)')
    parser.add_argument('-uc', '--use-classes', action='store_true', help='use specific classes')
    parser.add_argument('-sr', '--root-step', type=int, default=1, help='number of root steps')
    parser.add_argument('-sc', '--conv-step', type=int, default=3, help='number of conv steps')
    parser.add_argument('-ni', '--not-involve', type=int, default=1, help='number of last layers not involved in reducing the number of channels')
    parser.add_argument('-df', '--div-factor', type=int, default=-1, help='dividing factor in networks')
    parser.add_argument('-ls', '--lr-scheduler', action='store_true', help='enables lr scheduler')
    parser.add_argument('-lrg', '--lr-gamma', type=float, default=0.1, help='gamma of lr scheduler')
    parser.add_argument('-lrs', '--lr-step', type=int, default=30, help='steps of lr scheduler')
    parser.add_argument('-adm', '--adam', action='store_true', help='choose adam optimizer instead of sgd')
    parser.add_argument('-vis', '--visdom', action='store_true', help='use visdom to plot graphs')
    parser.add_argument('-val', '--val-mode', action='store_true', help='saves the best accuracy model in each test')
    parser.add_argument('-da', '--data-aug', type=int, default=1, choices=[1, 2], help='choose the data augmentation')
    parser.add_argument('-pre', '--pre-trained', action='store_true', help='saves the best accuracy model in each test')
    parser.add_argument('-mgpu', '--multi-gpu', action='store_true', help='enable using multiple gpu')
    args = parser.parse_args()
    return args


def main():
    global best_acc
    args = getArgs()

    if args.visdom:
        global vis
        vis = utils.Visualizations()

    test = args.test
    resume = args.resume
    same = args.same
    fine_tune = args.fine_tune
    prefs = args.prefs
    test_prefs = args.test_prefs

    last_epoch = 0

    no_classes = args.num_classes
    if args.cifar10:
        no_classes = 10
    samples = args.samples
    if no_classes != 365:
        if samples == 1000:
            samples = no_classes * 5

    if args.log:
        start_time = time.time()
        logfile = time.strftime("Logs/%y%m%d.log", time.localtime(start_time))
        logging.basicConfig(filename=logfile, level=logging.INFO)
        logging.info("---START---")
        logging.info(time.asctime(time.localtime(start_time)))
        if args.cifar10:
            logging.info("CIFAR-10")
        else:
            logging.info("Places-365")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cuda_args = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

    if args.cifar10:
        mean = (0.4914, 0.4822, 0.4465)
        sd = (0.2023, 0.1994, 0.2010)
        resize = 32
    else:
        mean = (0.485, 0.456, 0.406)
        sd = (0.229, 0.224, 0.225)
        resize = args.resize

    if args.data_aug == 1:
        train_data_transform = transforms.Compose([
            transforms.RandomCrop(resize, padding=(int(resize/8))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, sd)
        ])
        val_data_transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean, sd)
        ])
    elif args.data_aug == 2:
        train_data_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(20),
            transforms.RandomAffine(15, (0.2, 0.2)),
            transforms.Resize((resize ,resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean, sd)
        ])
        val_data_transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean, sd)
        ])
    else:
        train_data_transform = transforms.Compose([
            transforms.RandomResizedCrop(resize, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, sd)
        ])
        val_data_transform = transforms.Compose([
            transforms.Resize(int(resize/0.875)),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, sd)
        ])

    if args.cifar10:
        cifar_training_data = datasets.CIFAR10("../data/CIFAR10", train=True, transform=train_data_transform, download=True)
        cifar_testing_data = datasets.CIFAR10("../data/CIFAR10", train=False, transform=val_data_transform)
        train_loader = torch.utils.data.DataLoader(cifar_training_data, batch_size=args.batch_size, shuffle=True, **cuda_args)
        val_loader = torch.utils.data.DataLoader(cifar_testing_data, batch_size=args.test_batch_size, shuffle=True, **cuda_args)
    else:
        traindir = os.path.join('../places365/places365_standard', 'train')
        valdir = os.path.join('../places365/places365_standard', 'val')
        places_training_data = datasets.ImageFolder(traindir, transform=train_data_transform)
        places_validation_data = datasets.ImageFolder(valdir, transform=val_data_transform)
        if no_classes == 365:
            train_loader = torch.utils.data.DataLoader(places_training_data, batch_size=args.batch_size, shuffle=True, **cuda_args)
            val_loader = torch.utils.data.DataLoader(places_validation_data, batch_size=args.test_batch_size, shuffle=False, **cuda_args)
        else:
            if args.use_classes:
                train_indices = load_class_indices(places_training_data, no_classes, train_or_val=0, classes=class_labels)
                val_indices = load_class_indices(places_validation_data, no_classes, train_or_val=1, classes=class_labels)
            else:
                train_indices = load_class_indices(places_training_data, no_classes, train_or_val=0)
                val_indices = load_class_indices(places_validation_data, no_classes, train_or_val=1)
            train_loader = torch.utils.data.DataLoader(places_training_data, batch_size=args.batch_size,
                                                       sampler=SubsetRandomSampler(train_indices), **cuda_args)
            val_loader = torch.utils.data.DataLoader(places_validation_data, batch_size=args.test_batch_size,
                                                       sampler=SubsetRandomSampler(val_indices), **cuda_args)

    fcl_factor = resize // 32

    if args.mobile_net or args.model != 0:
        if args.mobile_net or args.model == 1:
            model = MobileNet(num_classes=no_classes, fcl=(fcl_factor*fcl_factor*1024))
            save_name = "mobilenet"
        elif args.model == 2:
            model = Models.vgg16_bn(pretrained=args.pre_trained, num_classes=no_classes)
            save_name = "vggnet"
        elif args.model == 3:
            model = Models.alexnet(pretrained=args.pre_trained, num_classes=no_classes)
            save_name = "alexnet"
        elif args.model == 4:
            model = Models.resnet18(pretrained=args.pre_trained, num_classes=no_classes)
            save_name = "resnet18"
        elif args.model == 5:
            model = Models.densenet161(pretrained=args.pre_trained, num_classes=no_classes)
            save_name = "densenet161"
        elif args.model == 6:
            model = Models.inception_v3(pretrained=args.pre_trained, num_classes=no_classes)
            save_name = "inceptionv3net"
        else:
            model = Models.squeezenet1_0(pretrained=args.pre_trained, num_classes=no_classes)
            save_name = "squeezenet1.0"
        if use_cuda and torch.cuda.device_count() > 1 and args.multi_gpu: model = torch.nn.DataParallel(model)
        model.to(device)
        if args.log:
            logging.info(save_name)
            if args.fine_tune:
                logging.info("fine-tune")
            elif resume:
                logging.info("resume")
            elif test:
                logging.info("test")
            logging.info("Learning Rate: " + str(args.lr))
            if args.adam:
                logging.info("Optimizer: Adam")
            else:
                logging.info("Optimizer: SGD")
            logging.info("Epochs: " + str(args.epochs))
            logging.info("Batch Size: " + str(args.batch_size))
            logging.info("Size of Images: " + str(resize))
            logging.info("Number of Classes: " + str(no_classes))
        if args.calc_params:
            no_params = calculate_no_of_params(model)
            print("Number of Parameters: " + str(no_params))
            if args.log:
                logging.info("Number of Parameters: " + str(no_params))
        if not test:
            if resume:
                state = torch.load('./saved/' + save_name + '.pth')
                model.load_state_dict(state['model'])
                best_acc = state['acc']
                last_epoch = state['epoch']
                if state['vis']:
                    vis.win_acc = state['vis-win-acc']
                    vis.win_loss = state['vis-win-loss']
            args.epochs += last_epoch
            for epoch in range(last_epoch + 1, args.epochs + 1):
                if args.just_train:
                    train_net(model, train_loader, device, epoch, args, save_name)
                else:
                    train_net(model, train_loader, device, epoch, args, save_name)
                    test_net(model, val_loader, device, args, epoch, save_name)
                    if test_prefs:
                        preference_table = np.load('preference_table.npy')
                        all_prefs = pref_table_to_all_prefs(preference_table.T)
                        test_net_all_preferences(model, val_loader, device, args, all_prefs)
        else:
            state = torch.load('./saved/' + save_name + '.pth')
            model.load_state_dict(state['model'])
            best_acc = state['acc']
            test_net(model, val_loader, device, args, save_name=save_name)
            if test_prefs:
                preference_table = np.load('preference_table.npy')
                all_prefs = pref_table_to_all_prefs(preference_table.T)
                test_net_all_preferences(model, val_loader, device, args, all_prefs)
    elif args.mobile_tree_net:
        print("Mobile Tree Net")
        load = resume or test or same or fine_tune
        root_node = utils.generate(no_classes, samples, load, prob=args.pref_prob)
        dividing_factor = 2 if args.div_factor == -1 else args.div_factor
        models, leaf_node_labels = generate_model_list(root_node, args.depth, device, fcl_factor,
                                                       root_step=args.root_step, step=args.conv_step, dividing_factor=dividing_factor,
                                                       not_involve=args.not_involve, log=(args.log and not args.limit_log))
        if args.log and not args.limit_log:
            logging.info("Mobile Tree Net")
            if fine_tune:
                logging.info("fine-tune")
            elif resume:
                logging.info("resume")
            elif test:
                logging.info("test")
            elif same:
                logging.info("same")
            logging.info("Leaf Node Labels:" + str(leaf_node_labels))
            logging.info("Learning Rate: " + str(args.lr))
            if args.adam:
                logging.info("Optimizer: Adam")
            else:
                logging.info("Optimizer: SGD")
            logging.info("Depth: " + str(args.depth))
            logging.info("Epochs: " + str(args.epochs))
            logging.info("Batch Size: " + str(args.batch_size))
            logging.info("Size of Images: " + str(resize))
            logging.info("Number of Classes: " + str(no_classes))
            if prefs:
                logging.info("Pref Classes: " + str(prefs))
            if args.weight_mult != 1.0:
                logging.info("Weight factor: " + str(args.weight_mult))
        elif args.log:
            logging.info("Learning Rate: " + str(args.lr))
            logging.info("Epochs: " + str(args.epochs))
            logging.info("Batch Size: " + str(args.batch_size))
            if prefs:
                logging.info("Pref Classes: " + str(prefs))
            if args.weight_mult != 1.0:
                logging.info("Weight factor: " + str(args.weight_mult))
        if args.calc_params:
            if prefs:
                pref_models = []
                in_ln_index = []
                j = 0
                for i, model in enumerate(models):
                    if isinstance(model, MobileTreeLeafNet):
                        if any(elem in leaf_node_labels[j] for elem in prefs):
                            in_ln_index.append(i)
                        j += 1
                for i, model in enumerate(models):
                    if not model is None:
                        if i == 0:
                            pref_models.append(model)
                        for k in in_ln_index:
                            while k > 0:
                                if k == i:
                                    pref_models.append(model)
                                k = (k + 1) // 2 - 1
                no_params = calculate_no_of_params(pref_models)
                no_params_all = calculate_no_of_params(models)
                print("Number of Parameters: " + str(no_params) + " / " + str(no_params_all))
                if args.log:
                    logging.info("Number of Parameters: " + str(no_params) + " / " + str(no_params_all))
            else:
                no_params = calculate_no_of_params(models)
                print("Number of Parameters: " + str(no_params))
                if args.log:
                    logging.info("Number of Parameters: " + str(no_params))
        if not test:
            if fine_tune:
                for i in range(len(models)):
                    if not models[i] is None:
                        state = torch.load('./saved/treemodel' + str(i) + '.pth')
                        models[i].load_state_dict(state['model'])
                        best_acc = state['acc']
                        last_epoch = state['epoch']
                        if state['vis']:
                            vis.win_acc = state['vis-win-acc']
                            vis.win_loss = state['vis-win-loss']
                args.epochs += last_epoch
                for epoch in range(last_epoch + 1, args.epochs + 1):
                    train_hierarchical(models, leaf_node_labels, train_loader, device, epoch, args, use_cuda, args.depth, args.depth)
                    if prefs is None:
                        test_tree(models, leaf_node_labels, val_loader, device, args, epoch)
                    else:
                        test_tree(models, leaf_node_labels, val_loader, device, args, epoch)
                        test_tree_personal(models, leaf_node_labels, val_loader, device, args, prefs)
            else:
                if resume:
                    for i in range(len(models)):
                        if not models[i] is None:
                            state = torch.load('./saved/treemodel' + str(i) + '.pth')
                            models[i].load_state_dict(state['model'])
                            best_acc = state['acc']
                            last_epoch = state['epoch']
                            if state['vis']:
                                vis.win_acc = state['vis-win-acc']
                                vis.win_loss = state['vis-win-loss']
                args.epochs += last_epoch
                for epoch in range(last_epoch + 1, args.epochs + 1):
                    if args.just_train:
                        train_tree(models, leaf_node_labels, train_loader, device, epoch, args, use_cuda)
                    else:
                        train_tree(models, leaf_node_labels, train_loader, device, epoch, args, use_cuda)
                        if prefs is None:
                            test_tree(models, leaf_node_labels, val_loader, device, args, epoch)
                            if test_prefs:
                                preference_table = np.load('preference_table.npy')
                                all_prefs = pref_table_to_all_prefs(preference_table.T)     #change binary table to list of labels
                                test_tree_all_preferences(models, leaf_node_labels, val_loader, device, args, all_prefs)
                                if args.calc_storage:
                                    calculate_params_all_preferences_tree(models, all_prefs, leaf_node_labels, args.log)
                        else:
                            test_tree(models, leaf_node_labels, val_loader, device, args, epoch)
                            test_tree_personal(models, leaf_node_labels, val_loader, device, args, prefs)
        else:
            for i in range(len(models)):
                if not models[i] is None:
                    state = torch.load('./saved/treemodel' + str(i) + '.pth')
                    models[i].load_state_dict(state['model'])
                    best_acc = state['acc']
            if prefs is None:
                test_tree(models, leaf_node_labels, val_loader, device, args)
                if test_prefs:
                    preference_table = np.load('preference_table.npy')
                    all_prefs = pref_table_to_all_prefs(preference_table.T)  # change binary table to list of labels
                    test_tree_all_preferences(models, leaf_node_labels, val_loader, device, args, all_prefs)
                    if args.calc_storage:
                        calculate_params_all_preferences_tree(models, all_prefs, leaf_node_labels, args.log)
            else:
                test_tree(models, leaf_node_labels, val_loader, device, args)
                test_tree_personal(models, leaf_node_labels, val_loader, device, args, prefs)
    elif args.mobile_tree_net_old:
        print("Mobile Tree Net Old")
        load = resume or test or same or fine_tune
        root_node = utils.generate(no_classes, samples, load, prob=args.pref_prob)
        dividing_factor = 2 if args.div_factor == -1 else args.div_factor
        models, leaf_node_labels = generate_model_list(root_node, args.depth, device, fcl_factor,
                                                       root_step=args.root_step, step=args.conv_step, dividing_factor=dividing_factor,
                                                       not_involve=args.not_involve, log=(args.log and not args.limit_log))
        if args.log and not args.limit_log:
            logging.info("Mobile Tree Net Old")
            if fine_tune:
                logging.info("fine-tune")
            elif resume:
                logging.info("resume")
            elif test:
                logging.info("test")
            elif same:
                logging.info("same")
            logging.info("Leaf Node Labels:" + str(leaf_node_labels))
            logging.info("Learning Rate: " + str(args.lr))
            if args.adam:
                logging.info("Optimizer: Adam")
            else:
                logging.info("Optimizer: SGD")
            logging.info("Depth: " + str(args.depth))
            logging.info("Epochs: " + str(args.epochs))
            logging.info("Batch Size: " + str(args.batch_size))
            logging.info("Size of Images: " + str(resize))
            logging.info("Number of Classes: " + str(no_classes))
            if prefs:
                logging.info("Pref Classes: " + str(prefs))
            if args.weight_mult != 1.0:
                logging.info("Weight factor: " + str(args.weight_mult))
        elif args.log:
            logging.info("Learning Rate: " + str(args.lr))
            logging.info("Epochs: " + str(args.epochs))
            logging.info("Batch Size: " + str(args.batch_size))
            if prefs:
                logging.info("Pref Classes: " + str(prefs))
            if args.weight_mult != 1.0:
                logging.info("Weight factor: " + str(args.weight_mult))
        if args.calc_params:
            if prefs:
                pref_models = []
                in_ln_index = []
                j = 0
                for i, model in enumerate(models):
                    if isinstance(model, MobileTreeLeafNet):
                        if any(elem in leaf_node_labels[j] for elem in prefs):
                            in_ln_index.append(i)
                        j += 1
                for i, model in enumerate(models):
                    if not model is None:
                        if i == 0:
                            pref_models.append(model)
                        for k in in_ln_index:
                            while k > 0:
                                if k == i:
                                    pref_models.append(model)
                                k = (k + 1) // 2 - 1
                no_params = calculate_no_of_params(pref_models)
                no_params_all = calculate_no_of_params(models)
                print("Number of Parameters: " + str(no_params) + " / " + str(no_params_all))
                if args.log:
                    logging.info("Number of Parameters: " + str(no_params) + " / " + str(no_params_all))
            else:
                no_params = calculate_no_of_params(models)
                print("Number of Parameters: " + str(no_params))
                if args.log:
                    logging.info("Number of Parameters: " + str(no_params))
        if not test:
            if fine_tune:
                for i in range(len(models)):
                    if not models[i] is None:
                        state = torch.load('./saved/treemodel' + str(i) + '.pth')
                        models[i].load_state_dict(state['model'])
                        best_acc = state['acc']
                        last_epoch = state['epoch']
                        if state['vis']:
                            vis.win_acc = state['vis-win-acc']
                            vis.win_loss = state['vis-win-loss']
                args.epochs += last_epoch
                for epoch in range(last_epoch + 1, args.epochs + 1):
                    train_hierarchical(models, leaf_node_labels, train_loader, device, epoch, args, use_cuda, args.depth, args.depth)
                    if prefs is None:
                        test_tree(models, leaf_node_labels, val_loader, device, args, epoch)
                    else:
                        test_tree(models, leaf_node_labels, val_loader, device, args, epoch)
                        test_tree_personal(models, leaf_node_labels, val_loader, device, args, prefs)
            else:
                if resume:
                    for i in range(len(models)):
                        if not models[i] is None:
                            state = torch.load('./saved/treemodel' + str(i) + '.pth')
                            models[i].load_state_dict(state['model'])
                            best_acc = state['acc']
                            last_epoch = state['epoch']
                            if state['vis']:
                                vis.win_acc = state['vis-win-acc']
                                vis.win_loss = state['vis-win-loss']
                args.epochs += last_epoch
                for epoch in range(last_epoch + 1, args.epochs + 1):
                    if args.just_train:
                        train_tree_old(models, leaf_node_labels, train_loader, device, epoch, args, use_cuda)
                    else:
                        train_tree_old(models, leaf_node_labels, train_loader, device, epoch, args, use_cuda)
                        if prefs is None:
                            test_tree(models, leaf_node_labels, val_loader, device, args, epoch)
                            if test_prefs:
                                preference_table = np.load('preference_table.npy')
                                all_prefs = pref_table_to_all_prefs(preference_table.T)     #change binary table to list of labels
                                test_tree_all_preferences(models, leaf_node_labels, val_loader, device, args, all_prefs)
                                if args.calc_storage:
                                    calculate_params_all_preferences_tree(models, all_prefs, leaf_node_labels, args.log)
                        else:
                            test_tree(models, leaf_node_labels, val_loader, device, args, epoch)
                            test_tree_personal(models, leaf_node_labels, val_loader, device, args, prefs)
        else:
            for i in range(len(models)):
                if not models[i] is None:
                    state = torch.load('./saved/treemodel' + str(i) + '.pth')
                    models[i].load_state_dict(state['model'])
                    best_acc = state['acc']
            if prefs is None:
                test_tree(models, leaf_node_labels, val_loader, device, args)
                if test_prefs:
                    preference_table = np.load('preference_table.npy')
                    all_prefs = pref_table_to_all_prefs(preference_table.T)  # change binary table to list of labels
                    test_tree_all_preferences(models, leaf_node_labels, val_loader, device, args, all_prefs)
                    if args.calc_storage:
                        calculate_params_all_preferences_tree(models, all_prefs, leaf_node_labels, args.log)
            else:
                test_tree(models, leaf_node_labels, val_loader, device, args)
                test_tree_personal(models, leaf_node_labels, val_loader, device, args, prefs)
    elif args.parallel_mobile_nets:
        cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]
        load = resume or test or same or fine_tune
        root_node = utils.generate(no_classes, samples, load, prob=args.pref_prob)
        leaf_node_labels = find_leaf_node_labels(root_node, args.depth)
        dividing_factor = len(leaf_node_labels) if args.div_factor == -1 else args.div_factor
        for i in range(0, len(cfg), 2):
            cfg[i] = cfg[i] // dividing_factor if isinstance(cfg[i], int) else (
            cfg[i][0] // dividing_factor, cfg[i][1])
        models = []
        for i in leaf_node_labels:
            branches = len(i) + 1
            models.append(MobileNet(num_classes=branches, channels=cfg, fcl=((fcl_factor*fcl_factor*1024) // dividing_factor)).to(device))
        if args.log:
            logging.info("Parallel Mobile Nets")
            if resume:
                logging.info("resume")
            elif test:
                logging.info("test")
            elif same:
                logging.info("same")
            logging.info("Leaf Node Labels:" + str(leaf_node_labels))
            logging.info("Learning Rate: " + str(args.lr))
            if args.adam:
                logging.info("Optimizer: Adam")
            else:
                logging.info("Optimizer: SGD")
            logging.info("Depth: " + str(args.depth))
            logging.info("Epochs: " + str(args.epochs))
            logging.info("Batch Size: " + str(args.batch_size))
            logging.info("Size of Images: " + str(args.resize))
            logging.info("Number of Classes: " + str(no_classes))
            if args.weight_mult != 1.0:
                logging.info("Weight factor: " + str(args.weight_mult))
        if args.calc_params:
            if prefs:
                pref_models = []
                for i, model in enumerate(models):
                    if any(elem in leaf_node_labels[i] for elem in prefs):
                        pref_models.append(model)
                no_params = calculate_no_of_params(pref_models)
                no_params_all = calculate_no_of_params(models)
                print("Number of Parameters: " + str(no_params) + " / " + str(no_params_all))
                if args.log:
                    logging.info("Number of Parameters: " + str(no_params) + " / " + str(no_params_all))
            else:
                no_params = calculate_no_of_params(models)
                print("Number of Parameters: " + str(no_params))
                if args.log:
                    logging.info("Number of Parameters: " + str(no_params))
        if not test:
            if resume:
                for i in range(len(models)):
                    state = torch.load('./saved/parallel_mobilenet' + str(i) + '.pth')
                    models[i].load_state_dict(state['model'])
                    best_acc = state['acc']
                    last_epoch = state['epoch']
                    if state['vis']:
                        vis.win_acc = state['vis-win-acc']
                        vis.win_loss = state['vis-win-loss']
            args.epochs += last_epoch
            for epoch in range(last_epoch + 1, args.epochs + 1):
                if args.just_train:
                    train_parallel_mobilenet(models, leaf_node_labels, train_loader, device, epoch, args, use_cuda)
                else:
                    train_parallel_mobilenet(models, leaf_node_labels, train_loader, device, epoch, args, use_cuda)
                    if prefs is None:
                        test_parallel_mobilenet(models, leaf_node_labels, val_loader, device, args, epoch)
                        if test_prefs:
                            preference_table = np.load('preference_table.npy')
                            all_prefs = pref_table_to_all_prefs(preference_table.T)  # change binary table to list of labels
                            test_parallel_all_preferences(models, leaf_node_labels, val_loader, device, args, all_prefs)
                            if args.calc_storage:
                                calculate_params_all_preferences_parallel(models, all_prefs, leaf_node_labels, args.log)
                    else:
                        test_parallel_mobilenet(models, leaf_node_labels, val_loader, device, args, epoch)
                        test_parallel_personal(models, leaf_node_labels, val_loader, device, args, prefs)
        else:
            for i in range(len(models)):
                state = torch.load('./saved/parallel_mobilenet' + str(i) + '.pth')
                models[i].load_state_dict(state['model'])
                best_acc = state['acc']
            if prefs is None:
                test_parallel_mobilenet(models, leaf_node_labels, val_loader, device, args)
                if test_prefs:
                    preference_table = np.load('preference_table.npy')
                    all_prefs = pref_table_to_all_prefs(preference_table.T)  # change binary table to list of labels
                    test_parallel_all_preferences(models, leaf_node_labels, val_loader, device, args, all_prefs)
                    if args.calc_storage:
                        calculate_params_all_preferences_parallel(models, all_prefs, leaf_node_labels, args.log)
            else:
                test_parallel_mobilenet(models, leaf_node_labels, val_loader, device, args)
                test_parallel_personal(models, leaf_node_labels, val_loader, device, args, prefs)

    if args.log:
        end_time = time.time()
        logging.info(time.asctime(time.localtime(end_time)))
        logging.info("--- %s seconds ---" % (end_time - start_time))
        logging.info("---END---\n")


if __name__ == '__main__':
    main()
