import argparse
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from utils import IndexSampler

import logging
import time
from random import randint

from models.mobilenet import MobileNet
from models.vgg import VGG16
from models.mobile_tree_net import MobileTreeRootNet, MobileTreeLeafNet, MobileTreeBranchNet
from models.vgg_tree_net import VGG_Branch, VGG_Leaf
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
            if isinstance(models[i], MobileTreeLeafNet) or isinstance(models[i], VGG_Leaf):
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
                    res = models[i](results[prev])
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
            if isinstance(models[i], MobileTreeLeafNet) or isinstance(models[i], VGG_Leaf):
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
        pred_probs = []
        concat_results = 0
        losses_to_print = []
        for i in range(len(leaf_node_paths)):
            optims[i].zero_grad()

            lbls = labels.clone()
            for l in range(len(lbls)):
                if lbls[l].item() in leaf_node_labels[i]:
                    lbls[l] = leaf_node_labels[i].index(lbls[l])
                else:
                    lbls[l] = len(leaf_node_labels[i])

            result = models[0](data)
            for j in range(len(leaf_node_paths[i]) - 1):
                k = leaf_node_paths[i][j]
                result = models[k](result)
            k = leaf_node_index[i]
            result = models[k](result)
            if not args.fast_train:
                pred.append(result.max(1, keepdim=True)[1])
                pred_probs.append(result.max(1, keepdim=True)[0])
                output_without_else = torch.stack([j[:-1] for j in result])
                if isinstance(concat_results, int) and concat_results == 0:
                    concat_results = output_without_else
                else:
                    concat_results = torch.cat((concat_results, output_without_else), dim=1)

            l = losses[i](result, lbls)
            l.backward(retain_graph=True)
            optims[i].step()
            losses_to_print.append(l.item())

        avg_loss += (sum(losses_to_print) / float(len(losses_to_print)))

        if not args.fast_train:
            full_pred = concat_results.max(1, keepdim=True)[1]
            leaf_labels = [i for sub in leaf_node_labels for i in sub]
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
                                if pred_probs[j][i] >= pred_probs[ln_index[0]][i]:
                                    definite = False
                    if definite:
                        definite_correct += 1
                elif pred[ln_index[0]][i] == len(leaf_node_labels[ln_index[0]]):
                    all_else = True
                    for j in range(len(leaf_node_index)):
                        if j != ln_index[0]:
                            if pred[j][i] != len(leaf_node_labels[j]):
                                all_else = False
                                break
                    if all_else:
                        if lbl == leaf_labels[full_pred[i].item()]:
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
            if isinstance(models[i], MobileTreeLeafNet) or isinstance(models[i], VGG_Leaf):
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
            result = models[k](layer)

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
            if isinstance(models[i], MobileTreeLeafNet) or isinstance(models[i], VGG_Leaf):
                leaf_node_index.append(i)

    for i in leaf_node_index:
        path = []
        while i > 0:
            path = [i] + path
            i = (i + 1) // 2 - 1
        leaf_node_paths.append(path)

    definite_correct = 0
    indefinite_correct = 0
    new_corrects = 0
    wrong = 0
    avg_loss = 0
    for data, label in test_loader:
        data, labels = data.to(device), label.to(device)
        if args.use_classes:
            labels = map_labels(labels).to(device)

        pred = []
        pred_probs = []
        losses_to_print = []
        leaf_node_results = []
        sum_of_losses = 0
        results = [None] * len(models)
        results[0] = models[0](data)
        concat_results = 0
        for i in range(1, len(models)):
            if not models[i] is None:
                prev = (i + 1) // 2 - 1
                if i in leaf_node_index:
                    res = models[i](results[prev])
                    results[i] = res
                    pred.append(res.max(1, keepdim=True)[1])
                    pred_probs.append(res.max(1, keepdim=True)[0])
                    output_without_else = torch.stack([i[:-1] for i in res])
                    if isinstance(concat_results, int) and concat_results == 0:
                        concat_results = output_without_else
                    else:
                        concat_results = torch.cat((concat_results, output_without_else), dim=1)

                    if not args.fast_train:
                        leaf_node_results.append(res)
                else:
                    results[i] = models[i](results[prev])

        full_pred = concat_results.max(1, keepdim=True)[1]
        leaf_labels = [i for sub in leaf_node_labels for i in sub]

        for i in range(len(labels)):
            if labels[i].item() == leaf_labels[full_pred[i].item()]:
                new_corrects += 1

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
                            if pred_probs[j][i] >= pred_probs[ln_index[0]][i]:
                                definite = False
                if definite:
                    definite_correct += 1
                else:
                    indefinite_correct += 1
            elif pred[ln_index[0]][i] == len(leaf_node_labels[ln_index[0]]):
                all_else = True
                for j in range(len(leaf_node_index)):
                    if j != ln_index[0]:
                        if pred[j][i] != len(leaf_node_labels[j]):
                            all_else = False
                            wrong += 1
                            break
                if all_else:
                    if lbl == leaf_labels[full_pred[i].item()]:
                        definite_correct += 1
                    else:
                        wrong += 1
            else:
                wrong += 1

    acc = 100. * definite_correct / len(test_loader.sampler)
    new_acc = 100. * new_corrects / len(test_loader.sampler)
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
        logging.info('Test set: New Accuracy: {}/{} ({:.2f}%)'.format(new_corrects, len(test_loader.sampler), new_acc))
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\tDefinite Corrects: {}/{} ({:.2f}%)\tAvg loss: {:.4f}\n'.format(
        (definite_correct + indefinite_correct), len(test_loader.sampler),
        100. * (definite_correct + indefinite_correct) / len(test_loader.sampler),
        definite_correct, len(test_loader.sampler), acc, avg_loss
    ))
    print('Test set: New Accuracy: {}/{} ({:.2f}%)'.format(new_corrects, len(test_loader.sampler), new_acc))


def add_deeper_leaves(check_list, index, leaf_node_index, models):
    first_child = ((index + 1) * 2) - 1
    second_child = first_child + 1
    if len(models) > first_child and models[first_child] is not None:
        if first_child in leaf_node_index:
            check_list.append(leaf_node_index.index(first_child))
        else:
            add_deeper_leaves(check_list, first_child, leaf_node_index, models)
    if len(models) > second_child and models[second_child] is not None:
        if second_child in leaf_node_index:
            check_list.append(leaf_node_index.index(second_child))
        else:
            add_deeper_leaves(check_list, second_child, leaf_node_index, models)


def get_order_of_checking_extra_leaf_indices(initial_model_indices, leaf_node_index, models):
    # list(reversed(initial_model_indices)) = [4,0] -> [13,4]
    # leaf_node_index = [4,7,11,12,13,14,17,18]
    # models = [0,1,2,3,4,5,6,7,8,None,None,11,12,13,14,None,None,17,18,?????]

    check_list = []
    init_labels = [leaf_node_index[i] for i in reversed(initial_model_indices)]
    for i in init_labels:
        index = i
        while index:
            index = index + 1 if index % 2 == 1 else index - 1      # sibling index
            if index in leaf_node_index:
                if leaf_node_index.index(index) not in initial_model_indices:
                    if leaf_node_index.index(index) not in check_list and index in leaf_node_index:
                        check_list.append(leaf_node_index.index(index))
                    else:
                        add_deeper_leaves(check_list, index, leaf_node_index, models)
            else:
                add_deeper_leaves(check_list, index, leaf_node_index, models)
            index = (index + 1) // 2 - 1        # parent index

        # Add remaining
        for j in range(len(leaf_node_index)):
            if j not in initial_model_indices and j not in check_list:
                check_list.append(j)
    return check_list


def test_tree_scenario(models, leaf_node_labels, test_users, class_indices, data_transform, device, args, cuda_args):
    leaf_node_index = []
    leaf_node_paths = []

    for i in range(len(models)):
        if not models[i] is None:
            models[i].eval()
            if isinstance(models[i], MobileTreeLeafNet) or isinstance(models[i], VGG_Leaf):
                leaf_node_index.append(i)

    for i in leaf_node_index:
        path = []
        while i > 0:
            path = [i] + path
            i = (i + 1) // 2 - 1
        leaf_node_paths.append(path)

    avg_acc = 0
    avg_new_acc = 0
    avg_mem = 0
    storage_check = True
    for each_user in test_users:
        # Getting the data for each user
        indices = []
        for label in each_user:
            i = class_indices[label][randint(0, len(class_indices[label])-1)]
            indices.append(i)
        if args.cifar10:
            data = datasets.CIFAR10("../data/CIFAR10", train=False, transform=data_transform)
        elif args.cifar100:
            data = datasets.CIFAR100("../data/CIFAR100", train=False, transform=data_transform)
        else:
            valdir = os.path.join('../places365/places365_standard', 'val')
            data = datasets.ImageFolder(valdir, transform=data_transform)
        data_loader = torch.utils.data.DataLoader(data, batch_size=args.test_batch_size,
                                                  sampler=IndexSampler(indices), **cuda_args)

        if args.cifar10:
            initialize_models_count = 10
        elif args.cifar100:
            initialize_models_count = 50
        else:
            initialize_models_count = 100
        initialized = False
        initialized_labels = []
        initial_model_indices = []
        initial_models_enough_count, all_models_used_count = 0, initialize_models_count
        extra_used_models = []
        extra_used_indices = []
        definite_correct = 0
        new_corrects = 0
        for data, label in data_loader:
            data, labels = data.to(device), label.to(device)

            if initialized:
                initialize_models_count = 0 - len(labels)
            if not initialized:
                initialize_models_count -= len(labels)
                if initialize_models_count > 0:
                    for lbl in labels:
                        if lbl.item() not in initialized_labels:
                            initialized_labels.append(lbl.item())
                else:
                    for lbl in labels[:initialize_models_count + len(labels)]:
                        if lbl.item() not in initialized_labels:
                            initialized_labels.append(lbl.item())
                    initialized = True
            if initialized:
                for lbl in initialized_labels:
                    for i in range(len(leaf_node_labels)):
                        if lbl in leaf_node_labels[i]:
                            if i not in initial_model_indices:
                                initial_model_indices.append(i)     # leaf node label index, not the model index!
                            break

            pred = []
            pred_probs = []
            leaf_node_results = 0
            results = [None] * len(models)
            results[0] = models[0](data)
            for i in range(1, len(models)):
                if not models[i] is None:
                    prev = (i + 1) // 2 - 1
                    if i in leaf_node_index:
                        res = models[i](results[prev])
                        results[i] = res
                        pred.append(res.max(1, keepdim=True)[1])
                        pred_probs.append(res.max(1, keepdim=True)[0])
                        output_without_else = torch.stack([i[:-1] for i in res])
                        if isinstance(leaf_node_results, int) and leaf_node_results == 0:
                            leaf_node_results = output_without_else
                        else:
                            leaf_node_results = torch.cat((leaf_node_results, output_without_else), dim=1)
                    else:
                        results[i] = models[i](results[prev])
            full_pred = leaf_node_results.max(1, keepdim=True)[1]
            leaf_labels = [i for sub in leaf_node_labels for i in sub]

            for i in range(len(labels)):
                if labels[i].item() == leaf_labels[full_pred[i].item()]:
                    new_corrects += 1

            if not initialized:
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
                                    if pred_probs[j][i] >= pred_probs[ln_index[0]][i]:
                                        definite = False
                        if definite:
                            definite_correct += 1
                    elif pred[ln_index[0]][i] == len(leaf_node_labels[ln_index[0]]):
                        all_else = True
                        for j in range(len(leaf_node_index)):
                            if j != ln_index[0]:
                                if pred[j][i] != len(leaf_node_labels[j]):
                                    all_else = False
                                    break
                        if all_else:
                            if lbl == leaf_labels[full_pred[i].item()]:
                                definite_correct += 1
            else:
                # remaining initial whole model use
                for i in range(initialize_models_count + len(labels)):
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
                                    if pred_probs[j][i] >= pred_probs[ln_index[0]][i]:
                                        definite = False
                        if definite:
                            definite_correct += 1
                    elif pred[ln_index[0]][i] == len(leaf_node_labels[ln_index[0]]):
                        all_else = True
                        for j in range(len(leaf_node_index)):
                            if j != ln_index[0]:
                                if pred[j][i] != len(leaf_node_labels[j]):
                                    all_else = False
                                    break
                        if all_else:
                            if lbl == leaf_labels[full_pred[i].item()]:
                                definite_correct += 1

                # first try with initial models, then use all models if necessary
                for i in range(initialize_models_count + len(labels), len(labels)):
                    lbl = labels[i].item()
                    ln_index = -1
                    for j in range(len(leaf_node_labels)):
                        if lbl in leaf_node_labels[j]:
                            k = leaf_node_labels[j].index(lbl)
                            ln_index = (j, k)
                            break

                    check_list = get_order_of_checking_extra_leaf_indices(initial_model_indices, leaf_node_index, models)

                    correct = True
                    found = False
                    last_prob = 0
                    for j in initial_model_indices:
                        if pred[j][i] != len(leaf_node_labels[j]):
                            if not found:
                                last_prob = pred_probs[j][i]
                                found = True
                                initial_models_enough_count += 1
                            else:
                                if pred_probs[j][i] < last_prob:
                                    continue
                            if j == ln_index[0]:
                                if pred[j][i] == ln_index[1]:
                                    correct = True
                                else:
                                    correct = False
                            else:
                                correct = False
                    if not found:
                        # Check other models in decreasing order
                        if args.scenario_use_full_model:
                            all_models_used_count += 1
                            if lbl != leaf_labels[full_pred[i].item()]:
                                found = True
                                correct = False
                        else:
                            if len(initial_model_indices) == len(leaf_node_labels):
                                initial_models_enough_count += 1
                            else:
                                for j in check_list:
                                    extra_used_indices.append(j)
                                    if pred[j][i] != len(leaf_node_labels[j]):
                                        found = True
                                        if j == ln_index[0]:
                                            if pred[j][i] != ln_index[1]:
                                                correct = False
                                        else:
                                            correct = False
                    if not found:
                        if lbl != leaf_labels[full_pred[i].item()]:
                            correct = False

                    if len(extra_used_indices):
                        extra_used_models.append(extra_used_indices)
                        extra_used_indices = []
                    if correct:
                        definite_correct += 1

        if initial_models_enough_count + all_models_used_count + len(extra_used_models) != len(data_loader.sampler):
            storage_check = False

        no_of_params = calculate_no_of_params_for_tree(models)

        initial_indices = [0]
        for i in initial_model_indices:
            path = leaf_node_paths[i]
            for j in path:
                if j not in initial_indices:
                    initial_indices.append(j)
        initial_storage = 0
        for i in initial_indices:
            initial_storage += no_of_params[i]

        extra_storage = 0
        for i in range(len(extra_used_models)):
            extra_storage += initial_storage
            extra_indices = []
            for j in range(len(extra_used_models[i])):
                path = leaf_node_paths[extra_used_models[i][j]]
                for k in path:
                    if k not in extra_indices and k not in initial_indices:
                        extra_indices.append(k)
            for j in extra_indices:
                extra_storage += no_of_params[j]

        storage = ((all_models_used_count * calculate_no_of_params_sum_each(models)) + (initial_storage * initial_models_enough_count) + extra_storage) / len(each_user)

        acc = 100. * definite_correct / len(data_loader.sampler)
        new_acc = 100. * new_corrects / len(data_loader.sampler)
        avg_acc += acc
        avg_new_acc += new_acc
        avg_mem += storage

    model_size = calculate_no_of_params(models)
    if storage_check:
        print("Storage Calculation Check Success!")
    else:
        print("Storage Calculation Check Failed!")
    avg_acc /= len(test_users)
    avg_new_acc /= len(test_users)
    avg_mem //= len(test_users)
    if args.log:
        logging.info('Test Scenario Average Accuracy: ({:.2f}%)'.format(avg_acc))
        logging.info('Test Scenario Average New Accuracy (Full Model): ({:.2f}%)'.format(avg_new_acc))
        logging.info('Test Scenario Average Memory: {}/{}'.format(avg_mem, model_size))
    print('Test Scenario Average Accuracy: ({:.2f}%)'.format(avg_acc))
    print('Test Scenario Average New Accuracy (Full Model): ({:.2f}%)'.format(avg_new_acc))
    print('Test Scenario Average Memory: {}/{}'.format(avg_mem, model_size))


def test_tree_personal(models, leaf_node_labels, test_loader, device, args, preferences):
    leaf_node_index = []
    leaf_node_paths = []  # NOT INCLUDING models[0]

    for i in range(len(models)):
        if not models[i] is None:
            models[i].eval()
            if isinstance(models[i], MobileTreeLeafNet) or isinstance(models[i], VGG_Leaf):
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
                    result = models[k](layer)
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
            if isinstance(models[i], MobileTreeLeafNet) or isinstance(models[i], VGG_Leaf):
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
                    result = models[k](layer)
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


def test_net_scenario(model, test_users, class_indices, data_transform, device, args, cuda_args):
    model.eval()
    loss = torch.nn.CrossEntropyLoss()
    loss.to(device)

    avg_acc = 0
    for each_user in test_users:
        # Getting the data for each user
        indices = []
        for label in each_user:
            i = class_indices[label][randint(0, len(class_indices[label])-1)]
            indices.append(i)

        if args.cifar10:
            data = datasets.CIFAR10("../data/CIFAR10", train=False, transform=data_transform)
        elif args.cifar100:
            data = datasets.CIFAR100("../data/CIFAR100", train=False, transform=data_transform)
        else:
            valdir = os.path.join('../places365/places365_standard', 'val')
            data = datasets.ImageFolder(valdir, transform=data_transform)
        data_loader = torch.utils.data.DataLoader(data, batch_size=args.test_batch_size,
                                                 sampler=IndexSampler(indices), **cuda_args)
        # Testing the data for each user
        test_loss = 0
        correct = 0
        for data, label in data_loader:
            data, labels = data.to(device), label.to(device)
            output = model(data)
            test_loss += loss(output, labels).item()
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()
        test_loss /= len(data_loader)
        acc = 100. * correct / len(data_loader.sampler)
        avg_acc += acc
    avg_acc /= len(test_users)
    num_params = calculate_no_of_params(model)
    if args.log:
        logging.info('Test Scenario Average Accuracy: ({:.2f}%), Storage: {}'.format(avg_acc, num_params))
    print('Test Scenario Average Accuracy: ({:.2f}%), Storage: {}'.format(avg_acc, num_params))


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


def train_parallel_net(models, leaf_node_labels, train_loader, device, epoch, args, use_cuda):
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
        pred_probs = []
        leaf_node_results = 0
        for i in range(len(models)):
            optims[i].zero_grad()
            output = models[i](data)
            if not args.fast_train:
                pred.append(output.max(1, keepdim=True)[1])
                pred_probs.append(output.max(1, keepdim=True)[1])
                output_without_else = torch.stack([i[:-1] for i in output])
                if isinstance(leaf_node_results, int) and leaf_node_results == 0:
                    leaf_node_results = output_without_else
                else:
                    leaf_node_results = torch.cat((leaf_node_results, output_without_else), dim=1)

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
            full_pred = leaf_node_results.max(1, keepdim=True)[1]
            leaf_labels = [i for sub in leaf_node_labels for i in sub]  # Unite leaf labels in a list
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
                                if pred_probs[j][i] >= pred_probs[ln_index[0]][i]:
                                    definite = False
                    if definite:
                        definite_correct += 1
                elif pred[ln_index[0]][i] == len(leaf_node_labels[ln_index[0]]):
                    all_else = True
                    for j in range(len(leaf_node_labels)):
                        if j != ln_index[0]:
                            if pred[j][i] != len(leaf_node_labels[j]):
                                all_else = False
                                break
                    if all_else:
                        if lbl == leaf_labels[full_pred[i].item()]:
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
            saveModel(models[i], acc, epoch, './saved/parallel_net' + str(i) + '.pth')
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


def test_parallel_net(models, leaf_node_labels, test_loader, device, args, epoch=0):
    global best_acc
    for model in models:
        model.eval()

    definite_correct = 0
    indefinite_correct = 0
    new_corrects = 0
    wrong = 0
    avg_loss = 0
    for data, label in test_loader:
        data, labels = data.to(device), label.to(device)
        if args.use_classes:
            labels = map_labels(labels).to(device)

        pred = []
        pred_probs = []
        losses_to_print = []
        sum_of_losses = 0
        leaf_node_results = []
        concat_results = 0
        for i in range(len(models)):
            output = models[i](data)
            pred.append(output.max(1, keepdim=True)[1])
            pred_probs.append(output.max(1, keepdim=True)[0])
            output_without_else = torch.stack([i[:-1] for i in output])
            if isinstance(concat_results, int) and concat_results == 0:
                concat_results = output_without_else
            else:
                concat_results = torch.cat((concat_results, output_without_else), dim=1)

            if not args.fast_train:
                leaf_node_results.append(output)

        full_pred = concat_results.max(1, keepdim=True)[1]
        leaf_labels = [i for sub in leaf_node_labels for i in sub]

        for i in range(len(labels)):
            if labels[i] == leaf_labels[full_pred[i].item()]:
                new_corrects += 1

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
                            if pred_probs[j][i] >= pred_probs[ln_index[0]][i]:
                                definite = False
                if definite:
                    definite_correct += 1
                else:
                    indefinite_correct += 1
            elif pred[ln_index[0]][i] == len(leaf_node_labels[ln_index[0]]):
                all_else = True
                for j in range(len(leaf_node_labels)):
                    if j != ln_index[0]:
                        if pred[j][i] != len(leaf_node_labels[j]):
                            all_else = False
                            wrong += 1
                            break
                if all_else:
                    if lbl == leaf_labels[full_pred[i].item()]:
                        definite_correct += 1
                    else:
                        wrong += 1
            else:
                wrong += 1

    acc = 100. * definite_correct / len(test_loader.sampler)
    new_acc = 100. * new_corrects / len(test_loader.sampler)
    if args.val_mode and acc > best_acc:
        best_acc = acc
        for i in range(len(models)):
            saveModel(models[i], acc, epoch, './saved/parallel_net' + str(i) + '.pth')

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
        logging.info('Test set: New Accuracy: {}/{} ({:.2f}%)'.format(new_corrects, len(test_loader.sampler), new_acc))
    print('Test set: Accuracy: {}/{} ({:.2f}%)\tDefinite Corrects: {}/{} ({:.2f}%)\tAvg loss: {:.4f}\n'.format(
        (definite_correct + indefinite_correct), len(test_loader.sampler),
        100. * (definite_correct + indefinite_correct) / len(test_loader.sampler),
        definite_correct, len(test_loader.sampler), acc, avg_loss
    ))
    print('Test set: New Accuracy: {}/{} ({:.2f}%)'.format(new_corrects, len(test_loader.sampler), new_acc))


def test_parallel_scenario(models, leaf_node_labels, test_users, class_indices, data_transform, device, args, cuda_args):
    for model in models:
        model.eval()

    avg_acc = 0
    avg_new_acc = 0
    avg_mem = 0
    storage_check = True
    for each_user in test_users:
        # Getting the data for each user
        indices = []
        for label in each_user:
            i = class_indices[label][randint(0, len(class_indices[label]) - 1)]
            indices.append(i)
        if args.cifar10:
            data = datasets.CIFAR10("../data/CIFAR10", train=False, transform=data_transform)
        elif args.cifar100:
            data = datasets.CIFAR100("../data/CIFAR100", train=False, transform=data_transform)
        else:
            valdir = os.path.join('../places365/places365_standard', 'val')
            data = datasets.ImageFolder(valdir, transform=data_transform)
        data_loader = torch.utils.data.DataLoader(data, batch_size=args.test_batch_size,
                                                  sampler=IndexSampler(indices), **cuda_args)

        if args.cifar10:
            initialize_models_count = 10
        elif args.cifar100:
            initialize_models_count = 50
        else:
            initialize_models_count = 100
        initialized = False
        initialized_labels = []
        initial_model_indices = []
        initial_models_enough_count, all_models_used_count = 0, initialize_models_count
        extra_used_models = []
        extra_used_indices = []
        definite_correct = 0
        new_corrects = 0
        for data, label in data_loader:
            data, labels = data.to(device), label.to(device)

            if initialized:
                initialize_models_count = 0 - len(labels)
            if not initialized:
                initialize_models_count -= len(labels)
                if initialize_models_count > 0:
                    for lbl in labels:
                        if lbl.item() not in initialized_labels:
                            initialized_labels.append(lbl.item())
                else:
                    for lbl in labels[:initialize_models_count + len(labels)]:
                        if lbl.item() not in initialized_labels:
                            initialized_labels.append(lbl.item())
                    initialized = True
            if initialized:
                for lbl in initialized_labels:
                    for i in range(len(leaf_node_labels)):
                        if lbl in leaf_node_labels[i]:
                            if i not in initial_model_indices:
                                initial_model_indices.append(i)
                            break

            pred = []
            pred_probs = []
            leaf_node_results = 0
            for i in range(len(models)):
                output = models[i](data)
                pred.append(output.max(1, keepdim=True)[1])
                pred_probs.append(output.max(1, keepdim=True)[0])
                output_without_else = torch.stack([i[:-1] for i in output])
                if isinstance(leaf_node_results, int) and leaf_node_results == 0:
                    leaf_node_results = output_without_else
                else:
                    leaf_node_results = torch.cat((leaf_node_results, output_without_else), dim=1)
            full_pred = leaf_node_results.max(1, keepdim=True)[1]
            leaf_labels = [i for sub in leaf_node_labels for i in sub]     # Unite leaf labels in a list

            for i in range(len(labels)):
                if labels[i].item() == leaf_labels[full_pred[i].item()]:
                    new_corrects += 1

            if not initialized:
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
                                    if pred_probs[j][i] >= pred_probs[ln_index[0]][i]:
                                        definite = False
                        if definite:
                            definite_correct += 1
                    elif pred[ln_index[0]][i] == len(leaf_node_labels[ln_index[0]]):
                        all_else = True
                        for j in range(len(leaf_node_labels)):
                            if j != ln_index[0]:
                                if pred[j][i] != len(leaf_node_labels[j]):
                                    all_else = False
                                    break
                        if all_else:
                            if lbl == leaf_labels[full_pred[i].item()]:
                                definite_correct += 1
            else:
                # remaining initial whole model use
                for i in range(initialize_models_count + len(labels)):
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
                                    if pred_probs[j][i] >= pred_probs[ln_index[0]][i]:
                                        definite = False
                        if definite:
                            definite_correct += 1
                    elif pred[ln_index[0]][i] == len(leaf_node_labels[ln_index[0]]):
                        all_else = True
                        for j in range(len(leaf_node_labels)):
                            if j != ln_index[0]:
                                if pred[j][i] != len(leaf_node_labels[j]):
                                    all_else = False
                                    break
                        if all_else:
                            if lbl == leaf_labels[full_pred[i].item()]:
                                definite_correct += 1

                # first try with initial models, then use all models if necessary
                for i in range(initialize_models_count + len(labels), len(labels)):
                    lbl = labels[i].item()
                    ln_index = -1
                    for j in range(len(leaf_node_labels)):
                        if lbl in leaf_node_labels[j]:
                            k = leaf_node_labels[j].index(lbl)
                            ln_index = (j, k)
                            break

                    correct = True
                    found = False
                    last_prob = 0
                    for j in initial_model_indices:
                        if pred[j][i] != len(leaf_node_labels[j]):
                            if not found:
                                last_prob = pred_probs[j][i]
                                found = True
                                initial_models_enough_count += 1
                            else:
                                if pred_probs[j][i] < last_prob:
                                    continue
                            if j == ln_index[0]:
                                if pred[j][i] == ln_index[1]:
                                    correct = True
                                else:
                                    correct = False
                            else:
                                correct = False
                    if not found:
                        # Check other models in decreasing order
                        if args.scenario_use_full_model:
                            all_models_used_count += 1
                            if lbl != leaf_labels[full_pred[i].item()]:
                                found = True
                                correct = False
                        else:
                            if len(initial_model_indices) == len(leaf_node_labels):
                                initial_models_enough_count += 1
                            else:
                                for j in reversed(range(len(leaf_node_labels))):
                                    if j not in initial_model_indices:
                                        extra_used_indices.append(j)
                                        if pred[j][i] != len(leaf_node_labels[j]):
                                            found = True
                                            if j == ln_index[0]:
                                                if pred[j][i] != ln_index[1]:
                                                    correct = False
                                            else:
                                                correct = False
                    if not found:
                        if lbl != leaf_labels[full_pred[i].item()]:
                            correct = False

                    if len(extra_used_indices):
                        extra_used_models.append(extra_used_indices)
                        extra_used_indices = []
                    if correct:
                        definite_correct += 1
        if initial_models_enough_count + all_models_used_count + len(extra_used_models) != len(data_loader.sampler):
            storage_check = False
            print(initial_models_enough_count + all_models_used_count + len(extra_used_models))
            print(data_loader.sampler)

        no_of_params = calculate_no_of_params_for_each(models)
        in_size, rem_size = 0, sum(no_of_params)
        for i in initial_model_indices:
            in_size += no_of_params[i]

        extra_storage = 0
        for i in range(len(extra_used_models)):
            extra_storage += in_size
            for j in extra_used_models[i]:
                extra_storage += no_of_params[j]

        storage = ((in_size * initial_models_enough_count) + (rem_size * all_models_used_count) + extra_storage) / len(each_user)

        avg_mem += storage

        acc = 100. * definite_correct / len(data_loader.sampler)
        new_acc = 100. * new_corrects / len(data_loader.sampler)
        avg_acc += acc
        avg_new_acc += new_acc

    if storage_check:
        print("Storage Calculation Check Success!")
    else:
        print("Storage Calculation Check Failed!")
    model_size = calculate_no_of_params(models)
    avg_acc /= len(test_users)
    avg_new_acc /= len(test_users)
    avg_mem //= len(test_users)
    if args.log:
        logging.info('Test Scenario Average Accuracy: ({:.2f}%)'.format(avg_acc))
        logging.info('Test Scenario Average New Accuracy (Full Model): ({:.2f}%)'.format(avg_new_acc))
        logging.info('Test Scenario Average Memory: {}/{}'.format(avg_mem, model_size))
    print('Test Scenario Average Accuracy: ({:.2f}%)'.format(avg_acc))
    print('Test Scenario Average New Accuracy (Full Model): ({:.2f}%)'.format(avg_new_acc))
    print('Test Scenario Average Memory: {}/{}'.format(avg_mem, model_size))


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


def generate_model_list(root_node, level, device, fcl_factor, model=1, root_step=3, step=3, dividing_factor=2, dividing_step= 2, not_involve=1, log=False):
    ### Model: 1 -- MobileNet V1
    ### Model: 2 -- VGG_16_BN
    leaf_node_labels = []
    if model == 1:
        cfg_full = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]
        models = [MobileTreeRootNet(cfg_full[:root_step]).to(device)]
    elif model == 2:
        cfg_full = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        models = [VGG_Branch(cfg_full[:root_step]).to(device)]
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
        if model == 1:
            in_planes = cfg_full[conv_step - 1] if isinstance(cfg_full[conv_step - 1], int) else cfg_full[conv_step - 1][0]
        elif model == 2:
            in_planes = cfg_full[conv_step - 1] if isinstance(cfg_full[conv_step - 1], int) else cfg_full[conv_step - 2]

        if prev_lvl < lvl:
            prev_lvl = lvl
            for i in range(conv_step, len(cfg_full) - not_involve, dividing_step):
                if isinstance(cfg_full[i], int):
                    cfg_full[i] = int(cfg_full[i] // dividing_factor)
                elif isinstance(cfg_full[i],tuple):
                    cfg_full[i] = (int(cfg_full[i][0] // dividing_factor), cfg_full[i][1])

        # LEFT BRANCH
        left = nodes[index][0].left
        if not isinstance(left, int):
            if left.count > 3 and lvl < level:
                if model == 1:
                    models.append(MobileTreeBranchNet(input=cfg_full[conv_step:conv_step + step], in_planes=in_planes).to(device))
                elif model == 2:
                    models.append(VGG_Branch(cfg_full[conv_step:conv_step + step], in_channels=in_planes).to(device))
                nodes.append((left, lvl))
                remaining += 1
            else:
                if model == 1:
                    models.append(MobileTreeLeafNet(branch=(left.count + 1), input=cfg_full[conv_step:], in_planes=in_planes, fcl=cfg_full[-1]*fcl_factor*fcl_factor).to(device))
                elif model == 2:
                    models.append(VGG_Leaf(cfg_full[conv_step:], in_channels=in_planes, out_channel=cfg_full[-2]*fcl_factor*fcl_factor, num_classes=(left.count + 1)).to(device))
                nodes.append(None)
                leaf_node_labels.append(left.value)
        else:
            if model == 1:
                models.append(MobileTreeLeafNet(branch=2, input=cfg_full[conv_step:], in_planes=in_planes, fcl=cfg_full[-1]*fcl_factor*fcl_factor).to(device))
            elif model == 2:
                models.append(VGG_Leaf(cfg_full[conv_step:], in_channels=in_planes,
                                       out_channel=cfg_full[-2] * fcl_factor * fcl_factor,
                                       num_classes=2).to(device))
            nodes.append(None)
            leaf_node_labels.append((left,))

        # RIGHT BRANCH
        right = nodes[index][0].right
        if not isinstance(right, int):
            if right.count > 3 and lvl < level:
                if model == 1:
                    models.append(MobileTreeBranchNet(input=cfg_full[conv_step:conv_step + step], in_planes=in_planes).to(device))
                elif model == 2:
                    models.append(VGG_Branch(cfg_full[conv_step:conv_step + step], in_channels=in_planes).to(device))
                nodes.append((right, lvl))
                remaining += 1
            else:
                if model == 1:
                    models.append(MobileTreeLeafNet(branch=(right.count + 1), input=cfg_full[conv_step:], in_planes=in_planes, fcl=cfg_full[-1]*fcl_factor*fcl_factor).to(device))
                elif model == 2:
                    models.append(VGG_Leaf(cfg_full[conv_step:], in_channels=in_planes,
                                           out_channel=cfg_full[-2] * fcl_factor * fcl_factor,
                                           num_classes=(right.count + 1)).to(device))
                nodes.append(None)
                leaf_node_labels.append(right.value)
        else:
            if model == 1:
                models.append(MobileTreeLeafNet(branch=2, input=cfg_full[conv_step:], in_planes=in_planes, fcl=cfg_full[-1]*fcl_factor*fcl_factor).to(device))
            elif model == 2:
                models.append(VGG_Leaf(cfg_full[conv_step:], in_channels=in_planes,
                                       out_channel=cfg_full[-2] * fcl_factor * fcl_factor,
                                       num_classes=2).to(device))

            nodes.append(None)
            leaf_node_labels.append((right,))

        index += 1
        remaining -= 1
    for lbls in leaf_node_labels:
        print(len(lbls), end=' ')
    print()
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


def calculate_all_indices_scenario(data, no_classes):
    indices = [[] for _ in range(no_classes)]
    print("Calculating All Indices...")
    for i in range(len(data)):
        _, label = data[i]
        indices[label].append(i)
        if i % 50000 == 0:
            print('{}/{} ({:.0f}%)'.format(i, len(data), 100. * i / len(data)))
    print("Calculation Done")
    return indices


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


def calculate_no_of_params_sum_each(models):
    length = 0
    for model in models:
        if not model is None:
            length += sum(p.numel() for p in model.parameters())
    return length


def calculate_no_of_params_for_tree(models):
    no_of_params = []
    for model in models:
        if not model is None:
            no_of_params.append(sum(p.numel() for p in model.parameters()))
        else:
            no_of_params.append(None)
    return no_of_params


def calculate_no_of_params_for_each(models):
    no_of_params = []
    for model in models:
        if not model is None:
            no_of_params.append(sum(p.numel() for p in model.parameters()))
    return no_of_params


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
            if isinstance(models[i], MobileTreeLeafNet) or isinstance(models[i], VGG_Leaf):
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
    resize = 224

    parser = argparse.ArgumentParser(description="Parameters for training Tree-Net")
    parser.add_argument('-cf', '--cifar10', action='store_true', help='uses Cifar-10 dataset')
    parser.add_argument('-cf2', '--cifar100', action='store_true', help='uses Cifar-100 dataset')
    parser.add_argument('-t', '--test', action='store_true', help='enables test mode')
    parser.add_argument('-jt', '--just-train', action='store_true', help='train only without testing')
    parser.add_argument('-tp', '--test-prefs', action='store_true', help='do not test for all preferences while training')
    parser.add_argument('-ts', '--test-scenario', action='store_true', help='scenario test')
    parser.add_argument('-r', '--resume', action='store_true', help='whether to resume training or not (default: 0)')
    parser.add_argument('-f', '--fine-tune', action='store_true', help='fine-tune optimization')
    parser.add_argument('-s', '--same', action='store_true', help='use same user preference table to generate the tree')
    parser.add_argument('-l', '--log', action='store_true', help='log the events')
    parser.add_argument('-ll', '--limit-log', action='store_true', help='do not log initial logs')
    parser.add_argument('-ft', '--fast-train', action='store_true', help='does not calculate unnecessary things')
    parser.add_argument('-nw', '--no-weights', action='store_false', help='train without class weights')
    parser.add_argument('-rs', '--resize', type=int, default=resize, help='resize images in the dataset (default: 256)')
    parser.add_argument('-p', '--prefs', nargs='+', type=int)
    parser.add_argument('-m', '--model', type=int, default=0, choices=[0, 1, 2], help='choose models')
    parser.add_argument('-mp', '--parallel-mobile-nets', action='store_true', help='train parallel-mobile-net')
    parser.add_argument('-vp', '--parallel-vgg', action='store_true', help='train parallel-vgg-net')
    parser.add_argument('-mtn', '--mobile-tree-net', action='store_true', help='train mobile-tree-net')
    parser.add_argument('-mt', '--mobile-tree-net-old', action='store_true', help='train mobile-tree-net-old')
    parser.add_argument('-vt', '--vgg-tree', action='store_true', help='train vgg-tree')
    parser.add_argument('-d', '--depth', type=int, default=depth, help='depth of the tree (default: 1)')
    parser.add_argument('-b', '--batch-size', type=int, default=batch_size, help='input batch size for training (default: 64)')
    parser.add_argument('-tb', '--test-batch-size', type=int, default=test_batch_size, help='input batch size for testing (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=epochs, help='number of epochs to train (default: 10)')
    parser.add_argument('-lr', '--lr', type=float, default=lr, help='learning rate (default: 0.001)')
    parser.add_argument('-cw', '--num-workers', type=int, default=0, help='number of workers for cuda')
    parser.add_argument('-w', '--weight-mult', type=float, default=1.0, help='class weight multiplier')
    parser.add_argument('-pp', '--pref-prob', type=float, default=0.3, help='class weight multiplier')
    parser.add_argument('-nc', '--num-classes', type=int, default=365, help='train for only first n classes (default: 365)')
    parser.add_argument('-sm', '--samples', type=int, default=1000, help='number of preferences in the preference table')
    parser.add_argument('-nut', '--num-user-types', type=int, default=10, help='number of scenario user types')
    parser.add_argument('-nsu', '--num-scenario-users', type=int, default=100, help='number of scenario test users')
    parser.add_argument('-lsu', '--load-scenario-users', action='store_true', help='number of scenario test users')
    parser.add_argument('-nsd', '--scenario-data-length', type=int, default=1000, help='number of test images per scenario test users')
    parser.add_argument('-sufm', '--scenario-use-full-model', action='store_true', help='use full model in a miss situation')
    parser.add_argument('-put', '--print-user-types', action='store_true', help='print user types')
    parser.add_argument('-ghfd', '--gen-from-dist', action='store_true', help='generate hierarchy from distribution instead of generated users')
    parser.add_argument('-ghnu', '--gen-from-new-users', action='store_false', help='do not load already generated users for generating hierarchy')
    parser.add_argument('-ghom', '--old-gen-method', action='store_true', help='generate hierarchy with the old method')
    parser.add_argument('-cp', '--calc-params', action='store_true', help='enable calculating parameters of the model')
    parser.add_argument('-cs', '--calc-storage', action='store_true', help='enable calculating storage of the models for all preferences')
    parser.add_argument('-li', '--log-interval', type=int, default=100, help='how many batches to wait before logging training status (default: 100)')
    parser.add_argument('-uc', '--use-classes', action='store_true', help='use specific classes')
    parser.add_argument('-sr', '--root-step', type=int, default=3, help='number of root steps')
    parser.add_argument('-sc', '--conv-step', type=int, default=3, help='number of conv steps')
    parser.add_argument('-ni', '--not-involve', type=int, default=1, help='number of last layers not involved in reducing the number of channels')
    parser.add_argument('-df', '--div-factor', type=float, default=1.4142, help='dividing factor in networks')
    parser.add_argument('-ds', '--div-step', type=int, default=1, help='dividing factor in networks')
    parser.add_argument('-ls', '--lr-scheduler', action='store_true', help='enables lr scheduler')
    parser.add_argument('-lrg', '--lr-gamma', type=float, default=0.1, help='gamma of lr scheduler')
    parser.add_argument('-lrs', '--lr-step', type=int, default=30, help='steps of lr scheduler')
    parser.add_argument('-adm', '--adam', action='store_true', help='choose adam optimizer instead of sgd')
    parser.add_argument('-vis', '--visdom', action='store_true', help='use visdom to plot graphs')
    parser.add_argument('-val', '--val-mode', action='store_true', help='saves the best accuracy model in each test')
    parser.add_argument('-da', '--data-aug', type=int, default=1, choices=[1, 2, 3], help='choose the data augmentation')
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
    load = resume or test or same or fine_tune
    prefs = args.prefs
    test_prefs = args.test_prefs

    last_epoch = 0

    no_classes = args.num_classes
    if args.cifar10 and no_classes > 10:
        no_classes = 10
    samples = args.samples
    if no_classes != 365 and samples == 1000:
        samples = no_classes * 5

    if args.log:
        start_time = time.time()
        logfile = time.strftime("Logs/%y%m%d.log", time.localtime(start_time))
        logging.basicConfig(filename=logfile, level=logging.INFO)
        logging.info("---START---")
        logging.info(time.asctime(time.localtime(start_time)))
        if args.cifar10:
            logging.info("CIFAR-10")
        elif args.cifar100:
            logging.info("CIFAR-100")
        else:
            logging.info("Places-365")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cuda_args = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

    if args.cifar10 or args.cifar100:
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
        val_loader = torch.utils.data.DataLoader(cifar_testing_data, batch_size=args.test_batch_size, shuffle=False, **cuda_args)
    elif args.cifar100:
        cifar_100_training_data = datasets.CIFAR100("../data/CIFAR100", train=True, transform=train_data_transform, download=True)
        cifar_100_testing_data = datasets.CIFAR100("../data/CIFAR100", train=False, transform=val_data_transform)
        train_loader = torch.utils.data.DataLoader(cifar_100_training_data, batch_size=args.batch_size, shuffle=True, **cuda_args)
        val_loader = torch.utils.data.DataLoader(cifar_100_testing_data, batch_size=args.test_batch_size, shuffle=False, **cuda_args)
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

    if args.old_gen_method:
        print("Old Generating Method!")
        root_node = utils.generate_hierarchy_from_type_distribution(no_classes, n_type=args.num_user_types, load=load, print_types=args.print_user_types)
    else:
        if args.cifar10:
            root_node = utils.generate_hierarchy_with_cooccurrence(no_classes, n_type=args.num_user_types, load=load, with_distribution=args.gen_from_dist, load_gen_users=args.gen_from_new_users, print_types=args.print_user_types)
        else:
            root_node = utils.generate_hierarchy_with_cooccurrence(no_classes, n_type=args.num_user_types, load=load, with_distribution=args.gen_from_dist, load_gen_users=args.gen_from_new_users, print_types=args.print_user_types, start=2, end_not_inc=21)
    if args.test_scenario:
        test_scenario_users = utils.generate_users(args.num_scenario_users, args.scenario_data_length,
                                                   load=args.load_scenario_users)
        if args.cifar10:
            if os.path.isfile('all_cifar_val_indices.npy'):
                class_indices = np.load('all_cifar_val_indices.npy')
            else:
                class_indices = calculate_all_indices_scenario(cifar_testing_data, 10)
                np.save('all_cifar_val_indices.npy', class_indices)
        elif args.cifar100:
            if os.path.isfile('all_cifar_100_val_indices.npy'):
                class_indices = np.load('all_cifar_100_val_indices.npy')
            else:
                class_indices = calculate_all_indices_scenario(cifar_100_testing_data, 100)
                np.save('all_cifar_100_val_indices.npy', class_indices)
        else:
            if os.path.isfile('all_places365_val_indices.npy'):
                class_indices = np.load('all_places365_val_indices.npy')
            else:
                class_indices = calculate_all_indices_scenario(places_validation_data, 365)
                np.save('all_places365_val_indices.npy', class_indices)

    fcl_factor = resize // 32

    if args.model != 0:
        if args.model == 1:
            model = MobileNet(num_classes=no_classes, fcl=(fcl_factor*fcl_factor*1024))
            save_name = "mobilenet"
        else:
            model = VGG16(num_classes=no_classes, fcl=(fcl_factor*fcl_factor*512))
            save_name = "vggnet"
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
            if args.test_scenario:
                test_net_scenario(model, test_scenario_users, class_indices, val_data_transform, device, args, cuda_args)
    elif args.mobile_tree_net:
        print("Mobile Tree Net")
        # root_node = utils.generate(no_classes, samples, load, prob=args.pref_prob)
        models, leaf_node_labels = generate_model_list(root_node, args.depth, device, fcl_factor, model=1,
                                                       root_step=args.root_step, step=args.conv_step, dividing_factor=args.div_factor, dividing_step=args.div_step,
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
                    if isinstance(model, MobileTreeLeafNet) or isinstance(models[i], VGG_Leaf):
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
    elif args.mobile_tree_net_old or args.vgg_tree:
        if args.mobile_tree_net_old:
            print("Mobile Tree Net Old")
            modelno = 1
        else:
            print("VGG Tree Net")
            modelno = 2
        # root_node = utils.generate(no_classes, samples, load, prob=args.pref_prob)
        models, leaf_node_labels = generate_model_list(root_node, args.depth, device, fcl_factor, model=modelno,
                                                       root_step=args.root_step, step=args.conv_step, dividing_factor=args.div_factor, dividing_step=args.div_step,
                                                       not_involve=args.not_involve, log=(args.log and not args.limit_log))
        print(leaf_node_labels)
        for i in leaf_node_labels:
            print(len(i), end=" ")
        print()
        if args.log and not args.limit_log:
            logging.info("Mobile Tree Net Old") if modelno == 1 else logging.info("VGG Tree Net")
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
                    if isinstance(model, MobileTreeLeafNet) or isinstance(models[i], VGG_Leaf):
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
            if args.test_scenario:
                test_tree_scenario(models, leaf_node_labels, test_scenario_users, class_indices, val_data_transform, device, args, cuda_args)
    elif args.parallel_mobile_nets or args.parallel_vgg:
        if args.parallel_mobile_nets:
            cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]
        elif args.parallel_vgg:
            cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        # root_node = utils.generate(no_classes, samples, load, prob=args.pref_prob)
        leaf_node_labels = find_leaf_node_labels(root_node, args.depth)
        print(leaf_node_labels)
        for i in leaf_node_labels:
            print(len(i), end=" ")
        print()
        for i in range(0, len(cfg), args.div_step):
            if isinstance(cfg[i], int):
                cfg[i] = int(cfg[i] // args.div_factor)
            elif isinstance(cfg[i], tuple):
                cfg[i] = (int(cfg[i][0] // args.div_factor), cfg[i][1])
        print(cfg)
        models = []
        for i in leaf_node_labels:
            branches = len(i) + 1
            if args.parallel_mobile_nets:
                models.append(MobileNet(num_classes=branches, channels=cfg, fcl=(int((fcl_factor*fcl_factor*cfg[-1])))).to(device))
            elif args.parallel_vgg:
                models.append(VGG16(num_classes=branches, cfg=cfg, fcl=(int((fcl_factor*fcl_factor*512) // args.div_factor))).to(device))
        if args.log:
            if args.parallel_mobile_nets:
                logging.info("Parallel Mobile Nets")
            elif args.parallel_vgg:
                logging.info("Parallel VGG Nets")
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
                    state = torch.load('./saved/parallel_net' + str(i) + '.pth')
                    models[i].load_state_dict(state['model'])
                    best_acc = state['acc']
                    last_epoch = state['epoch']
                    if state['vis']:
                        vis.win_acc = state['vis-win-acc']
                        vis.win_loss = state['vis-win-loss']
            args.epochs += last_epoch
            for epoch in range(last_epoch + 1, args.epochs + 1):
                if args.just_train:
                    train_parallel_net(models, leaf_node_labels, train_loader, device, epoch, args, use_cuda)
                else:
                    train_parallel_net(models, leaf_node_labels, train_loader, device, epoch, args, use_cuda)
                    if prefs is None:
                        test_parallel_net(models, leaf_node_labels, val_loader, device, args, epoch)
                        if test_prefs:
                            preference_table = np.load('preference_table.npy')
                            all_prefs = pref_table_to_all_prefs(preference_table.T)  # change binary table to list of labels
                            test_parallel_all_preferences(models, leaf_node_labels, val_loader, device, args, all_prefs)
                            if args.calc_storage:
                                calculate_params_all_preferences_parallel(models, all_prefs, leaf_node_labels, args.log)
                    else:
                        test_parallel_net(models, leaf_node_labels, val_loader, device, args, epoch)
                        test_parallel_personal(models, leaf_node_labels, val_loader, device, args, prefs)
        else:
            for i in range(len(models)):
                state = torch.load('./saved/parallel_net' + str(i) + '.pth')
                models[i].load_state_dict(state['model'])
                best_acc = state['acc']
            if prefs is None:
                test_parallel_net(models, leaf_node_labels, val_loader, device, args)
                if test_prefs:
                    preference_table = np.load('preference_table.npy')
                    all_prefs = pref_table_to_all_prefs(preference_table.T)  # change binary table to list of labels
                    test_parallel_all_preferences(models, leaf_node_labels, val_loader, device, args, all_prefs)
                    if args.calc_storage:
                        calculate_params_all_preferences_parallel(models, all_prefs, leaf_node_labels, args.log)
            else:
                test_parallel_net(models, leaf_node_labels, val_loader, device, args)
                test_parallel_personal(models, leaf_node_labels, val_loader, device, args, prefs)
            if args.test_scenario:
                test_parallel_scenario(models, leaf_node_labels, test_scenario_users, class_indices, val_data_transform, device, args, cuda_args)

    if args.log:
        end_time = time.time()
        logging.info(time.asctime(time.localtime(end_time)))
        logging.info("--- %s seconds ---" % (end_time - start_time))
        logging.info("---END---\n")


if __name__ == '__main__':
    main()
