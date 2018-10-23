import numpy as np
import argparse
import torch
from torchvision import datasets, transforms

from models.tree_net import TreeRootNet, TreeBranchNet
from models.simplenet import SimpleNet
from models.mobilenet import MobileNet
from models.mobile_static_tree_net import StaticTreeRootNet, StaticTreeBranchNet
from models.mobile_tree_net import MobileTreeRootNet, MobileTreeLeafNet, MobileTreeBranchNet

import utils


def train_tree(models, train_loader, device, epoch, args, use_cuda):
    models[0].train()
    models[1].train()
    models[2].train()

    # FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    # weights = [1.0, 1.0, 1.0, 1.0, 1.0, 0.2]
    # class_weights = FloatTensor(weights).to(device)

    loss_b1 = torch.nn.CrossEntropyLoss()
    loss_b2 = torch.nn.CrossEntropyLoss()
    loss_b1.to(device)
    loss_b2.to(device)

    optim_b1 = torch.optim.Adam(list(models[0].parameters()) + list(models[1].parameters()), lr=args.lr, betas=(0.5, 0.999))
    optim_b2 = torch.optim.Adam(list(models[0].parameters()) + list(models[2].parameters()), lr=args.lr, betas=(0.5, 0.999))

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        optim_b1.zero_grad()
        optim_b2.zero_grad()

        layer = models[0](data)
        out_b1, _ = models[1](layer)
        out_b2, _ = models[2](layer)

        b1_labels = labels.clone()
        b2_labels = labels.clone() - 5

        b1_labels[b1_labels > 4] = 5
        b2_labels[b2_labels < 0] = 5

        # b1_labels = torch.zeros((labels.size(0), 5), device=device)
        # b2_labels = torch.zeros((labels.size(0), 5), device=device)
        #
        # for i in range(labels.size(0)):
        #     if labels[i].item() < 5:
        #         b1_labels[i][labels[i].item()] = 1
        #     else:
        #         b2_labels[i][labels[i].item()-5] = 1


        b1_loss = loss_b1(out_b1, b1_labels)
        b1_loss.backward(retain_graph=True)  ## retain_graph=True
        optim_b1.step()


        b2_loss = loss_b2(out_b2, b2_labels)
        b2_loss.backward()
        optim_b2.step()
        

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tB1 Loss: {:.6f}\tB2 Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), b1_loss.item(), b2_loss.item()))


        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #                100. * batch_idx / len(train_loader), l.item(), ))


def test_tree(models, test_loader, device, use_cuda):
    models[0].eval()
    models[1].eval()
    models[2].eval()

    # correct_b1 = 0
    # correct_b2 = 0
    corrects = 0
    no_class = 0
    both_class = 0
    false_in_class = 0
    correct_from_both = 0
    max_correct_from_both = 0

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
                        if (out_b1[i][pred_b1[i].item()].item() > out_b2[i][pred_b2[i].item()].item()) and (labels[i].item() == pred_b1[i].item()):
                            max_correct_from_both += 1
                        elif (out_b2[i][pred_b2[i].item()].item() > out_b1[i][pred_b1[i].item()].item()) and (labels[i].item() == (pred_b2[i].item() + 5)):
                            max_correct_from_both += 1


        # out = torch.cat((out_b1, out_b2), dim=1)

        # pred = out.max(1, keepdim=True)[1]
        # corrects += pred.eq(labels.view_as(pred)).sum().item()

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n\t  F_in: {} None: {} Both: ({}/{}/{})\n'.format(
        corrects, len(test_loader.dataset),
        100. * corrects / len(test_loader.dataset),
        false_in_class, no_class, both_class, correct_from_both, max_correct_from_both
    ))

    # print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     corrects, len(test_loader.dataset),
    #     100. * corrects / len(test_loader.dataset)
    # ))


def train_dynamic_tree(models, leaf_node_labels, train_loader, device, epoch, args, use_cuda):
    leaf_node_index = []
    leaf_node_paths = []    # NOT INCLUDING models[0]

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
            i = (i+1)//2 - 1
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
        for i in range(len(leaf_node_paths)):   # for every branch(path) going to a leaf node
            layer = models[0](data)
            for j in range(len(leaf_node_paths[i])):
                k = leaf_node_paths[i][j]
                if j+1 == len(leaf_node_paths[i]):
                    result, _= models[k](layer)

                    lbls = labels.clone()
                    for l in range(len(lbls)):
                        if lbls[l].item() == leaf_node_labels[i] or (not isinstance(leaf_node_labels[i], int) and lbls[l].item() in leaf_node_labels[i]):
                            lbls[l] = leaf_node_labels[i].index(lbls[l])
                        else:
                            if not isinstance(leaf_node_labels[i], int):
                                lbls[l] = len(leaf_node_labels[i])
                            else:
                                lbls[l] = 1

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


def test_dynamic_tree(models, leaf_node_labels, test_loader, device, use_cuda):
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
            i = (i+1)//2 - 1
        leaf_node_paths.append(path)

    definite_correct = 0
    indefinite_correct = 0
    wrong = 0

    for data, label in test_loader:
        data, labels = data.to(device), label.to(device)

        pred = []
        layer = models[0](data)
        for i in range(len(leaf_node_paths)):  # for every branch(path) going to a leaf node
            for j in range(len(leaf_node_paths[i])):
                k = leaf_node_paths[i][j]
                if j + 1 == len(leaf_node_paths[i]):
                    result, _ = models[k](layer)

                    lbls = labels.clone()
                    for l in range(len(lbls)):
                        if lbls[l].item() == leaf_node_labels[i] or (not isinstance(leaf_node_labels[i], int) and lbls[l].item() in leaf_node_labels[i]):
                            lbls[l] = leaf_node_labels[i].index(lbls[l])
                        else:
                            if not isinstance(leaf_node_labels[i], int):
                                lbls[l] = len(leaf_node_labels[i])
                            else:
                                lbls[l] = 1

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

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n\tDefinite Corrects: {}\n'.format(
        (definite_correct + indefinite_correct), len(test_loader.dataset),
        100. * (definite_correct + indefinite_correct) / len(test_loader.dataset), definite_correct
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
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss.item()))


def test_net(model, test_loader, device):
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
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def generate_model_list(root_node, lvl, device):
    leaf_node_labels = []
    cfg_full = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]
    root_step = 1
    models = [MobileTreeRootNet(cfg_full[:root_step]).to(device)]
    nodes = [root_node]
    index = 0
    remaining = 1
    steps = [3,6,6,9,9,9,9,12,12,12,12,12,12,12,12,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
    while remaining > 0:
        if nodes[index] is None:
            models.append(None)
            models.append(None)
            nodes.append(None)
            nodes.append(None)
            index += 1
            continue

        conv_step = steps[index] - (3 - root_step)

        in_planes = cfg_full[conv_step-1] if isinstance(cfg_full[conv_step-1], int) else cfg_full[conv_step-1][0]

        # LEFT BRANCH
        left = nodes[index].left
        if not isinstance(left, int):
            if left.count > 3:
                models.append(MobileTreeBranchNet(input=cfg_full[conv_step:conv_step+3], in_planes=in_planes).to(device))
                nodes.append(left)
                remaining += 1
            else:
                models.append(MobileTreeLeafNet(branch=(left.count+1), input=cfg_full[conv_step:], in_planes=in_planes).to(device))
                nodes.append(None)
                leaf_node_labels.append(left.value)
        else:
            models.append(MobileTreeLeafNet(branch=2, input=cfg_full[conv_step:], in_planes=in_planes).to(device))
            nodes.append(None)
            leaf_node_labels.append(left)

        # RIGHT BRANCH
        right = nodes[index].right
        if not isinstance(right, int):
            if right.count > 3:
                models.append(MobileTreeBranchNet(input=cfg_full[conv_step:conv_step+3], in_planes=in_planes).to(device))
                nodes.append(right)
                remaining += 1
            else:
                models.append(MobileTreeLeafNet(branch=(right.count+1), input=cfg_full[conv_step:], in_planes=in_planes).to(device))
                nodes.append(None)
                leaf_node_labels.append(right.value)
        else:
            models.append(MobileTreeLeafNet(branch=2, input=cfg_full[conv_step:], in_planes=in_planes).to(device))
            nodes.append(None)
            leaf_node_labels.append(right)

        index += 1
        remaining -= 1
    print(root_node)
    # print(models)
    return models, leaf_node_labels


def main():
    batch_size = 64
    test_batch_size = 1000
    epochs = 20
    lr = 0.002
    depth = 2

    parser = argparse.ArgumentParser(description="Parameters for Training CIFAR-10")
    parser.add_argument('--test', action='store_true', help='enables test mode')
    parser.add_argument('--resume', action='store_true', help='whether to resume training or not (default: 0)')
    parser.add_argument('--simple-net', action='store_true', help='train simple-net instead of tree-net')
    parser.add_argument('--mobile-net', action='store_true', help='train mobile-net instead of tree-net')
    parser.add_argument('--mobile-static-tree-net', action='store_true', help='train mobile-static-tree-net instead of tree-net')
    parser.add_argument('--mobile-tree-net', action='store_true', help='train mobile-tree-net instead of tree-net')
    parser.add_argument('--depth', type=int, default=depth, choices=[1,2,3], metavar='lvl', help='depth of the tree')
    parser.add_argument('--batch-size', type=int, default=batch_size, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=test_batch_size, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=epochs, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=lr, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--num-workers', type=int, default=1, metavar='N', help='number of workers for cuda')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    test = args.test
    resume = args.resume
    # test = True
    # resume = True

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cuda_args = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

    train_data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.4),
        transforms.RandomRotation(20),
        transforms.RandomAffine(45, (0.2, 0.2)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    cifar_training_data = datasets.CIFAR10("../data/CIFAR10", train=True, transform=train_data_transform, download=True)
    cifar_testing_data = datasets.CIFAR10("../data/CIFAR10", train=False, transform=test_data_transform)
    train_loader = torch.utils.data.DataLoader(cifar_training_data, batch_size=args.batch_size, shuffle=True,
                                               **cuda_args)
    test_loader = torch.utils.data.DataLoader(cifar_testing_data, batch_size=args.test_batch_size, shuffle=True,
                                              **cuda_args)

    args.mobile_tree_net = True

    if args.simple_net:
        model = SimpleNet().to(device)

        if not test:
            if resume:
                model.load_state_dict(torch.load('./saved/simplenet.pth'))
            for epoch in range(1, args.epochs + 1):
                train_net(model, train_loader, device, epoch, args)
                test_net(model, test_loader, device)
            torch.save(model.state_dict(), './saved/simplenet.pth')
        else:
            model.load_state_dict(torch.load('./saved/simplenet.pth'))
            test(model, test_loader, device)
    elif args.mobile_net:
        model = MobileNet().to(device)

        if not test:
            if resume:
                model.load_state_dict(torch.load('./saved/mobilenet.pth'))
            for epoch in range(1, args.epochs + 1):
                train_net(model, train_loader, device, epoch, args)
                test_net(model, test_loader, device)
            torch.save(model.state_dict(), './saved/mobilenet.pth')
        else:
            model.load_state_dict(torch.load('./saved/mobilenet.pth'))
            test(model, test_loader, device)
    elif args.mobile_static_tree_net:
        models = [StaticTreeRootNet().to(device), StaticTreeBranchNet().to(device), StaticTreeBranchNet().to(device)]
        # LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

        if not test:
            if resume:
                models[0].load_state_dict(torch.load('./saved/root.pth'))
                models[1].load_state_dict(torch.load('./saved/branch1.pth'))
                models[2].load_state_dict(torch.load('./saved/branch2.pth'))
            for epoch in range(1, args.epochs + 1):
                train_tree(models, train_loader, device, epoch, args, use_cuda)
                test_tree(models, test_loader, device, use_cuda)

            torch.save(models[0].state_dict(), './saved/root.pth')
            torch.save(models[1].state_dict(), './saved/branch1.pth')
            torch.save(models[2].state_dict(), './saved/branch2.pth')

        if test:
            models[0].load_state_dict(torch.load('./saved/root.pth'))
            models[1].load_state_dict(torch.load('./saved/branch1.pth'))
            models[2].load_state_dict(torch.load('./saved/branch2.pth'))

            test_tree(models, test_loader, device, use_cuda)
    elif args.mobile_tree_net:
        root_node = utils.generate(10, 80)
        lvl = args.depth
        models, leaf_node_labels = generate_model_list(root_node, lvl, device)

        if not test:
            if resume:
                for i in range(len(models)):
                    if not models[i] is None:
                        models[i].load_state_dict(torch.load('./saved/treemodel' + str(i) + '.pth'))
            for epoch in range(1, args.epochs + 1):
                train_dynamic_tree(models, leaf_node_labels, train_loader, device, epoch, args, use_cuda)
                test_dynamic_tree(models, leaf_node_labels, test_loader, device, use_cuda)

            for i in range(len(models)):
                if not models[i] is None:
                    torch.save(models[i].state_dict(), './saved/treemodel' + str(i) + '.pth')

        if test:
            for i in range(len(models)):
                if not models[i] is None:
                    models[i].load_state_dict(torch.load('./saved/treemodel' + str(i) + '.pth'))

            test_dynamic_tree(models, leaf_node_labels, test_loader, device, use_cuda)
    else:
        models = [TreeRootNet().to(device), TreeBranchNet().to(device), TreeBranchNet().to(device)]
        # LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

        if not test:
            if resume:
                models[0].load_state_dict(torch.load('./saved/root.pth'))
                models[1].load_state_dict(torch.load('./saved/branch1.pth'))
                models[2].load_state_dict(torch.load('./saved/branch2.pth'))
            for epoch in range(1, args.epochs + 1):
                train_tree(models, train_loader, device, epoch, args, use_cuda)
                test_tree(models, test_loader, device, use_cuda)

            torch.save(models[0].state_dict(), './saved/root.pth')
            torch.save(models[1].state_dict(), './saved/branch1.pth')
            torch.save(models[2].state_dict(), './saved/branch2.pth')

        if test:
            models[0].load_state_dict(torch.load('./saved/root.pth'))
            models[1].load_state_dict(torch.load('./saved/branch1.pth'))
            models[2].load_state_dict(torch.load('./saved/branch2.pth'))

            test_tree(models, test_loader, device, use_cuda)


if __name__ == '__main__':
    main()
