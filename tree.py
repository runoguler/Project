import numpy as np
import argparse
import torch
from torchvision import datasets, transforms

from models.tree_net import TreeRootNet, TreeBranchNet
from models.simplenet import SimpleNet
from models.mobilenet import MobileNet


def train_tree(models, train_loader, device, epoch, args, LongTensor):
    models[0].train()
    models[1].train()
    models[2].train()

    loss_b1 = torch.nn.CrossEntropyLoss()   # TODO: Pass Class Weigts
    loss_b2 = torch.nn.CrossEntropyLoss()   # TODO: Pass Class Weigts
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


def test_tree(models, test_loader, device, LongTensor):
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


def main():
    batch_size = 64
    test_batch_size = 1000
    epochs = 20
    lr = 0.002

    parser = argparse.ArgumentParser(description="Parameters for Training CIFAR-10")
    parser.add_argument('--test', action='store_true', help='enables test mode')
    parser.add_argument('--resume', action='store_true', help='whether to resume training or not (default: 0)')
    parser.add_argument('--simple-net', action='store_true', help='train simple-net instead of tree-net')
    parser.add_argument('--mobile-net', action='store_true', help='train simple-net instead of tree-net')
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
    else:
        models = [TreeRootNet().to(device), TreeBranchNet().to(device), TreeBranchNet().to(device)]
        LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

        if not test:
            if resume:
                models[0].load_state_dict(torch.load('./saved/root.pth'))
                models[1].load_state_dict(torch.load('./saved/branch1.pth'))
                models[2].load_state_dict(torch.load('./saved/branch2.pth'))
            for epoch in range(1, args.epochs + 1):
                train_tree(models, train_loader, device, epoch, args, LongTensor)
                test_tree(models, test_loader, device, LongTensor)

            torch.save(models[0].state_dict(), './saved/root.pth')
            torch.save(models[1].state_dict(), './saved/branch1.pth')
            torch.save(models[2].state_dict(), './saved/branch2.pth')

        if test:
            models[0].load_state_dict(torch.load('./saved/root.pth'))
            models[1].load_state_dict(torch.load('./saved/branch1.pth'))
            models[2].load_state_dict(torch.load('./saved/branch2.pth'))

            test_tree(models, test_loader, device, LongTensor)


if __name__ == '__main__':
    main()
