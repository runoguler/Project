import numpy as np
import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable

from models.deep_tree_cnn import TreeRootNet, TreeBranchNet
from models.simplenet import SimpleNet


def train_tree(models, train_loader, device, epoch, args, LongTensor):
    models[0].train()
    models[1].train()
    models[2].train()

    loss_r = torch.nn.NLLLoss()
    loss_b1 = torch.nn.NLLLoss()
    loss_b2 = torch.nn.NLLLoss()
    loss_r.to(device)
    loss_b1.to(device)
    loss_b2.to(device)

    optim_r = torch.optim.Adam(models[0].parameters(), lr=args.lr, betas=(0.5, 0.999))
    optim_b1 = torch.optim.Adam(models[1].parameters(), lr=args.lr, betas=(0.5, 0.999))
    optim_b2 = torch.optim.Adam(models[2].parameters(), lr=args.lr, betas=(0.5, 0.999))

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        root_labels = labels/LongTensor((np.ones(labels.size(0))*5))

        optim_r.zero_grad()
        output_root, layer = models[0](data)
        root_loss = loss_r(output_root, root_labels)
        root_loss.backward(retain_graph=True)
        optim_r.step()

        # Seperate data and labels into 2 according to the classification done in the root
        b1_data = torch.empty(0, 16, 8, 8)
        b2_data = torch.empty(0, 16, 8, 8)
        b1_labels = torch.empty(0, dtype=torch.long)
        b2_labels = torch.empty(0, dtype=torch.long)
        # root_probabilities = torch.exp(output_root)
        # print(root_labels[0].item())
        for i in range(output_root.size(0)):
            if root_labels[i].item() == 0: # root_probabilities[i][0] >= root_probabilities[i][1]:
                b1_data = torch.cat((b1_data, layer[i:i+1]))
                b1_labels = torch.cat((b1_labels, labels[i:i+1]))
            else:
                b2_data = torch.cat((b2_data, layer[i:i+1]))
                b2_labels = torch.cat((b2_labels, labels[i:i+1] - 5))

        '''
        print(b1_data.shape)
        print(b2_data.shape)
        print(b1_labels.shape)
        print(b2_labels.shape)
        '''

        if b1_data.size(0):
            optim_b1.zero_grad()
            output_b1, _ = models[1](b1_data)
            b1_loss = loss_b1(output_b1, b1_labels)      ## more class labels than output options
            b1_loss.backward(retain_graph=True)
            optim_b1.step()
        if b2_data.size(0):
            optim_b2.zero_grad()
            output_b2, _ = models[2](b2_data)
            b2_loss = loss_b2(output_b2, b2_labels)      ## more class labels than output options
            b2_loss.backward()
            optim_b2.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tRoot Loss: {:.6f}\tB1 Loss: {:.6f}\tB2 Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), root_loss.item(), b1_loss.item(), b2_loss.item()))


def test_tree(models, test_loader, device, LongTensor):
    models[0].eval()
    models[1].eval()
    models[2].eval()

    loss_b1 = torch.nn.NLLLoss()
    loss_b2 = torch.nn.NLLLoss()
    loss_b1.to(device)
    loss_b2.to(device)

    test_loss = 0
    correct_root = 0
    correct_b1 = 0
    correct_b2 = 0
    for data, label in test_loader:
        data, labels = data.to(device), label.to(device)

        output_root, layer = models[0](data)

        # Seperate data and labels into 2 according to the classification done in the root
        b1_data = torch.empty(0, 16, 8, 8)
        b2_data = torch.empty(0, 16, 8, 8)
        b1_labels = torch.empty(0, dtype=torch.long)
        b2_labels = torch.empty(0, dtype=torch.long)
        root_probabilities = torch.exp(output_root)
        root_labels = labels / LongTensor((np.ones(labels.size(0)) * 5))
        pred_root = root_probabilities.max(1, keepdim=True)[1]
        correct_root += pred_root.eq(root_labels.view_as(pred_root)).sum().item()
        for i in range(output_root.size(0)):
            if root_probabilities[i][0] >= root_probabilities[i][1]:
                b1_data = torch.cat((b1_data, layer[i:i + 1]))
                b1_labels = torch.cat((b1_labels, labels[i:i + 1]))
            else:
                b2_data = torch.cat((b2_data, layer[i:i + 1]))
                b2_labels = torch.cat((b2_labels, labels[i:i + 1] - 5))

        if b1_data.size(0):
            output_b1, _ = models[1](b1_data)
            # b1_loss = loss_b1(output_b1, b1_labels)
            # print(output_b1)
            # print(output_b1.max(1, keepdim=True)[1])
            # print(b1_labels)
            pred = output_b1.max(1, keepdim=True)[1]
            correct_b1 += pred.eq(b1_labels.view_as(pred)).sum().item()
        if b2_data.size(0):
            output_b2, _ = models[2](b2_data)
            # b2_loss = loss_b2(output_b2, b2_labels)
            pred = output_b2.max(1, keepdim=True)[1]
            correct_b2 += pred.eq(b2_labels.view_as(pred)).sum().item()

        # output = model(data)
        # test_loss += loss(output, labels).item()
        # pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        # correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.0f}%), Root Accuracy: {}/{} ({:.0f}%), Corrects: {}, {}\n'.format(
        (correct_b1 + correct_b2), len(test_loader.dataset),
        100. * (correct_b1 + correct_b2) / len(test_loader.dataset),
        correct_root, len(test_loader.dataset),
        100. * correct_root / len(test_loader.dataset),
        correct_b1, correct_b2
    ))



def train(model, train_loader, device, epoch, args):
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


def test(model, test_loader, device):
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
    train = 0
    batch_size = 64
    test_batch_size = 1000
    epochs = 20
    lr = 0.002
    resume = 0

    parser = argparse.ArgumentParser(description="Parameters for Training CIFAR-10")
    parser.add_argument('--batch-size', type=int, default=batch_size, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=test_batch_size, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=epochs, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=lr, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--num-workers', type=int, default=1, metavar='N', help='number of workers for cuda')
    parser.add_argument('--model-no', type=int, default=1, metavar='N', help='number of workers for cuda')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cuda_args = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

    train_data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomHorizontalFlip(0.4),
        transforms.RandomRotation(20),
        transforms.RandomAffine(45, (0.2, 0.2))
    ])
    test_data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    cifar_training_data = datasets.CIFAR10("../data/CIFAR10", train=True, transform=train_data_transform, download=True)
    cifar_testing_data = datasets.CIFAR10("../data/CIFAR10", train=False, transform=test_data_transform)
    train_loader = torch.utils.data.DataLoader(cifar_training_data, batch_size=args.batch_size, shuffle=True, **cuda_args)
    test_loader = torch.utils.data.DataLoader(cifar_testing_data, batch_size=args.test_batch_size, shuffle=True, **cuda_args)

    '''
    model = SimpleNet().to(device)

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, device, epoch, args)
        test(model, test_loader, device)
    '''

    models = [TreeRootNet().to(device), TreeBranchNet().to(device), TreeBranchNet().to(device)]
    LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

    if train:
        for epoch in range(1, args.epochs + 1):
            train_tree(models, train_loader, device, epoch, args, LongTensor)
            # test_tree(models, test_loader, device)

        torch.save(models[0].state_dict(), './saved/root.pth')
        torch.save(models[1].state_dict(), './saved/branch1.pth')
        torch.save(models[2].state_dict(), './saved/branch2.pth')

    if not train:
        models[0].load_state_dict(torch.load('./saved/root.pth'))
        models[1].load_state_dict(torch.load('./saved/branch1.pth'))
        models[2].load_state_dict(torch.load('./saved/branch2.pth'))

        test_tree(models, test_loader, device, LongTensor)


if __name__ == '__main__':
    main()
