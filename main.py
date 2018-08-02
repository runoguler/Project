import torch
from torchvision import datasets, transforms

import argparse

from models.deep_tree_cnn import Net


def train(args, data_loader, device):
    epochs = args.epochs
    model = Net()
    model.train()

    loss = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

    model.to(device)
    loss.to(device)

    for epoch in range(epochs):
        ...



def main():
    batch_size = 64
    test_batch_size = 64
    epochs = 10
    lr = 0.001
    resume = 0

    parser = argparse.ArgumentParser(description="Parameters for Training Deep Tree Network on CIFAR10 dataset")
    parser.add_argument('--batch-size', type=int, default=batch_size, help='batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=test_batch_size, help='batch size for testing (default: 64)')
    parser.add_argument('--num-workers', type=int, default=1, help='number of workers for cuda')
    parser.add_argument('--lr', type=float, default=lr, help='learning rate')
    parser.add_argument('--epochs', type=int, default=epochs, help='epoch number to train (default: 10)')
    parser.add_argument('--resume', type=int, default=resume, help='continue training if 1 (default: 0)')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cuda_args = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_data = datasets.CIFAR10("../data/CIFAR10", train=True, transform=data_transform, download=True)
    test_data = datasets.CIFAR10("../data/CIFAR10", train=False, transform=data_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **cuda_args)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True, **cuda_args)

    train(args, train_loader, device)


if __name__ == '__main__':
    main()
