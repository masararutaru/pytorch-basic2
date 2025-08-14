from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(dataset: str, batch_size: int = 128, num_workers: int = 2):
    if dataset == "mnist":
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=train_tf)
        test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=test_tf)
        num_classes = 10

    elif dataset == "cifar10":
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
        test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)
        num_classes = 10
    else:
        raise ValueError(dataset)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, num_classes 