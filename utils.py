import os
import sys
from typing import Sequence

import numpy as np
import torch
from torch import optim, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, \
    RandomHorizontalFlip, RandomCrop, transforms
from tqdm import tqdm

from base import Cub2011


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_dataset(name, model_name, augmentation=False):
    if name == 'mnist':
        t = [Resize((32, 32)),
             ToTensor(),
             Normalize((0.1307,), (0.3081,)),
             ]
        if model_name == 'lenet-300-100':
            t.append(torch.nn.Flatten())

        t = Compose(t)

        train_set = datasets.MNIST(
            root='~/datasets/mnist/',
            train=True,
            transform=t,
            download=True
        )

        test_set = datasets.MNIST(
            root='~/datasets/mnist/',
            train=False,
            transform=t,
            download=True
        )

        classes = 10
        input_size = (1, 32, 32)

    elif name == 'flat_mnist':
        t = Compose([ToTensor(),
                     Normalize(
                         (0.1307,), (0.3081,)),
                     torch.nn.Flatten(0)
                     ])

        train_set = datasets.MNIST(
            root='~/datasets/mnist/',
            train=True,
            transform=t,
            download=True
        )

        test_set = datasets.MNIST(
            root='~/datasets/mnist/',
            train=False,
            transform=t,
            download=True
        )

        classes = 10
        input_size = 28 * 28

    elif name == 'svhn':
        if augmentation:
            tt = [RandomHorizontalFlip(),
                  RandomCrop(32, padding=4)]
        else:
            tt = []

        mn, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

        tt.extend([ToTensor(),
                   # Normalize(mn, std)
                  ])

        t = [
            ToTensor(),
            # Normalize(mn, std)
        ]

        # if 'resnet' in model_name:
        #     tt = [transforms.Resize(256), transforms.CenterCrop(224)] + tt
        #     t = [transforms.Resize(256), transforms.CenterCrop(224)] + t

        transform = Compose(t)
        train_transform = Compose(tt)

        train_set = datasets.SVHN(
            root='~/datasets/svhn', split='train', download=True,
            transform=train_transform)

        test_set = datasets.SVHN(
            root='~/datasets/svhn', split='test', download=True,
            transform=transform)

        input_size, classes = (3, 32, 32), 10

    elif name == 'cifar10':

        if augmentation:
            tt = [RandomHorizontalFlip(),
                  RandomCrop(32, padding=4)]
        else:
            tt = []

        tt.extend([
            ToTensor(),
                   # Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                   ])
        
        t = [   
            ToTensor(),
            # Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ]
        
        transform = Compose(t)
        train_transform = Compose(tt)
        
        train_set = datasets.CIFAR10(
            root='~/datasets/cifar10', train=True, download=True,
            transform=train_transform)

        test_set = datasets.CIFAR10(
            root='~/datasets/cifar10', train=False, download=True,
            transform=transform)

        input_size, classes = (3, 32, 32), 10

    # elif name == 'cifar100':
    #     tt = [
    #         RandomCrop(32, padding=4),
    #         RandomHorizontalFlip(),
    #         ToTensor(),
    #         Normalize((0.4914, 0.4822, 0.4465),
    #                              (0.2023, 0.1994, 0.2010))]
    #
    #     t = [
    #         ToTensor(),
    #         Normalize((0.4914, 0.4822, 0.4465),
    #                              (0.2023, 0.1994, 0.2010))]
    #
    #     transform = Compose(t)
    #     train_transform = Compose(tt)
    #
    #     train_set = datasets.CIFAR100(
    #         root='~/datasets/cifar100', train=True, download=True,
    #         transform=train_transform)
    #
    #     test_set = datasets.CIFAR100(
    #         root='~/datasets/cifar100', train=False, download=True,
    #         transform=transform)
    #
    #     input_size, classes = 3, 100
    #
    # elif name == 'tinyimagenet':
    #     tt = [
    #         ToTensor(),
    #         # transforms.RandomCrop(56),
    #         RandomResizedCrop(64),
    #         RandomHorizontalFlip(),
    #         Normalize((0.4802, 0.4481, 0.3975),
    #                              (0.2302, 0.2265, 0.2262))
    #     ]
    #
    #     t = [
    #         ToTensor(),
    #         Normalize((0.4802, 0.4481, 0.3975),
    #                              (0.2302, 0.2265, 0.2262))
    #     ]
    #
    #     transform = Compose(t)
    #     train_transform = Compose(tt)
    #
    #     # train_set = TinyImageNet(
    #     #     root='~/datasets/tiny-imagenet-200', split='train',
    #     #     transform=transform)
    #
    #     train_set = datasets.ImageFolder('~/datasets/tiny-imagenet-200/train',
    #                                      transform=train_transform)
    #
    #     # for x, y in train_set:
    #     #     if x.shape[exp_0] == 1:
    #     #         print(x.shape[exp_0] == 1)
    #
    #     # test_set = TinyImageNet(
    #     #     root='~/datasets/tiny-imagenet-200', split='val',
    #     #     transform=train_transform)
    #     test_set = datasets.ImageFolder('~/datasets/tiny-imagenet-200/val',
    #                                     transform=transform)
    #
    #     # for x, y in test_set:
    #     #     if x.shape[exp_0] == 1:
    #     #         print(x.shape[exp_0] == 1)
    #
    #     input_size, classes = 3, 200

    elif name == 'tinyimagenet':
        tt = [
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            # transforms.RandomCrop(56),
            # RandomCrop(64, padding=4),
            # transforms.RandomRotation(20),
            # transforms.RandomHorizontalFlip(),
            # transforms.Normalize((0.4802, 0.4481, 0.3975),
            #                      (0.2302, 0.2265, 0.2262))
        ]

        t = [
            transforms.ToTensor(),
            # transforms.Normalize((0.4802, 0.4481, 0.3975),
            #                      (0.2302, 0.2265, 0.2262))
        ]

        transform = transforms.Compose(t)
        train_transform = transforms.Compose(tt)

        # train_set = TinyImageNet(
        #     root='./datasets/tiny-imagenet-200', split='train',
        #     transform=transform)

        train_set = datasets.ImageFolder('~/datasets/tiny-imagenet-200/train',
                                         transform=train_transform)

        # for x, y in train_set:
        #     if x.shape[0] == 1:
        #         print(x.shape[0] == 1)

        # test_set = TinyImageNet(
        #     root='./datasets/tiny-imagenet-200', split='val',
        #     transform=train_transform)
        test_set = datasets.ImageFolder('~/datasets/tiny-imagenet-200/val',
                                        transform=transform)

        # for x, y in test_set:
        #     if x.shape[0] == 1:
        #         print(x.shape[0] == 1)

        input_size, classes = 3, 200

    elif name == 'cub200':
        tt = [
            transforms.ToTensor(),
            # transforms.RandomCrop(56),
            # transforms.RandomResizedCrop(64),
            # transforms.RandomHorizontalFlip(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ]

        t = [
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ]

        transform = transforms.Compose(t)
        train_transform = transforms.Compose(tt)

        train_set = Cub2011(root='~/datasets/', train=True, transform=train_transform)
        test_set = Cub2011(root='~/datasets/', train=False,
                            transform=transform)

        classes = 200
        input_size = 3

    else:
        assert False

    return train_set, test_set, input_size, classes


def get_optimizer(parameters,
                  name: str,
                  lr: float,
                  momentum: float = 0.0,
                  weight_decay: float = 0):

    name = name.lower()
    if name == 'adam':
        return optim.Adam(parameters, lr, weight_decay=weight_decay)
    elif name == 'sgd':
        return optim.SGD(parameters, lr, momentum=momentum,
                         weight_decay=weight_decay)
    else:
        raise ValueError('Optimizer must be adam or sgd')


def ece_score(ground_truth: Sequence,
              predictions: Sequence,
              probs: Sequence,
              bins: int =30):

    ground_truth = np.asarray(ground_truth)
    predictions = np.asarray(predictions)
    probs = np.asarray(probs)

    probs = np.max(probs, -1)

    prob_pred = np.zeros((0,))
    prob_true = np.zeros((0,))
    ece = 0

    mce = []

    for b in range(1, int(bins) + 1):
        i = np.logical_and(probs <= b / bins, probs > (
                    b - 1) / bins)  # indexes for p in the current bin

        s = np.sum(i)

        if s == 0:
            prob_pred = np.hstack((prob_pred, 0))
            prob_true = np.hstack((prob_true, 0))
            continue

        m = 1 / s
        acc = m * np.sum(predictions[i] == ground_truth[i])
        conf = np.mean(probs[i])

        prob_pred = np.hstack((prob_pred, conf))
        prob_true = np.hstack((prob_true, acc))
        diff = np.abs(acc - conf)

        mce.append(diff)

        ece += (s / len(ground_truth)) * diff

    return ece, prob_pred, prob_true, mce


def model_training(backbone: nn.Module,
                   classifier: nn.Module,
                   epochs: int,
                   optimizer: Optimizer,
                   dataloader: DataLoader,
                   device: str = 'cpu'):

    backbone.to(device)
    classifier.to(device)

    for epoch in tqdm(range(epochs)):
        backbone.train()
        classifier.train()

        for i, (inputs, labels) in enumerate(dataloader, 0):
            inputs = inputs.to(device)
            labels = labels.to(device)

            e = backbone(inputs)
            outputs = classifier(e)

            loss = nn.functional.cross_entropy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return backbone, classifier


@torch.no_grad()
def model_evaluator(backbone: nn.Module,
                    classifier: nn.Module,
                    dataloader: DataLoader,
                    device: str = 'cpu'):
    backbone.to(device)
    classifier.to(device)

    backbone.eval()
    classifier.eval()

    total = 0
    correct = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        e = backbone(inputs)
        outputs = classifier(e)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    score = correct / total

    return score, total, correct
