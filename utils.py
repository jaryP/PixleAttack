import csv
import os
import sys
from typing import Sequence

import numpy as np
import torch
from torch import optim, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.datasets.folder import default_loader, DatasetFolder, \
    ImageFolder
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, \
    RandomHorizontalFlip, RandomCrop, transforms
from tqdm import tqdm
from pathlib import Path
from base import Cub2011


class TinyImagenet(Dataset):
    """Tiny Imagenet Pytorch Dataset,
    based on Avalanche implementation: https://github.com/ContinualAI/avalanche/
    blob/4763ceacd1ab961167d1a1deddbf88a9a10220a0/avalanche/benchmarks/datasets/
    tiny_imagenet/tiny_imagenet.py"""

    filename = ('tiny-imagenet-200.zip',
                'http://cs231n.stanford.edu/tiny-imagenet-200.zip')

    md5 = '90528d7ca1a48142e341f4ef8d21d0de'

    def __init__(
            self,
            root,
            *,
            train: bool = True,
            transform=None,
            target_transform=None,
            loader=default_loader,
            download=True):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.loader = loader

        self.root = Path(root).expanduser()

        self.data_folder = self.root / 'tiny-imagenet-200'

        label2id = {}
        id2label = {}

        with open(str(os.path.join(self.data_folder, 'wnids.txt')), 'r') as f:

            reader = csv.reader(f)
            curr_idx = 0
            for ll in reader:
                if ll[0] not in label2id:
                    label2id[ll[0]] = curr_idx
                    id2label[curr_idx] = ll[0]
                    curr_idx += 1

        self.label2id, self.id2label = label2id, id2label

        self.data, self.targets = self.load_data()

    @staticmethod
    def labels2dict(data_folder):
        """
        Returns dictionaries to convert class names into progressive ids
        and viceversa.
        :param data_folder: The root path of tiny imagenet
        :returns: label2id, id2label: two Python dictionaries.
        """

        label2id = {}
        id2label = {}

        with open(str(os.path.join(data_folder, 'wnids.txt')), 'r') as f:

            reader = csv.reader(f)
            curr_idx = 0
            for ll in reader:
                if ll[0] not in label2id:
                    label2id[ll[0]] = curr_idx
                    id2label[curr_idx] = ll[0]
                    curr_idx += 1

        return label2id, id2label

    def load_data(self):
        """
        Load all images paths and targets.
        :return: train_set, test_set: (train_X_paths, train_y).
        """

        data = [[], []]

        classes = list(range(200))
        for class_id in classes:
            class_name = self.id2label[class_id]

            if self.train:
                X = self.get_train_images_paths(class_name)
                Y = [class_id] * len(X)
            else:
                # test set
                X = self.get_test_images_paths(class_name)
                Y = [class_id] * len(X)

            data[0] += X
            data[1] += Y

        return data

    def get_train_images_paths(self, class_name):
        """
        Gets the training set image paths.
        :param class_name: names of the classes of the images to be
            collected.
        :returns img_paths: list of strings (paths)
        """

        train_img_folder = self.data_folder / 'train' / class_name / 'images'

        img_paths = [f for f in train_img_folder.iterdir() if f.is_file()]

        return img_paths

    def get_test_images_paths(self, class_name):
        """
        Gets the test set image paths
        :param class_name: names of the classes of the images to be
            collected.
        :returns img_paths: list of strings (paths)
        """
        val_img_folder = self.data_folder / 'val' / 'images'
        annotations_file = self.data_folder / 'val' / 'val_annotations.txt'

        valid_names = []
        with open(str(annotations_file), 'r') as f:

            reader = csv.reader(f, dialect='excel-tab')
            for ll in reader:
                if ll[1] == class_name:
                    valid_names.append(ll[0])

        img_paths = [val_img_folder / f for f in valid_names]

        return img_paths

    def __len__(self):
        """ Returns the length of the set """
        return len(self.data)

    def __getitem__(self, index):
        """ Returns the index-th x, y pattern of the set """

        path, target = self.data[index], int(self.targets[index])

        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_dataset(name, model_name, augmentation=False, path=None):
    if path is None:
        if name == 'imagenet':
            dataset_base_path = 'var/datasets/imagenet/'
        dataset_base_path = '~/datasets/'
    else:
        dataset_base_path = path
    
    if name == 'mnist':
        t = [Resize((32, 32)),
             ToTensor(),
             Normalize((0.1307,), (0.3081,)),
             ]
        if model_name == 'lenet-300-100':
            t.append(torch.nn.Flatten())

        t = Compose(t)

        train_set = datasets.MNIST(
            root=dataset_base_path,
            train=True,
            transform=t,
            download=True
        )

        test_set = datasets.MNIST(
            root=dataset_base_path,
            train=False,
            transform=t,
            download=True
        )

        classes = 10
        input_size = (1, 32, 32)

    elif name == 'svhn':
        if augmentation:
            tt = [RandomHorizontalFlip(),
                  RandomCrop(32, padding=4)]
        else:
            tt = []

        tt.extend([ToTensor(),
                   # Normalize(mn, std)
                   ])

        t = [
            ToTensor(),
            # Normalize(mn, std)
        ]

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
        ])

        t = [
            ToTensor(),
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

    elif name == 'tinyimagenet':
        tt = [
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),

        ]

        t = [
            transforms.ToTensor(),

        ]

        transform = transforms.Compose(t)
        train_transform = transforms.Compose(tt)


        train_set = TinyImagenet('~/datasets/',
                                 transform=train_transform, train=True)

        test_set = TinyImagenet('~/datasets/',
                                transform=transform, train=False)

        input_size, classes = 3, 200

    elif name == 'imagenet':
        train_set = None

        test_set = ImageFolder(os.path.join(dataset_base_path, 'val'),
                               transform=Compose([transforms.ToTensor(),
                                                  Resize((256, 256))]))

        input_size = (3, 256, 256)
        classes = 1000

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
              bins: int = 30):
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
