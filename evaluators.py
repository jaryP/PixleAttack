import tabnanny

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def accuracy_score(dataset: DataLoader,
                   model: nn.Module,
                   device: str = 'cpu'):

    total = 0
    correct = 0

    model.eval()
    model.to(device)

    for t in dataset:
        image, label = t[:2]
        image = image.to(device)
        label = label.to(device)

        pred = model(image)
        pred = torch.argmax(pred, 1)

        total += label.size(0)
        correct += (pred == label).sum().item()

    score = correct / total

    return score, total, correct


@torch.no_grad()
def get_probabilities(dataset: DataLoader,
                      model: nn.Module,
                      device: str = 'cpu'):
    ground_truth = []
    predicted = []
    probs = []

    model.eval()
    model.to(device)

    for image, label in dataset:
        image = image.to(device)
        label = label.to(device)

        logits = model(image)

        p = torch.softmax(logits, dim=-1)

        _, pred = torch.max(model(image), 1)

        predicted.extend(pred.cpu().tolist())
        ground_truth.extend(label.cpu().tolist())
        probs.extend(p.cpu().tolist())

    return ground_truth, predicted, probs


@torch.no_grad()
def ece_score(dataset: DataLoader,
              model: nn.Module,
              device: str = 'cpu',
              bins: int = 30):

    ground_truth, predictions, probs = get_probabilities(dataset,
                                                         model,
                                                         device)

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

