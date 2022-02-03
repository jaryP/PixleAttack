import json
import logging
import os
import time
from collections import defaultdict
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, TensorDataset
from torchattacks import OnePixel
from tqdm import tqdm
from attacks.psa import PatchesSwap

from base import get_model
from evaluators import accuracy_score

from attacks.base import IndexedDataset, get_default_attack_config, get_attack
from utils import get_dataset, get_optimizer, HiddenPrints


def calculate_scores(results: dict):
    times = []
    corrects = 0
    total = 0

    for key in results.keys():
        if key in ['values', 'name']:
            continue

        item = results[key]

        label = item['label']
        res = item['attacks'][str(label)]
        # print(label, item['attacks'])
        pred = res['prediction']
        times.append(res['time'])

        total += 1
        if label != pred:
            corrects += 1

    mean_time, std_time = np.mean(times), np.std(times)

    correctly_attacked = corrects / total

    return correctly_attacked, mean_time, std_time


class NpEncoder(json.JSONEncoder):
    def default(self, object):
        if isinstance(object, np.generic):
            return object.item()
        return super(NpEncoder, self).default(object)

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


def standard_training(cfg: DictConfig, save_path: str,
                      device: str):
    log = logging.getLogger(__name__)

    cfg = OmegaConf.to_container(cfg)

    experiment_cfg = cfg['experiment']
    load, save, path, experiments = experiment_cfg.get('load', True), \
                                    experiment_cfg.get('save', True), \
                                    experiment_cfg.get('path', None), \
                                    experiment_cfg.get('experiments', 1)

    if path is None:
        path = os.getcwd()
    else:
        os.chdir(path)
        os.makedirs(path, exist_ok=True)

    model_cfg = cfg['model']
    model_name = model_cfg['name']

    dataset_cfg = cfg['dataset']
    dataset_name = dataset_cfg['name']
    augmented_dataset = dataset_cfg.get('augment', False)

    experiment_cfg = cfg['experiment']

    training_cfg = cfg['training']
    epochs, batch_size = training_cfg['epochs'], \
                         training_cfg['batch_size']

    optimizer_cfg = cfg['optimizer']
    optimizer_name, lr, momentum, weight_decay = optimizer_cfg.get('optimizer',
                                                                   'sgd'), \
                                                 optimizer_cfg.get('lr', 1e-1), \
                                                 optimizer_cfg.get('momentum',
                                                                   0.9), \
                                                 optimizer_cfg.get(
                                                     'weight_decay', 0)

    os.makedirs(path, exist_ok=True)

    train_set, test_set, input_size, classes = \
        get_dataset(name=dataset_name,
                    model_name=None,
                    augmentation=augmented_dataset)

    train_set = IndexedDataset(train_set)
    test_set = IndexedDataset(test_set)

    trainloader = torch.utils.data.DataLoader(train_set,
                                              batch_size=batch_size,
                                              shuffle=True)

    testloader = torch.utils.data.DataLoader(test_set,
                                             batch_size=batch_size,
                                             shuffle=False)

    model = get_model(model_name,
                      image_size=input_size,
                      classes=classes + 1)

    model.to(device)

    if os.path.exists(os.path.join(save_path,
                                   'model.pt')):
        log.info('Model loaded.')

        model.load_state_dict(torch.load(
            os.path.join(save_path, 'model.pt'),
            map_location=device))

    else:
        log.info('Training model.')

        parameters = chain(model.parameters())

        optimizer = get_optimizer(parameters=parameters,
                                  name=optimizer_name,
                                  lr=lr,
                                  momentum=momentum,
                                  weight_decay=weight_decay)

        model.to(device)

        bar = tqdm(range(epochs))

        for epoch in bar:
            model.train()

            for _, (inputs, labels, i) in enumerate(trainloader, 0):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = nn.functional.cross_entropy(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()

            test_score, _, _ = accuracy_score(model=model,
                                              dataset=testloader,
                                              device=device)

            train_score, _, _ = accuracy_score(model=model,
                                               dataset=trainloader,
                                               device=device)

            bar.set_postfix({'train score': train_score,
                             'test sore': test_score})

        torch.save(model.state_dict(),
                   os.path.join(save_path, 'model.pt'))

    model.eval()

    score, _, _ = accuracy_score(model=model,
                                 dataset=trainloader,
                                 device=device)

    log.info('Train score: {}'.format(score))

    score, _, _ = accuracy_score(model=model,
                                 dataset=testloader,
                                 device=device)

    log.info('Test score: {}'.format(score))

    return model.state_dict()
