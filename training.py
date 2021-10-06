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
        # if isinstance(object, tuple):
        #     return list(object)
        return super(NpEncoder, self).default(object)
    # def default(self, obj):
    #     if isinstance(obj, np.integer):
    #         return int(obj)
    #     if isinstance(obj, np.floating):
    #         return float(obj)
    #     if isinstance(obj, np.ndarray):
    #         return obj.tolist()
    #     if isinstance(obj, tuple):
    #         return list(obj)
    #     return super(NpEncoder, self).default(obj)


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()
    # if isinstance(object, tuple):
    #     return list(object)
    # return super(NpEncoder, self).default(obj)


def standard_training(cfg: DictConfig, save_path: str,
                      device: str):
    log = logging.getLogger(__name__)

    cfg = OmegaConf.to_container(cfg)

    # log.info(OmegaConf.to_yaml(cfg))
    # print(OmegaConf.to_yaml(cfg))
    # print("Working directory : {}".format(os.getcwd()))

    # if method_name == 'naive':
    #     methods(cfg)
    # if method_name == 'batch_ensemble':
    #     batch_ensemble(cfg)
    # if method_name == 'dropout':
    #     dropout(cfg)
    # method_name = cfg['method']['name']

    experiment_cfg = cfg['experiment']
    load, save, path, experiments = experiment_cfg.get('load', True), \
                                    experiment_cfg.get('save', True), \
                                    experiment_cfg.get('path', None), \
                                    experiment_cfg.get('experiments', 1)

    # device = cfg['training'].get('device', 'cpu')

    # if torch.cuda.is_available() and device != 'cpu':
    #     torch.cuda.set_device(device)
    #     device = 'cuda:{}'.format(device)
    # else:
    #     warnings.warn("Device not found or CUDA not available."
    # device = torch.device(device)

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
    # load, save, path, experiments = experiment_cfg.get('load', True), \
    #                                 experiment_cfg.get('save', True), \
    #                                 experiment_cfg.get('path', None), \
    #                                 experiment_cfg.get('experiments', 1)

    experiments = experiment_cfg.get('experiments', 1)
    plot = experiment_cfg.get('plot', False)

    # method_cfg = cfg['method']

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

    images = []
    labels = []
    indexes = []

    counter = defaultdict(int)

    th = 1 if 'imagenet' in dataset_name else 50

    with torch.no_grad():
        for img, y, i in tqdm(DataLoader(test_set,
                                         batch_size=256,
                                         shuffle=False), leave=False):
            img = img.to(device)

            _y = model(img)
            predictions = torch.argmax(_y, -1)

            for j, (im, p, gt) in enumerate(zip(img.cpu(),
                                                predictions.cpu().numpy(),
                                                y.numpy())):

                if p == gt and counter[gt] < th:
                    counter[gt] += 1
                    images.append(im)
                    indexes.append(i[j])
                    labels.append(gt)

    images = torch.stack(images, 0)
    labels = torch.tensor(labels)
    indexes = torch.tensor(indexes)

    dataset = TensorDataset(images, labels, indexes)

    atks = cfg.get('attacks', {})
    # atks = cfg.get('attacks', {})

    for k, v in atks.items():
        v = get_default_attack_config(v)
        attack_factory = get_attack(v)

        log.info('Attack {}, Parameters {}'.format(k, v))

        attack_save_path = os.path.join(save_path, '{}.json'.format(k))
        attack_images_save_path = os.path.join(save_path, '{}_images'.format(k))

        if os.path.exists(attack_save_path) \
                and v.get('load', True):
            try:
                with open(attack_save_path, 'r') \
                        as json_file:
                    attack_results = json.load(json_file)
            except Exception as e:
                attack_results = {'name': k,
                                  'values': v}

        else:
            attack_results = {'name': k,
                              'values': v}

        attack = attack_factory(model)

        if isinstance(attack, OnePixel):
            attack._supported_mode = ['default', 'targeted']

        for img, y, i in tqdm(DataLoader(dataset, batch_size=1,
                                         shuffle=False), leave=False):

            img = img.to(device)
            probs = softmax(model(img), dim=1)[0].tolist()

            i = str(i.item())

            if i in attack_results:
                d = attack_results[i]
            else:
                d = {'label': y.item(),
                     'probs': probs,
                     'attacks': {}}

            for offset in tqdm(range(classes - 1), leave=False):

                with HiddenPrints():
                    if offset > 0:
                        break
                        f = lambda x, label: (label + offset) % classes
                        attack.set_mode_targeted_by_function(f)
                        attack_label = f(img, y)

                        if attack_label == y:
                            attack.set_mode_default()
                    else:
                        attack.set_mode_default()
                        attack_label = y

                # if str(attack_label.item()) in d['attacks']:
                #     continue

                if str(attack_label.item()) not in d['attacks']:
                    # initial_probs = attack._get_prob(img).tolist()
                    #
                    # a = {'label': y.item(),
                    #      'porbs': initial_probs}
                    #
                    # if attack.get_mode == 'targeted':
                    #     attack_label = attack._get_target_label(img, y)
                    # else:
                    #     attack_label = y

                    attack_label = attack_label.item()

                    start = time.time()
                    pert_image = attack(img, y)

                    end = time.time()
                    elapsed_time = end - start

                    if isinstance(attack, PatchesSwap):
                        pert_image, iterations, statistics = pert_image
                    else:
                        iterations = -1
                        statistics = None

                    if v.get('save_images', False):
                        os.makedirs(attack_images_save_path, exist_ok=True)
                        f = plt.figure()
                        plt.imshow(
                            np.moveaxis(pert_image.cpu().numpy()[0], 0, -1))
                        f.savefig((os.path.join(attack_images_save_path,
                                                'p{}'.format(i))))
                        plt.close(f)
                        f = plt.figure()
                        plt.imshow(
                            np.moveaxis(img.cpu().numpy()[0], 0, -1))
                        f.savefig((os.path.join(attack_images_save_path,
                                                '{}'.format(i))))
                        plt.close(f)

                        # im1 = pert_image.cpu().numpy()[0][0]
                        # im2 = img.cpu().numpy()[0][0]
                        # _, s1, _ = np.linalg.svd(im1)
                        # _, s2, _ = np.linalg.svd(im2)

                        # print(s1)
                        # print(s2)
                        # d = np.linalg.norm(s1 - s2)
                        # print(d)
                        # input()

                    final_prob = softmax(model(pert_image), dim=1)[0].tolist()

                    diff = (pert_image - img).view(-1)

                    norms = {norm_t: torch.linalg.norm(diff,
                                                       ord=norm_t).item() / 3
                             for norm_t in [0, 2, float('inf')]}

                    res = {
                        # 'attacked_label': attack_label,
                        'time': elapsed_time,
                        'probs': final_prob,
                        'prediction': np.argmax(final_prob),
                        'norms': norms,
                        'iterations': iterations,
                        'statistics': statistics}

                    d['attacks'][str(attack_label)] = res

                    # print(d)
                    attack_results[i] = d

                    # correctly_attacked, mean_time, std_time = \
                    #     calculate_scores(attack_results)
                    #
                    # print(correctly_attacked)

            with open(attack_save_path, 'w') \
                    as json_file:
                # data = json.load(json_file)
                json.dump(attack_results, json_file,
                          indent=4, cls=NpEncoder)

        corrects = 0

        times = []

        with open(os.path.join(save_path, attack_save_path), 'r') \
                as json_file:
            attack_results = json.load(json_file)

            # for key in attack_results.keys():
            #     if key in ['values', 'name']:
            #         continue
            #
            #     item = attack_results[key]
            #
            #     label = item['label']
            #     res = item['attacks'][str(label)]
            #     # print(label, item['attacks'])
            #     pred = res['prediction']
            #     times.append(res['time'])
            #
            #     if label != pred:
            #         corrects += 1
            #
            # mean_time, std_time = np.mean(times), np.std(times)
            #
            # correctly_attacked = corrects / len(dataset)

            correctly_attacked, mean_time, std_time = \
                calculate_scores(attack_results)

            log.info('\t\tCorrectly attacked: {}/{} ({})'
                     .format(int(correctly_attacked * len(dataset)),
                             len(dataset), correctly_attacked))

            log.info('\t\tTime required per image: {}(+-{})'
                     .format(mean_time, std_time))

    return model
