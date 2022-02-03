import json
import logging
import os
import time
import warnings
from collections import defaultdict
from itertools import chain

import hydra
import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from torch import softmax, nn
from torch.utils.data import DataLoader, TensorDataset

from torchattacks import OnePixel
from tqdm import tqdm

from attacks.base import IndexedDataset, get_default_attack_config, get_attack
from base import get_model
from evaluators import accuracy_score
from utils import HiddenPrints, get_optimizer, get_dataset


def calculate_scores(results: dict):
    times = []
    zero_norms = []
    iterations = []

    corrects = 0
    total = 0

    for key in results.keys():
        if key in ['values', 'name']:
            continue

        item = results[key]

        label = item['label']
        res = item['attacks'][str(label)]
        pred = res['prediction']
        times.append(res['time'])
        iterations.append(res['iterations'])
        zero_norms.append(res['norms']['0'])
        total += 1
        if label != pred:
            corrects += 1

    mean_time, std_time = np.mean(times), np.std(times)
    mean_iterations, std_iterations = np.mean(iterations), np.std(iterations)
    mean_zeron, std_zeron = np.mean(zero_norms), np.std(zero_norms)

    correctly_attacked = corrects / total

    return correctly_attacked, \
           mean_time, std_time, \
           mean_iterations, std_iterations, \
           mean_zeron, std_zeron


class NpEncoder(json.JSONEncoder):
    def default(self, object):
        if isinstance(object, np.generic):
            return object.item()
        return super(NpEncoder, self).default(object)


@hydra.main(config_path="configs",
            config_name="config")
def my_app(cfg: DictConfig) -> None:
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))

    experiment_cfg = cfg['experiment']
    load, save, path, experiments = experiment_cfg.get('load', True), \
                                    experiment_cfg.get('save', True), \
                                    experiment_cfg.get('path', None), \
                                    experiment_cfg.get('experiments', 1)

    device = cfg['training'].get('device', 'cpu')

    if torch.cuda.is_available() and device != 'cpu':
        torch.cuda.set_device(device)
        device = 'cuda:{}'.format(device)
    else:
        warnings.warn("Device not found or CUDA not available.")

    device = torch.device(device)

    if path is None:
        path = os.getcwd()
    else:
        os.chdir(path)
        os.makedirs(path, exist_ok=True)

    for image_index in range(experiments):
        torch.manual_seed(image_index)
        np.random.seed(image_index)

        experiment_path = os.path.join(path, 'exp_{}'.format(image_index))
        os.makedirs(experiment_path, exist_ok=True)

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
        pre_trained = model_cfg.get('pretrained', False)

        dataset_cfg = cfg['dataset']
        dataset_name = dataset_cfg['name']
        augmented_dataset = dataset_cfg.get('augment', False)
        images_to_attack_per_label = dataset_cfg.get('images_to_attack_'
                                                     'per_label', 50)
        dataset_path = dataset_cfg.get('path', None)

        is_imagenet = dataset_name.lower() == 'imagenet'

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
        optimizer_name, lr, momentum, weight_decay = optimizer_cfg.get(
            'optimizer',
            'sgd'), \
                                                     optimizer_cfg.get('lr',
                                                                       1e-1), \
                                                     optimizer_cfg.get(
                                                         'momentum',
                                                         0.9), \
                                                     optimizer_cfg.get(
                                                         'weight_decay', 0)

        os.makedirs(path, exist_ok=True)

        train_set, test_set, input_size, classes = \
            get_dataset(name=dataset_name,
                        model_name=None,
                        augmentation=augmented_dataset,
                        path=dataset_path)

        test_set = IndexedDataset(test_set)

        if not is_imagenet:
            train_set = IndexedDataset(train_set)
            trainloader = torch.utils.data.DataLoader(train_set,
                                                      batch_size=batch_size,
                                                      shuffle=True)

        testloader = torch.utils.data.DataLoader(test_set,
                                                 batch_size=batch_size,
                                                 shuffle=False)

        model = get_model(model_name,
                          image_size=input_size,
                          classes=classes + 1 if not is_imagenet else classes,
                          is_imagenet=is_imagenet,
                          pre_trained=pre_trained)

        model.to(device)

        if not is_imagenet:
            if os.path.exists(os.path.join(experiment_path,
                                           'model.pt')):
                log.info('Model loaded.')

                model.load_state_dict(torch.load(
                    os.path.join(experiment_path, 'model.pt'),
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

                    for _, (inputs, labels, image_index) in enumerate(
                            trainloader, 0):
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
                           os.path.join(experiment_path, 'model.pt'))

        model.eval()

        images = []
        labels = []
        indexes = []

        counter = defaultdict(int)

        with torch.no_grad():
            for img, y, image_index in tqdm(DataLoader(test_set,
                                                       batch_size=256,
                                                       shuffle=False),
                                            leave=False):
                img = img.to(device)

                _y = model(img)
                predictions = torch.argmax(_y, -1)

                for j, (im, p, gt) in enumerate(zip(img.cpu(),
                                                    predictions.cpu().numpy(),
                                                    y.numpy())):

                    if p == gt and counter[gt] < images_to_attack_per_label:
                        counter[gt] += 1
                        images.append(im)
                        indexes.append(image_index[j])
                        labels.append(gt)

        images = torch.stack(images, 0)
        labels = torch.tensor(labels)
        indexes = torch.tensor(indexes)

        dataset = TensorDataset(images, labels, indexes)

        atks = cfg.get('attacks', {})
        # atks = cfg.get('attacks', {})
        atks = OmegaConf.to_container(atks)

        for k, v in atks.items():

            v = get_default_attack_config(v)
            attack_factory = get_attack(v)

            log.info('Attack {}, Parameters {}'.format(k, v))

            attack_save_path = os.path.join(experiment_path, 'attacks',
                                            '{}.json'.format(k))

            attack_images_save_path = os.path.join(experiment_path, 'attacks',
                                                   '{}_images'.format(k))

            base_images_save_path = os.path.join(experiment_path, 'attacks',
                                                 'attacked_images')

            os.makedirs(attack_images_save_path, exist_ok=True)
            os.makedirs(base_images_save_path, exist_ok=True)

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

            if len(attack_results) - 2 != len(dataset):
                for img, y, image_index in tqdm(DataLoader(dataset,
                                                           batch_size=1,
                                                           shuffle=False),
                                                leave=False):

                    img = img.to(device)
                    probs = softmax(model(img), dim=1)[0].tolist()
                    image_index = str(image_index.item())

                    if image_index in attack_results:
                        d = attack_results[image_index]
                        continue
                    else:
                        d = {'label': y.item(),
                             # 'probs': probs,
                             'attacks': {}}

                    for offset in tqdm(range(classes), leave=False):
                        with HiddenPrints():
                            if offset > 0 and v.get('targeted', False):
                                # break
                                f = lambda x, label: (label + offset) % classes
                                attack.set_mode_targeted_by_function(f)
                                attack_label = f(img, y)
                                if attack_label == y:
                                    attack.set_mode_default()

                            else:
                                offset = 0
                                attack.set_mode_default()
                                attack_label = y

                        attack_label = attack_label.item()

                        if str(attack_label) not in d['attacks']:

                            start = time.time()
                            pert_images = attack(img, y)

                            end = time.time()
                            elapsed_time = end - start

                            iterations = -1
                            statistics = None

                            if getattr(attack, 'probs_statistics', None) \
                                    is not None:
                                statistics = attack.probs_statistics[0]
                            if getattr(attack, 'required_iterations', None) \
                                    is not None:
                                iterations = attack.required_iterations[0]

                            if v.get('save_images', False):
                                os.makedirs(attack_images_save_path,
                                            exist_ok=True)
                                f = plt.figure()
                                plt.imshow(
                                    np.moveaxis(pert_images.cpu().numpy()[0],
                                                0, -1))
                                plt.axis('off')
                                f.savefig((os.path.join(attack_images_save_path,
                                                        'p_{}_{}'.format(image_index, offset))), bbox_inches='tight')
                                plt.close(f)
                                f = plt.figure()
                                plt.imshow(
                                    np.moveaxis(img.cpu().numpy()[0], 0,
                                                -1))
                                plt.axis('off')
                                f.savefig((os.path.join(base_images_save_path,
                                                        'p_{}_{}'.format(image_index, offset))), bbox_inches='tight')
                                plt.close(f)

                            final_prob = softmax(model(pert_images), dim=1)[0]
                            final_prob = final_prob.tolist()

                            diff = (pert_images - img).view(-1)

                            norms = {norm_t: torch.linalg.norm(diff,
                                                               ord=norm_t).item() / 3

                                     for norm_t in [0, 2, float('inf')]}

                            res = {
                                # 'attacked_label': attack_label,
                                'time': elapsed_time,
                                # 'probs': final_prob,
                                'prediction': np.argmax(final_prob),
                                'norms': norms,
                                'iterations': iterations,
                                'statistics': statistics}

                            d['attacks'][str(attack_label)] = res

                    attack_results[image_index] = d

                    with open(attack_save_path, 'w') \
                            as json_file:
                        json.dump(attack_results, json_file,
                                  indent=4, cls=NpEncoder)

            with open(os.path.join(experiment_path, attack_save_path), 'r') \
                    as json_file:
                attack_results = json.load(json_file)

                correctly_attacked, mean_time, std_time, \
                mean_iterations, std_iterations, mean_zeron, std_zeron = \
                    calculate_scores(attack_results)

                log.info('\t\tCorrectly attacked: {}/{} ({})'
                         .format(int(correctly_attacked * len(dataset)),
                                 len(dataset), correctly_attacked))

                log.info('\t\tIterations required per image: {}(+-{})'
                         .format(mean_iterations, std_iterations))

                log.info('\t\tZero norm per image: {}(+-{})'
                         .format(mean_zeron, std_zeron))

                log.info('\t\tTime required per image: {}(+-{})'
                         .format(mean_time, std_time))

        return model


if __name__ == "__main__":
    my_app()
