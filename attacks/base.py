import logging
import os
import sys

import torch
from autoattack import AutoAttack
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchattacks import FGSM, OnePixel, Square, PGD, SparseFool
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

from attacks.psa import PatchesSwap
from attacks.scratch import ScratchThat


class IndexedDataset(Dataset):
    def __init__(self, dataset: VisionDataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]

        return data, target, index

    def __len__(self):
        return len(self.dataset)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout



def get_default_attack_config(cfg: DictConfig):
    name = cfg['name']
    name = name.lower()

    if name == 'fgsm':
        d = {'eps': 0.007}
    elif name == 'pgd':
        d = {'eps': 0.3,
             'alpha': 0.00784313725490196,
             'steps': 40,
             'random_start': True}
    elif name == 'onepixel':
        d = {'pixels': 1,
             'popsize': 400,
             'steps': 75,
             'inf_batch': 128}
    elif name == 'autoattack':
        d = {'norm': 'Linf',
             'eps': 0.3,
             'version': 'standard',
             'n_classes': 10,
             'seed': None}
    elif name == 'square':
        d = {'norm': 'Linf',
             'eps': None,
             'n_queries': 5000,
             'n_restarts': 1,
             'loss': 'margin',
             'p_init': 0.8,
             'resc_schedule': True,
             'seed': None}
    elif name == 'sparsefool':
        d = {'steps': 20,
             'lam': 3,
             'overshoot': 0.02,}
    elif name == 'patches':
        d = {'eps': 0.1,
             'patches': 500,
             'p1_x_dimensions': (2, 2),
             'p1_y_dimensions': (12, 12),
             'patches_to_use': 100,
             'train_selection': False,
             'select_subset': False,
             'fixed_selection_quantile': 0.8,
             'seed': None}
    elif name == 'patchesswap':
        d = {'population': 1,
             'same_size': True,
             'max_iterations': 100,
             'restarts': 0,
             'algorithm': 'de',
             'p1_x_dimensions': (0, 10),
             'p1_y_dimensions': (0, 10),
             'p2_x_dimensions': None,
             'p2_y_dimensions': None,
             'restart_callback': True,
             'update_each_iteration': False}

    elif name == 'scratch_that':
        d = {'population': 1,
             'mutation_rate': (0.5, 1),
             'crossover_rate': 0.7,
             'scratch_type': 'line',
             'n_scratches': 1,
             'max_iterations': 100
             }
    else:
        assert False

    d.update(cfg)

    return d


def get_attack(cfg: DictConfig):
    name = cfg['name']
    name = name.lower()

    cfg = get_default_attack_config(cfg)

    def method(model):
        if name == 'fgsm':
            return FGSM(model,
                        eps=cfg.get('eps', 0.007))
        elif name == 'pgd':
            eps = cfg.get('eps', 0.3)
            return PGD(model,
                       eps=eps,
                       alpha=0.00784313725490196,
                       steps=cfg.get('steps', 40),
                       random_start=cfg.get('random_start', True))
        elif name == 'onepixel':
            return OnePixel(model,
                            pixels=cfg.get('pixels', 1),
                            steps=cfg.get('steps', 75),
                            popsize=cfg.get('popsize', 400),
                            inf_batch=cfg.get('inf_batch', 128))
        elif name == 'autoattack':
            return AutoAttack(model,
                              norm=cfg.get('norm', 'Linf'),
                              eps=cfg.get('eps', 0.3),
                              version=cfg.get('version', 'standard'),
                              n_classes=cfg.get('n_classes', 10),
                              seed=cfg.get('seed', None),
                              verbose=False)
        elif name == 'square':
            return Square(model,
                          norm=cfg.get('norm', 'Linf'),
                          eps=cfg.get('eps', None),
                          n_queries=cfg.get('n_queries', 5000),
                          n_restarts=cfg.get('n_restarts', 1),
                          p_init=cfg.get('p_init', 0.8),
                          loss=cfg.get('loss', 'margin'),
                          resc_schedule=cfg.get('resc_schedule', True),
                          seed=cfg.get('seed', 0),
                          verbose=False)
        elif name == 'sparsefool':
            return SparseFool(model,
                          steps=cfg.get('steps', 20),
                          lam=cfg.get('lam', 3),
                          overshoot=cfg.get('overshoot', 0.02))
        # elif name == 'patches':
        #     return RandomPatches(model=model,
        #                          patches=cfg.get('patches', 500),
        #                          min_dimensions=cfg.get('p1_x_dimensions',
        #                                                 (2, 2)),
        #                          max_dimensions=cfg.get('p1_y_dimensions',
        #                                                 (12, 12)),
        #                          patches_to_use=cfg.get('patches_to_use', 100),
        #                          train_selection=
        #                          cfg.get('train_selection', False),
        #                          select_subset=
        #                          cfg.get('select_subset', False),
        #                          fixed_selection_percentile=
        #                          cfg.get('fixed_selection_quantile', 0.8),
        #                          eps=cfg.get('eps', 0.01))
        elif name == 'patchesswap':
            return PatchesSwap(model,
                               population=cfg.get('population', 1),
                               same_size=cfg.get('same_size', False),
                               restarts=cfg.get('restarts', 0),
                               algorithm=cfg.get('algorithm', 'de'),
                               p1_x_dimensions=cfg.get('p1_x_dimensions',
                                                       (0, 10)),
                               p1_y_dimensions=cfg.get('p1_y_dimensions',
                                                       (0, 10)),
                               p2_x_dimensions=cfg.get('p2_x_dimensions',
                                                       None),
                               p2_y_dimensions=cfg.get('p2_y_dimensions',
                                                       None),
                               restart_callback=cfg.get('restart_callback',
                                                        True),
                               update_each_iteration=cfg.get('update_each_iteration',
                                                        False)
                               )
        elif name == 'scratch_that':
            return ScratchThat(model,
                               population=cfg.get('population', 1),
                               mutation_rate=cfg.get('mutation_rate', (0.5, 1)),
                               crossover_rate=cfg.get('crossover_rate', 0.7),
                               scratch_type=cfg.get('scratch_type', 'line'),
                               n_scratches=cfg.get('n_scratches', 1),
                               max_iterations=cfg.get('max_iterations', 1000))
        else:
            assert False

    return method


def get_attacked_dataset(attack, dataset: DataLoader, forward: nn.Module,
                         device):
    log = logging.getLogger(__name__)

    images, labels = [], []
    bns = []

    attack = attack(forward)
    log.info('Attacking the dataset {}'.format(attack))

    for i, (image, label) in tqdm(enumerate(dataset),
                                  total=len(dataset),
                                  leave=False):
        image = image.to(device)
        label = label.to(device)

        adv_images = attack(image, label)

        adv_images = adv_images.cpu()
        label = label.cpu()

        images.append(adv_images)
        labels.append(label)

    images = torch.cat(images, 0)
    labels = torch.cat(labels, 0)

    d = TensorDataset(images, labels)
    # d = DataLoader(d, batch_size=dataset.batch_size)

    return d
