import logging
import os
import warnings

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# from adv_training.ed_training.training import pixel_regions_training
# from attacks import get_attack, get_attacked_dataset
# from evaluators import accuracy_score, ece_score
# # from loss_landscape.plots.embedding_plots import flat_embedding_plots, \
# #     flat_embedding_cosine_similarity_plot
# from methods.trainers import branch_dirichlet_trainer, shift_trainer
# from utils import get_dataset
# import matplotlib.pyplot as plt
from training import standard_training


@hydra.main(config_path="configs",
            config_name="config")
def my_app(cfg: DictConfig) -> None:
    # from methods.trainers import ensemble_trainer, wrapper_ensemble
    log = logging.getLogger(__name__)

    log.info(OmegaConf.to_yaml(cfg))
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

    for i in range(experiments):
        torch.manual_seed(i)
        np.random.seed(i)

        experiment_path = os.path.join(path, 'exp_{}'.format(i))
        os.makedirs(experiment_path, exist_ok=True)

        standard_training(cfg, experiment_path, device=device)

        # if method_name == 'naive':
        #     ff = ensemble_trainer(cfg, experiment_path, device=device)
        # elif method_name == 'dropout':
        #     ff = wrapper_ensemble(cfg, experiment_path,
        #                           ensemble_type='dropout',
        #                           device=device)
        # elif method_name == 'batch_ensemble':
        #     ff = wrapper_ensemble(cfg, experiment_path,
        #                           ensemble_type='batch_ensemble',
        #                           device=device)
        # elif method_name == 'dirichlet':
        #     ff = branch_dirichlet_trainer(cfg, device=device,
        #                                   save_path=experiment_path,
        #                                   logits_training=False)
        # elif method_name == 'dirichlet_logits':
        #     ff = branch_dirichlet_trainer(cfg, logits_training=True,
        #                                   device=device,
        #                                   save_path=experiment_path)
        # elif method_name == 'dirichlet_transformations':
        #     ff = shift_trainer(cfg, logits_training=True, device=device,
        #                        save_path=experiment_path)
        # else:
        #     assert False
        #
        # dataset_cfg = cfg['dataset']
        # dataset_name = dataset_cfg['name']
        # augmented_dataset = dataset_cfg.get('augment', False)
        # training_cfg = cfg['training']
        # batch_size, device = training_cfg['batch_size'], \
        #                      training_cfg.get('device', 'cpu')
        #
        # train_set, test_set, input_size, classes = \
        #     get_dataset(name=dataset_name,
        #                 model_name=None,
        #                 augmentation=augmented_dataset)
        #
        # testloader = torch.utils.data.DataLoader(test_set,
        #                                          batch_size=batch_size,
        #                                          shuffle=False)
        #
        # atks = cfg.get('attacks', {})
        #
        # for k, v in atks.items():
        #     log.info('Attack {}: {}'.format(k, v))
        #
        #     attack = get_attack(v)
        #     dataset = get_attacked_dataset(attack, testloader, ff, device)
        #
        #     dataset = DataLoader(dataset, batch_size=batch_size)
        #
        #     score, _, _ = accuracy_score(model=ff,
        #                                  dataset=dataset,
        #                                  device=device)
        #     log.info('Score: {}'.format(score))
        #
        #     ece, _, _, _ = ece_score(model=ff,
        #                              dataset=dataset,
        #                              device=device)
        #
        #     log.info('ECE: {}'.format(ece))


if __name__ == "__main__":
    my_app()
