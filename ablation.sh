#!/usr/bin/env bash

DEVICE=$1

python main.py +dataset=cifar10 experiment=base +model=vgg11 optimizer=sgd_momentum +training=cifar10 hydra.run.dir='./outputs_new/cifar10/vgg11' hydra.job_logging.handlers.file.filename=ablation.log +attacks=ablation_iterations training.device="$DEVICE"
python main.py +dataset=cifar10 experiment=base +model=vgg11 optimizer=sgd_momentum +training=cifar10 hydra.run.dir='./outputs_new/cifar10/vgg11' hydra.job_logging.handlers.file.filename=ablation_dimensions.log +attacks=ablation_dimensions training.device="$DEVICE"
python main.py +dataset=cifar10 experiment=base +model=vgg11 optimizer=sgd_momentum +training=cifar10 hydra.run.dir='./outputs_new/cifar10/vgg11' hydra.job_logging.handlers.file.filename=ablation_mapping.log +attacks=ablation_mapping training.device="$DEVICE"
