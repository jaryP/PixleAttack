#!/usr/bin/env bash
ARG1=${1:-0}

python main.py +dataset=svhn experiment=base +model=resnet20 optimizer=sgd_momentum +training=svhn hydra.run.dir='./outputs/svhn/' +attacks=ps training.device="$ARG1"
python main.py +dataset=cifar10 experiment=base +model=resnet20 optimizer=sgd_momentum +training=cifar10 hydra.run.dir='./outputs/cifar10/' +attacks=ps training.device="$ARG1"
python main.py +dataset=tinyimagenet experiment=base +model=resnet18 optimizer=sgd_momentum +training=tinyimagenet hydra.run.dir='./outputs/tinyimagenet/' +attacks=ps training.device="$ARG1"
