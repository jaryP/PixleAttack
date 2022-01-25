#!/usr/bin/env bash

DATASET=$1
MODEL=$2
DEVICE=$3

case $DATASET in
  cifar10)
    case $MODEL in
    vgg11)
      python main.py +dataset=cifar10 experiment=base +model=vgg11 optimizer=sgd_momentum +training=cifar10 hydra.run.dir='./outputs_new/cifar10/vgg11' +attacks=cifar10_targeted training.device="$DEVICE" dataset.images_to_attack_per_label=20 hydra.job_logging.handlers.file.filename=targeted.log
    ;;
    resnet18)
      python main.py +dataset=cifar10 experiment=base +model=resnet18 optimizer=sgd_momentum +training=cifar10 hydra.run.dir='./outputs_new/cifar10/resnet18' +attacks=cifar10_targeted training.device="$DEVICE" dataset.images_to_attack_per_label=20 hydra.job_logging.handlers.file.filename=targeted.log
    ;;
    *)
      echo -n "Unrecognized model"
    esac
  ;;
  *)
  echo -n "Unrecognized dataset"
esac
