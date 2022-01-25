#!/usr/bin/env bash

DATASET=$1
MODEL=$2
DEVICE=$3

case $DATASET in
  cifar10)
    case $MODEL in
    vgg11)
      python main.py +dataset=cifar10 experiment=base +model=vgg11 optimizer=sgd_momentum +training=cifar10 hydra.run.dir='./outputs_new/cifar10/vgg11' +attacks=cifar10 training.device="$DEVICE"
    ;;
    resnet18)
      python main.py +dataset=cifar10 experiment=base +model=resnet18 optimizer=sgd_momentum +training=cifar10 hydra.run.dir='./outputs_new/cifar10/resnet18' +attacks=cifar10 training.device="$DEVICE"
    ;;
    *)
      echo -n "Unrecognized model"
    esac
  ;;
  tinyimagenet)
    case $MODEL in
      vgg16)
        python main.py +dataset=tinyimagenet experiment=base +model=vgg16 optimizer=sgd_momentum +training=tinyimagenet hydra.run.dir='./outputs_new/tinyimagenet/vgg16' +attacks=imagenet training.device="$DEVICE"
      ;;
      resnet50)
        python main.py +dataset=tinyimagenet experiment=base +model=resnet50 optimizer=sgd_momentum +training=tinyimagenet hydra.run.dir='./outputs_new/tinyimagenet/resnet50/' +attacks=imagenet training.device="$DEVICE"
      ;;
      *)
      echo -n "Unrecognized model"
    esac
  ;;
  imagenet)
    case $MODEL in
      vgg16)
        python main.py +dataset=imagenet experiment=base +model=vgg16 optimizer=sgd_momentum +training=tinyimagenet hydra.run.dir='./outputs_new/imagenet/vgg16/' +attacks=imagenet training.device="$DEVICE"
      ;;
      resnet50)
        python main.py +dataset=imagenet experiment=base +model=resnet50 optimizer=sgd_momentum +training=tinyimagenet hydra.run.dir='./outputs_new/imagenet/resnet50/' +attacks=imagenet training.device="$DEVICE"
      ;;
      *)
      echo -n "Unrecognized model"
    esac
  ;;
  *)
  echo -n "Unrecognized dataset"
esac
