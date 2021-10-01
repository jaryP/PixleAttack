from abc import ABC, abstractmethod
from omegaconf import DictConfig
from torchvision.models import resnet18

from models.alexnet import AlexNet
from models.resnet import resnet20


class Training(ABC):
    def __init__(self, config: DictConfig):
        self.config = config

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def load(self, path, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, path, *args, **kwargs):
        pass


def get_model(name, image_size, classes):
    name = name.lower()
    if name == 'alexnet':
        return AlexNet(image_size[0], classes)
    elif 'resnet' in name:
        if name == 'resnet20':
            return resnet20(classes)
        elif name == 'resnet18':
            return resnet18(num_classes=classes)
        else:
            assert False
    else:
        assert False

# def get_branch_model(name, image_size, classes, equalize_embedding=True):
#     name = name.lower()
#     if name == 'alexnet':
#         model = AlexNet(image_size[0])
#     elif name == 'resnet20':
#         model = branch_resnet20()
#         # return AlexNet(input_channels), AlexNetClassifier(classes)
#     # elif 'resnet' in name:
#     #     if name == 'resnet20':
#     #         model
#     #         return resnet20(None), ResnetClassifier(classes)
#     #     else:
#     #         assert False
#     else:
#         assert False
#
#     classifiers = get_intermediate_classifiers(model,
#                                                image_size,
#                                                classes,
#                                                equalize_embedding=equalize_embedding)
#
#     return model, classifiers