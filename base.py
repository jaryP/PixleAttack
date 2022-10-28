from abc import ABC, abstractmethod
from omegaconf import DictConfig
from torchvision.models import resnet18, resnet34, resnet50, alexnet, vgg11, \
    vgg16
import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset


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


def get_model(name, image_size, classes, pre_trained=False,
              is_imagenet=False):
    name = name.lower()

    pre_trained = pre_trained or is_imagenet

    if name == 'alexnet':
        if is_imagenet:
            return alexnet(pretrained=True)
        else:
            return AlexNet(image_size[0], classes)
    if 'vgg' in name:
        if name == 'vgg11':
            return vgg11(num_classes=classes, pretrained=pre_trained)
        elif name == 'vgg16':
            return vgg16(num_classes=classes, pretrained=pre_trained)
        else:
            assert False
    elif 'resnet' in name:
        if name == 'resnet20':
            return resnet20(classes)
        elif name == 'resnet18':
            return resnet18(num_classes=classes, pretrained=pre_trained)
        elif name == 'resnet34':
            return resnet34(num_classes=classes, pretrained=pre_trained)
        elif name == 'resnet50':
            return resnet50(num_classes=classes, pretrained=pre_trained)
        else:
            assert False
    else:
        assert False


class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    # url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    url = 'https://drive.google.com/u/1/uc?id=1GDr1OkoXdhaXWGA8S3MAq3a522Tak-nx&export=download'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
