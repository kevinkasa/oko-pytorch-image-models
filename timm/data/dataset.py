""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2019, Ross Wightman
"""
import io
import logging
from typing import Optional
import json
import os
from collections import defaultdict

import torch
import numpy as np
import torch.utils.data as data
from torchvision.datasets.folder import ImageFolder, default_loader
from PIL import Image

from .readers import create_reader

_logger = logging.getLogger(__name__)

_ERROR_RETRY = 50


class Plantnet(ImageFolder):
    def __init__(self, root, split, **kwargs):
        self.root = root
        self.split = split
        super().__init__(self.split_folder, **kwargs)

    @property
    def split_folder(self):
        return os.path.join(self.root, self.split)

class PlantnetOKO(ImageFolder):
    def __init__(self, root, split, transform=None, target_transform=None, loader=default_loader, **kwargs):
        self.root = root
        self.split = split

        # Call parent class initialization
        super().__init__(self.split_folder, transform=transform, target_transform=target_transform, loader=loader,
                         **kwargs)

        # Create samples by class dictionary for efficient sampling
        self.samples_by_class = defaultdict(list)
        for path, label in self.samples:
            self.samples_by_class[label].append(path)

        # Precompute other classes for each class
        self.other_classes = {
            class_label: np.array([l for l in self.samples_by_class.keys() if l != class_label])
            for class_label in self.samples_by_class.keys()
        }

    @property
    def split_folder(self):
        return os.path.join(self.root, self.split)

    def __getitem__(self, index):
        """
        Get a sample with OKO functionality - returns a concatenation of:
        1. Current image
        2. Random image from the same class
        3. Random image from a different class

        Args:
            index (int): Index of the sample

        Returns:
            tuple: (concatenated_images, target) where concatenated_images contains
                   the current image, a same-class image, and a different-class image
        """
        # Get the current sample
        cur_path, cur_target = self.samples[index]

        # Select a random sample from the same class
        same_class_samples = self.samples_by_class[cur_target]
        if len(same_class_samples) > 1:
            # Exclude the current sample to avoid selecting it
            available_samples = [p for p in same_class_samples if p != cur_path]
            same_class_path = np.random.choice(available_samples)
        else:
            # If only one sample, use the same sample
            same_class_path = cur_path

        # Select a random sample from a different class
        different_class_label = np.random.choice(self.other_classes[cur_target])
        different_class_samples = self.samples_by_class[different_class_label]
        different_class_path = np.random.choice(different_class_samples)

        # Load and transform the current image
        cur_sample = self.loader(cur_path)
        if self.transform is not None:
            cur_sample = self.transform(cur_sample)
        if self.target_transform is not None:
            cur_target = self.target_transform(cur_target)

        # Initialize the images list with the current sample
        images = [cur_sample]

        # Load and transform the same-class and different-class images
        for path in [same_class_path, different_class_path]:
            image = self.loader(path)
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        # Concatenate the images along the first dimension (assuming channel-first format)
        return np.concatenate(images, axis=0), cur_target

class INatOKODataset(ImageFolder):
    def __init__(self, root, split='train', year=2018, category='name', transform=None, k=1, loader=default_loader,
                 target_transform=None):
        # Initialize the ImageFolder parent class
        # super().__init__(os.path.join(root, f'train_val{year}'), )

        self.root = root
        self.split = split
        self.year = year
        self.category = category
        self.transform = transform
        self.k = k
        self.loader = loader
        self.target_transform = target_transform

        path_json = os.path.join(root, f'{split}{year}.json')
        with open(path_json) as json_file:
            self.data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            self.data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")
        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        self.targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = self.data_catg[int(elem['category_id'])][category]
            if king not in self.targeter.keys():
                self.targeter[king] = indexer
                indexer += 1

        self.nb_classes = len(self.targeter)

        self.samples = []
        self.samples_by_class = {}

        for elem in self.data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[1], cut[2], cut[3])
            categors = self.data_catg[target_current]
            target_current_true = self.targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

            if target_current_true not in self.samples_by_class:
                self.samples_by_class[target_current_true] = []
            self.samples_by_class[target_current_true].append(path_current)

        # Precompute other classes for each class
        self.other_classes = {
            class_label: np.array([l for l in self.samples_by_class.keys() if l != class_label])
            for class_label in self.samples_by_class.keys()
        }
        # Group samples by class
        self.samples_by_class = defaultdict(list)
        for path, label in self.samples:
            self.samples_by_class[label].append(path)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # current sample
        cur_path, cur_target = self.samples[index]

        # Select a random sample from the same class
        same_class_samples = self.samples_by_class[cur_target]
        same_class_path = np.random.choice(same_class_samples)

        # Select a random sample from a different class for the odd-k sample
        different_class_label = np.random.choice(self.other_classes[cur_target])

        different_class_samples = self.samples_by_class[different_class_label]
        different_class_path = np.random.choice(different_class_samples)

        # path, target = self.samples[index]
        cur_sample = self.loader(cur_path)
        if self.transform is not None:
            cur_sample = self.transform(cur_sample)
        if self.target_transform is not None:
            cur_target = self.target_transform(cur_target)
        images = [cur_sample]
        # targets = [cur_target]
        for path in [same_class_path, different_class_path]:
            # Load image using the default loader
            image = self.loader(path)

            # Apply transformation if specified
            if self.transform is not None:
                image = self.transform(image)

            images.append(image)
            # targets.append(target)
        return np.concatenate(images, axis=0), cur_target

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader, **kwargs):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        self.root = root
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')  # TODO: load test set also
        with open(path_json) as json_file:
            data = json.load(json_file)
        print("train" if train else "val")
        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[1], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))
        print(f'num samples: {len(self.samples)}')


class ImageDataset(data.Dataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            class_map=None,
            load_bytes=False,
            img_mode='RGB',
            transform=None,
            target_transform=None,
    ):
        if reader is None or isinstance(reader, str):
            reader = create_reader(
                reader or '',
                root=root,
                split=split,
                class_map=class_map
            )
        self.reader = reader
        self.load_bytes = load_bytes
        self.img_mode = img_mode
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
        img, target = self.reader[index]

        try:
            img = img.read() if self.load_bytes else Image.open(img)
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.reader.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.reader))
            else:
                raise e
        self._consecutive_errors = 0

        if self.img_mode and not self.load_bytes:
            img = img.convert(self.img_mode)
        if self.transform is not None:
            img = self.transform(img)

        if target is None:
            target = -1
        elif self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.reader)

    def filename(self, index, basename=False, absolute=False):
        return self.reader.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)


class IterableImageDataset(data.IterableDataset):

    def __init__(
            self,
            root,
            reader=None,
            split='train',
            is_training=False,
            batch_size=None,
            seed=42,
            repeats=0,
            download=False,
            transform=None,
            target_transform=None,
    ):
        assert reader is not None
        if isinstance(reader, str):
            self.reader = create_reader(
                reader,
                root=root,
                split=split,
                is_training=is_training,
                batch_size=batch_size,
                seed=seed,
                repeats=repeats,
                download=download,
            )
        else:
            self.reader = reader
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

    def __iter__(self):
        for img, target in self.reader:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            yield img, target

    def __len__(self):
        if hasattr(self.reader, '__len__'):
            return len(self.reader)
        else:
            return 0

    def set_epoch(self, count):
        # TFDS and WDS need external epoch count for deterministic cross process shuffle
        if hasattr(self.reader, 'set_epoch'):
            self.reader.set_epoch(count)

    def set_loader_cfg(
            self,
            num_workers: Optional[int] = None,
    ):
        # TFDS and WDS readers need # workers for correct # samples estimate before loader processes created
        if hasattr(self.reader, 'set_loader_cfg'):
            self.reader.set_loader_cfg(num_workers=num_workers)

    def filename(self, index, basename=False, absolute=False):
        assert False, 'Filename lookup by index not supported, use filenames().'

    def filenames(self, basename=False, absolute=False):
        return self.reader.filenames(basename, absolute)


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)
