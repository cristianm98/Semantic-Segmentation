import os
from collections import OrderedDict

import torch.utils.data as data
from torch.utils.data.dataset import T_co

import datasets.utils as utils


def get_paths(root_dir, mode):
    data_dir = os.path.join(root_dir, mode, "data")
    target_dir = os.path.join(root_dir, mode, "target")
    return data_dir, target_dir


def split_dataset(full_dataset, split_ratio):
    train_size = int(split_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = data.random_split(full_dataset, [train_size, test_size])
    val_size = int(test_size / 2)
    test_size -= val_size
    val_dataset, test_dataset = data.random_split(test_dataset, [val_size, test_size])
    return train_dataset, val_dataset, test_dataset


class _InfraRedHelper(data.Dataset):
    def __init__(self, root_dir, day: bool, class_encoding, data_transform=None, label_transform=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.day = day
        self.data_transform = data_transform
        self.label_transform = label_transform
        if day:
            data_dir, target_dir = get_paths(root_dir, "day")
        else:
            data_dir, target_dir = get_paths(root_dir, "night")
        self.all_data = utils.get_files(data_dir)
        self.all_targets = utils.get_files(target_dir)
        self.class_encoding = class_encoding

    def __getitem__(self, index) -> T_co:
        data_path, label_path = self.all_data[index], self.all_targets[index]
        img = utils.get_data_transformed(data_path, self.data_transform)
        label = utils.get_data_transformed(label_path, self.label_transform)
        label = utils.get_target_mask(label, self.class_encoding)
        return img, label

    def __len__(self):
        return len(self.all_data)


class _InfraRed(data.Dataset):
    class_encoding = OrderedDict([
        ('Background', (0, 0, 0)),
        ('Road', (128, 0, 0))
    ])

    def __init__(self, root_dir, use_day=True, mode='train', data_transform=None, label_transform=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.use_day = use_day
        self.mode = mode
        self.data_transform = data_transform
        self.label_transform = label_transform
        full_dataset = _InfraRedHelper(root_dir, use_day, self.class_encoding, data_transform, label_transform)
        self.train_dataset, self.val_dataset, self.test_dataset = split_dataset(full_dataset, 0.7)

    def __getitem__(self, index) -> T_co:
        if self.mode.lower() == 'train':
            return self.train_dataset.__getitem__(index)
        elif self.mode.lower() == 'val':
            return self.val_dataset.__getitem__(index)
        elif self.mode.lower() == 'test':
            return self.test_dataset.__getitem__(index)
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val and test.")

    def __len__(self):
        if self.mode.lower() == 'train':
            return len(self.train_dataset)
        elif self.mode.lower() == 'val':
            return len(self.val_dataset)
        elif self.mode.lower() == 'test':
            return len(self.test_dataset)
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val and test.")


class InfraRedDay(_InfraRed):
    def __init__(self, root_dir, mode='train', data_transform=None, label_transform=None):
        super(InfraRedDay, self).__init__(root_dir=root_dir, use_day=True, mode=mode, data_transform=data_transform,
                                          label_transform=label_transform)


class InfraRedNight(_InfraRed):
    def __init__(self, root_dir, mode='train', data_transform=None, label_transform=None):
        super(InfraRedNight, self).__init__(root_dir=root_dir, use_day=False, mode=mode, data_transform=data_transform,
                                            label_transform=label_transform)
