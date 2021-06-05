import os
from collections import OrderedDict

import torch.utils.data as data
from torch.utils.data.dataset import T_co

import datasets.utils as utils


def split_dataset(full_dataset, split_ratio):
    train_size = int(split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = data.random_split(full_dataset, [train_size, val_size])
    print("Val dataset length: {0}".format(len(val_dataset)))
    return val_dataset, train_dataset


# TODO check again kitti dataset
class _KittiTrain(data.Dataset):
    def __init__(self, root_dir, train_folder, train_folder_labeled, class_encoding, data_transform=None,
                 label_transform=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.train_folder = train_folder
        self.train_folder_labeled = train_folder_labeled
        self.class_encoding = class_encoding
        self.data_transform = data_transform
        self.label_transform = label_transform
        self.train_data = utils.get_files(os.path.join(root_dir, self.train_folder))
        self.train_labels = utils.get_files(os.path.join(root_dir, self.train_folder_labeled))

    def __getitem__(self, index) -> T_co:
        data_path, label_path = self.train_data[index], self.train_labels[index]
        img = utils.get_data_transformed(data_path, self.data_transform)
        label = utils.get_data_transformed(label_path, self.label_transform)
        label = utils.get_target_mask(label, self.class_encoding)
        return img, label

    def __len__(self):
        return len(self.train_data)


class _KittiTest(data.Dataset):
    def __init__(self, root_dir, test_folder, class_encoding=None, data_transform=None, label_transform=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.test_folder = test_folder
        self.class_encoding = class_encoding
        self.data_transform = data_transform
        self.label_transform = label_transform
        self.test_data = utils.get_files(os.path.join(root_dir, self.test_folder))

    def __getitem__(self, index) -> T_co:
        data_path = self.test_data[index]
        img = utils.get_data_transformed(data_path, self.data_transform)
        return img

    def __len__(self):
        return len(self.test_data)


class Kitti(data.Dataset):
    # Training dataset folders
    train_folder = 'training/image_2'
    train_folder_labeled = 'training/gt_image_2'
    # Test dataset folders
    test_folder = 'testing/image_2'

    class_encoding = OrderedDict([
        ('Road', (255, 0, 255)),
        ('Void', (0, 0, 0)),
        ('Background', (255, 0, 0)),
    ])

    def __init__(self, root_dir, mode='train', data_transform=None, label_transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.data_transform = data_transform
        self.label_transform = label_transform
        full_train_dataset = _KittiTrain(root_dir=root_dir, train_folder=self.train_folder,
                                         class_encoding=self.class_encoding,
                                         train_folder_labeled=self.train_folder_labeled,
                                         data_transform=data_transform, label_transform=label_transform)
        self.test_dataset = _KittiTest(root_dir=root_dir, test_folder=self.test_folder,
                                       class_encoding=self.class_encoding,
                                       data_transform=data_transform, label_transform=label_transform)
        self.val_dataset, self.train_dataset = split_dataset(full_train_dataset, 0.8)

    def __getitem__(self, index) -> T_co:
        if self.mode.lower() == 'train':
            return self.train_dataset.__getitem__(index)
        elif self.mode.lower() == 'val':
            return self.val_dataset.__getitem__(index)
        elif self.mode.lower() == 'test':
            return self.test_dataset.__getitem__(index)
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train and test.")

    def __len__(self):
        if self.mode.lower() == 'train':
            return self.train_dataset.__len__()
        elif self.mode.lower() == 'val':
            return self.val_dataset.__len__()
        elif self.mode.lower() == 'test':
            return self.test_dataset.__len__()
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train and test.")
