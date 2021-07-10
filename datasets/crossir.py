from collections import OrderedDict

import os
import torch.utils.data as data
from torch.utils.data.dataset import T_co

import datasets.utils as utils

test_folder = 'images'


class CrossirTest(data.Dataset):
    class_encoding = OrderedDict([
        ('Background', (0, 0, 0)),
        ('Road', (128, 0, 0))
    ])

    def __init__(self, root_dir, mode='test', data_transform=None, label_transform=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.data_transform = data_transform
        self.label_transform = label_transform
        test_dir = os.path.join(root_dir, test_folder)
        self.test_data = utils.get_files(test_dir)

    def __getitem__(self, index) -> T_co:
        data_path = self.test_data[index]
        img = utils.get_data_transformed(data_path, self.data_transform)
        return img, data_path

    def __len__(self):
        return len(self.test_data)
