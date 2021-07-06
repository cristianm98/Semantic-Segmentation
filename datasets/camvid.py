import os
from collections import OrderedDict

import torch.utils.data as data
from torch.utils.data.dataset import T_co

import datasets.utils as utils


class CamVid(data.Dataset):
    # Training dataset folders
    train_folder = 'train'
    train_folder_labeled = 'train_labels'
    # Validation dataset folders
    val_folder = 'val'
    val_folder_labeled = 'val_labels'
    # Test dataset folders
    test_folder = 'test'
    test_folder_labeled = 'test_labels'
    # File containing color encoding for each class
    class_labels_file = 'class_dict.csv'

    class_encoding = OrderedDict([
        # ('Animal', (64, 128, 64)),
        # ('Archway', (192, 0, 128)),
        # ('Bicyclist', (0, 128, 92)),
        # ('Bridge', (0, 128, 64)),
        # ('Building', (128, 0, 0)),
        # ('Car', (64, 0, 128)),
        # ('CartLuggagePram', (64, 0, 192)),
        # ('Child', (192, 128, 64)),
        # ('Column_Pole', (192, 192, 128)),
        # ('Fence', (64, 64, 128)),
        # ('LaneMkgsDriv', (128, 0, 192)),
        # ('LaneMkgsNonDriv', (192, 0, 64)),
        # ('Misc_Text', (128, 128, 64)),
        # ('MotorcycleScooter', (192, 0, 192)),
        # ('OtherMoving', (128, 64, 64)),
        # ('ParkingBlock', (64, 192, 128)),
        # ('Pedestrian', (64, 64, 0)),
        ('Road', (128, 64, 128)),
        # ('RoadShoulder', (128, 128, 192)),
        # ('Sidewalk', (0, 0, 192)),
        # ('SignSymbol', (192, 128, 128)),
        # ('Sky', (128, 128, 128)),
        # ('SUVPickupTruck', (64, 128, 192)),
        # ('TrafficCone', (0, 0, 64)),
        # ('TrafficLight', (0, 64, 64)),
        # ('Train', (192, 64, 128)),
        # ('Tree', (128, 128, 0)),
        # ('Truck_Bus', (192, 128, 192)),
        # ('Tunnel', (64, 0, 64)),
        # ('VegetationMisc', (192, 192, 0)),
        # ('Void', (0, 0, 0)),
        # ('Wall', (64, 192, 0))
    ])

    def __init__(self, root_dir, mode='train', data_transform=None, label_transform=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.data_transform = data_transform
        self.label_transform = label_transform

        if self.mode.lower() == 'train':
            train_dir = os.path.join(root_dir, self.train_folder)
            train_dir_labeled = os.path.join(root_dir, self.train_folder_labeled)
            self.train_data = utils.get_files(train_dir)
            self.train_labels = utils.get_files(train_dir_labeled)
        elif self.mode.lower() == 'val':
            val_dir = os.path.join(root_dir, self.val_folder)
            val_dir_labeled = os.path.join(root_dir, self.val_folder_labeled)
            self.val_data = utils.get_files(val_dir)
            self.val_labels = utils.get_files(val_dir_labeled)
        elif self.mode.lower() == 'test':
            test_dir = os.path.join(root_dir, self.test_folder)
            test_dir_labeled = os.path.join(root_dir, self.test_folder_labeled)
            self.test_data = utils.get_files(test_dir)
            self.test_labels = utils.get_files(test_dir_labeled)
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val and test.")

    def __getitem__(self, index) -> T_co:
        if self.mode.lower() == 'train':
            data_path, label_path = self.train_data[index], self.train_labels[index]
        elif self.mode.lower() == 'val':
            data_path, label_path = self.val_data[index], self.val_labels[index]
        elif self.mode.lower() == 'test':
            data_path, label_path = self.test_data[index], self.test_labels[index]
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val and test.")
        img = utils.get_data_transformed(data_path, self.data_transform)
        label = utils.get_data_transformed(label_path, self.label_transform)
        label = utils.get_target_mask(label, self.class_encoding)
        return img, label

    def __len__(self):
        if self.mode.lower() == 'train':
            return len(self.train_data)
        elif self.mode.lower() == 'val':
            return len(self.val_data)
        elif self.mode.lower() == 'test':
            return len(self.test_data)
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val and test.")
