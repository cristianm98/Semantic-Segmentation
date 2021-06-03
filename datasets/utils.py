import os

import numpy as np
import torch.utils.data
import torch.utils.data as data
import torchvision.transforms as TF
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler

from commons.arguments import get_arguments

args = get_arguments()


def name_cond(name_filter: str):
    if name_filter is None:
        return lambda filename: True
    else:
        return lambda filename: name_filter in filename


def ext_cond(ext_filter: str):
    if ext_filter is None:
        return lambda filename: True
    else:
        return lambda filename: filename.endswith(ext_filter)


def get_files(dir_path, name_filter=None, extension_filter=None):
    if not os.path.isdir(dir_path):
        raise RuntimeError("\"{0}\" is not a directory.".format(dir_path))
    result = []
    for path, _, files in os.walk(dir_path):
        files.sort()
        for f in files:
            full_path = os.path.join(path, f)
            result.append(full_path)
    return result


def load_dataset(dataset, split_dataset=False):
    if split_dataset:
        train_set = get_dataset(dataset, 'train')
        val_set = get_dataset(dataset, 'train')
    else:
        train_set = get_dataset(dataset, 'train')
        val_set = get_dataset(dataset, 'val')
    test_set = get_dataset(dataset, 'test')
    train_loader, val_loader, test_loader = get_loaders(train_set, val_set, test_set, split_dataset)
    class_encoding = dataset.class_encoding
    return (train_loader, val_loader, test_loader), class_encoding


def get_dataset(dataset, mode):
    image_transform = get_image_transform(mode)
    target_transform = get_target_transform(mode)
    return dataset(
        root_dir=args.dataset_dir,
        mode=mode,
        data_transform=image_transform,
        label_transform=target_transform
    )


# TODO double check data augmentation, bc results are worse when applied
def get_image_transform(mode):
    image_transform = [TF.Resize((args.width, args.height))]
    if args.data_aug and mode == 'train':
        image_transform.append(TF.RandomHorizontalFlip(0.3))
        image_transform.append(TF.RandomRotation(15))
    image_transform.append(TF.ToTensor())
    # image_transform.append(TF.Normalize(mean=[0.485, 0.456, 0.406],
    #                                     std=[0.229, 0.224, 0.225]))
    return TF.Compose(image_transform)


def get_target_transform(mode):
    # TODO POSSIBLE PROBLEM: PILToTensor transform
    target_transform = [TF.Resize((args.width, args.height), TF.InterpolationMode.NEAREST)]
    if args.data_aug and mode == 'train':
        target_transform.append(TF.RandomHorizontalFlip(0.5))
        target_transform.append(TF.RandomRotation(15))
    target_transform.append(TF.PILToTensor())
    # target_transform.append(TF.Normalize(mean=[0.485, 0.456, 0.406],
    #                                      std=[0.229, 0.224, 0.225]))
    return TF.Compose(target_transform)


def get_samplers(train_set, shuffle=True, validation_split=.2, random_seed=42):
    dataset_size = len(train_set)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, val_sampler


def get_loaders(train_set, val_set, test_set, split_dataset=False):
    if split_dataset:
        train_sampler, val_sampler = get_samplers(train_set, shuffle=True)
        train_loader = get_dataloader(train_set, sampler=train_sampler)
        val_loader = get_dataloader(val_set, sampler=val_sampler)
    else:
        train_loader = get_dataloader(train_set, shuffle=True)
        val_loader = get_dataloader(val_set)
    test_loader = get_dataloader(test_set)
    return train_loader, val_loader, test_loader


def get_dataloader(dataset, shuffle=False, sampler=None):
    return data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        drop_last=True,
        sampler=sampler
    )


def get_target_mask(target, class_encoding):
    if not torch.is_tensor(target):
        raise RuntimeError("The given target is not a tensor. Most likely, there is no transform applied to the data.")
    colors = class_encoding.values()
    mapping = {tuple(c): t for c, t in zip(colors, range(len(colors)))}
    target_size = list(target.size())
    mask = torch.zeros(target_size[1], target_size[2], dtype=torch.long)
    for k in mapping:
        idx = (target == torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
        validx = (idx.sum(0) == 3)
        mask[validx] = torch.tensor(mapping[k], dtype=torch.long)
    return mask


def get_data_transformed(data_path, transform):
    result = Image.open(data_path).convert('RGB')
    if transform is not None:
        result = transform(result)
    return result
