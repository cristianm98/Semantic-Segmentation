import os

import torch.utils.data
import torch.utils.data as data
import torchvision.transforms as TF
from PIL import Image

from utils.arguments import get_arguments

args = get_arguments()


def get_files(dir_path):
    if not os.path.isdir(dir_path):
        raise RuntimeError("\"{0}\" is not a directory.".format(dir_path))
    result = []
    for path, _, files in os.walk(dir_path):
        files.sort()
        for f in files:
            full_path = os.path.join(path, f)
            result.append(full_path)
    return result


def load_dataset(dataset):
    train_set = get_dataset(dataset, 'train')
    val_set = get_dataset(dataset, 'val')
    test_set = get_dataset(dataset, 'test')
    train_loader = get_dataloader(train_set, shuffle=True)
    val_loader = get_dataloader(val_set)
    test_loader = get_dataloader(test_set)
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


def get_image_transform(mode):
    image_transform = []
    if args.data_aug and mode == 'train':
        image_transform = [TF.Resize((args.width, args.height)), TF.RandomHorizontalFlip(0.3), TF.RandomRotation(15)]
    image_transform.append(TF.ToTensor())
    return TF.Compose(image_transform)


def get_target_transform(mode):
    target_transform = []
    if args.data_aug and mode == 'train':
        target_transform = [TF.Resize((args.width, args.height), TF.InterpolationMode.NEAREST),
                            TF.RandomHorizontalFlip(0.5), TF.RandomRotation(15)]
    target_transform.append(TF.PILToTensor())
    return TF.Compose(target_transform)


def get_dataloader(dataset, shuffle=False):
    return data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        drop_last=True
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
