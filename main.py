import torch

import commons.utils as commons_utils
import datasets.utils as dataset_utils
from utils.arguments import get_arguments
from commons.checkpoint import load_checkpoint, BEST_MODE

args = get_arguments()

# TODO remove workers from dataset => possible fix for non-deterministic results
if __name__ == '__main__':
    print(torch.__version__)
    # TODO allow choosing split ratio
    if args.dataset.lower() == 'camvid':
        from datasets.camvid import CamVid as dataset
        data_loaders, class_encoding = dataset_utils.load_dataset(dataset)
    elif args.dataset.lower() == 'kitti':
        from datasets.kitti_data_road import Kitti as dataset
        data_loaders, class_encoding = dataset_utils.load_dataset(dataset)
    elif args.dataset.lower() == 'infrared':
        if args.use_day:
            from datasets.infrared import InfraRedDay as dataset
        else:
            from datasets.infrared import InfraRedNight as dataset
        data_loaders, class_encoding = dataset_utils.load_dataset(dataset)
    else:
        raise RuntimeError('\"{0}\" is not a supported dataset.'.format(args.dataset))
    train_loader, val_loader, test_loader = data_loaders
    num_classes = len(class_encoding)
    model, criterion, optimizer, metric = commons_utils.get_parameters(num_classes)
    if args.load_model:
        model, optimizer, epoch, miou = load_checkpoint(model, optimizer, args.save_dir, BEST_MODE)
    else:
        model = commons_utils.train(model, optimizer, criterion, metric, train_loader, val_loader, class_encoding)
    print("[Test] Testing on best model...")
    model, optimizer, _, _ = load_checkpoint(model, optimizer, args.save_dir, BEST_MODE)
    commons_utils.test(model, criterion, metric, test_loader, class_encoding)
