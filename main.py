import numpy as np
import torch
from torch.backends import cudnn

import commons.utils as commons_utils
import datasets.utils as dataset_utils
from commons.arguments import get_arguments
from commons.checkpoint import load_checkpoint, VAL_MODE, TRAIN_MODE

args = get_arguments()

# TODO remove workers from dataset => possible fix for non-deterministic results
if __name__ == '__main__':
    # np.random.seed(123)
    # torch.manual_seed(123)
    # device = torch.device(args.device)
    # if device.type == 'cuda':
    #     torch.cuda.manual_seed(123)
    # cudnn.benchmark = True
    print(torch.__version__)
    if args.dataset.lower() == 'camvid':
        from datasets.camvid import CamVid as dataset
        data_loaders, class_encoding = dataset_utils.load_dataset(dataset)
    elif args.dataset.lower() == 'kitti':
        from datasets.kitti_data_road import Kitti as dataset
        data_loaders, class_encoding = dataset_utils.load_dataset(dataset, split_dataset=True)
    else:
        raise RuntimeError('\"{0}\" is not a supported dataset.'.format(args.dataset))
    # data_loaders, class_encoding = dataset_utils.load_dataset(dataset)
    train_loader, val_loader, test_loader = data_loaders
    num_classes = len(class_encoding)
    model, criterion, optimizer, metric = commons_utils.get_parameters(num_classes)
    if args.load_model:
        model, optimizer, epoch, miou = load_checkpoint(model, optimizer, args.save_dir, VAL_MODE)
    else:
        model = commons_utils.train(model, optimizer, criterion, metric, train_loader, val_loader, class_encoding)
    print("[Test] Testing on best val model...")
    model, optimizer, _, _ = load_checkpoint(model, optimizer, args.save_dir, VAL_MODE)
    commons_utils.test(model, criterion, metric, test_loader, class_encoding)
    print("[Test] Testing on best train model...")
    model, optimizer, _, _ = load_checkpoint(model, optimizer, args.save_dir, TRAIN_MODE)
    commons_utils.test(model, criterion, metric, test_loader, class_encoding)
