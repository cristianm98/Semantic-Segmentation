import torch

import commons.utils as commons_utils
import datasets.utils as dataset_utils
from commons.arguments import get_arguments
from commons.checkpoint import load_checkpoint

args = get_arguments()

if __name__ == '__main__':
    print(torch.__version__)
    if args.dataset.lower() == 'camvid':
        from datasets.camvid import CamVid as dataset
    else:
        raise RuntimeError('\"{0}\" is not a supported dataset.'.format(args.dataset))
    data_loaders, class_encoding = dataset_utils.load_dataset(dataset)
    train_loader, val_loader, test_loader = data_loaders
    num_classes = len(class_encoding)
    model, criterion, optimizer, metric = commons_utils.get_parameters(num_classes)
    if args.load_model:
        model = load_checkpoint(model, optimizer, args.save_dir, args.name)
    else:
        model = commons_utils.train(model, optimizer, criterion, metric, train_loader, val_loader, class_encoding)
    commons_utils.test(model, criterion, metric, test_loader, class_encoding)
