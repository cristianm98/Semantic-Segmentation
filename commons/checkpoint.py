import os

import torch
import torch.nn as nn
import torch.optim as optim

from utils.arguments import get_arguments

args = get_arguments()
device = torch.device(args.device)
BEST_MODE = "best"
LAST_MODE = "last"


def save_checkpoint(model, optimizer, epoch, miou, ious, mode):
    args_path, model_path = get_checkpoint_paths(mode)
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'miou': miou,
    }
    torch.save(checkpoint, model_path)
    with open(args_path, 'w') as args_file:
        sorted_args = sorted(vars(args))
        for arg in sorted_args:
            entry = "{0}: {1}\n".format(arg, getattr(args, arg))
            args_file.write(entry)
        args_file.write("Epoch: {0}\n".format(epoch))
        args_file.write("Mean IoU: {0}\n".format(miou))
        args_file.write("IoUs: {0}\n".format(ious))


def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, load_dir, mode):
    _, model_path = get_checkpoint_paths(mode)
    assert os.path.isdir(load_dir), \
        '\"{0}\" directory does not exist'.format(load_dir)
    assert os.path.isfile(model_path), \
        '\"{0}\" file does not exist'.format(model_path)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    if device.type == 'cuda':
        model = model.cuda()
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    miou = checkpoint['miou']
    return model.to(device), optimizer, epoch, miou


def get_checkpoint_paths(mode):
    if mode == LAST_MODE:
        model_path = os.path.join(args.save_dir, 'last', args.dataset)
    elif mode == BEST_MODE:
        model_path = os.path.join(args.save_dir, 'best', args.dataset)
    else:
        raise RuntimeError("Unexpected checkpoint mode. Supported modes are: best and last.")
    if args.dataset == 'infrared':
        if args.use_day:
            model_path = os.path.join(model_path, 'day')
        else:
            model_path = os.path.join(model_path, 'night')
    model_path = os.path.join(model_path, args.model)
    args_path = os.path.join(model_path, '_args.txt')
    return args_path, model_path
