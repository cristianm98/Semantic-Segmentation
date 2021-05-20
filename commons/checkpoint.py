import os
import torch
import torch.nn as nn
import torch.optim as optim

from commons.arguments import get_arguments

args = get_arguments()
device = torch.device(args.device)


def save_checkpoint(model, optimizer, epoch, miou, ious, mode):
    args_path, model_path = get_checkpoint_paths(mode)
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'miou': miou,
        'ious': ious,
    }
    torch.save(checkpoint, model_path)
    with open(args_path, 'w') as args_file:
        sorted_args = sorted(vars(args))
        for arg in sorted_args:
            entry = "{0}: {1}\n".format(arg, getattr(args, arg))
            args_file.write(entry)
        args_file.write("Epoch: {0}\n".format(epoch))
        args_file.write("Mean IoU: {0}\n".format(miou))


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
    ious = checkpoint['ious']
    return model.to(device), optimizer, epoch, miou, ious


def get_checkpoint_paths(mode):
    if mode == 'train_best' or mode == 'val_best' or mode == 'last':
        args_path = os.path.join(args.save_dir, mode + '_' + args.name + '_' + args.dataset + '_args.txt')
        model_path = os.path.join(args.save_dir, mode + '_' + args.name + '_' + args.dataset)
    else:
        raise RuntimeError("Unexpected checkpoint mode. Supported modes are: val_best, train_best and last.")
    return args_path, model_path
