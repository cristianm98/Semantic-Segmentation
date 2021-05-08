import os
import torch
import torch.nn as nn
import torch.optim as optim

from commons.arguments import get_arguments

args = get_arguments()
device = torch.device(args.device)


def save_checkpoint(model, optimizer, epoch, miou, args):
    name = args.name
    save_dir = args.save_dir
    model_path = os.path.join(save_dir, name)
    checkpoint = {
        'epoch': epoch,
        'miou': miou,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, model_path)
    args_file = os.path.join(save_dir, name + '_args.txt')
    with open(args_file, 'w') as args_file:
        sorted_args = sorted(vars(args))
        for arg in sorted_args:
            entry = "{0}: {1}\n".format(arg, getattr(args, arg))
            args_file.write(entry)
        args_file.write("Epoch: {0}\n".format(epoch))
        args_file.write("Mean IoU: {0}\n".format(miou))


def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, load_dir, name):
    assert os.path.isdir(load_dir), \
        '\"{0}\" directory does not exist'.format(load_dir)
    model_path = os.path.join(load_dir, name)
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
