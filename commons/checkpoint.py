import os
import torch
import torch.nn as nn
import torch.optim as optim

from commons.arguments import get_arguments

args = get_arguments()
device = torch.device(args.device)


def save_checkpoint(model, optimizer, epoch, miou, ious, save_best_result: bool):
    if save_best_result:
        args_file = os.path.join(args.save_dir, 'best_' + args.name + '_' + args.dataset + '_args.txt')
        model_path = os.path.join(args.save_dir, 'best_' + args.name + '_' + args.dataset)
    else:
        args_file = os.path.join(args.save_dir, 'last_' + args.name + '_' + args.dataset + '_args.txt')
        model_path = os.path.join(args.save_dir, 'last_' + args.name + '_' + args.dataset)
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'miou': miou,
        'ious': ious,
    }
    torch.save(checkpoint, model_path)
    with open(args_file, 'w') as args_file:
        sorted_args = sorted(vars(args))
        for arg in sorted_args:
            entry = "{0}: {1}\n".format(arg, getattr(args, arg))
            args_file.write(entry)
        args_file.write("Epoch: {0}\n".format(epoch))
        args_file.write("Mean IoU: {0}\n".format(miou))


def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, load_dir, load_best_result: bool):
    assert os.path.isdir(load_dir), \
        '\"{0}\" directory does not exist'.format(load_dir)
    if load_best_result:
        model_path = os.path.join(args.save_dir, 'best_' + args.name + '_' + args.dataset)
    else:
        model_path = os.path.join(args.save_dir, 'last_' + args.name + '_' + args.dataset)
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
