import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch import optim
from tqdm.auto import tqdm

import models.utils as model_utils
from utils import transforms as ext_transforms
from utils.arguments import get_arguments
from commons.checkpoint import save_checkpoint, load_checkpoint, BEST_MODE, LAST_MODE
from commons.tester import Tester
from commons.trainer import Trainer
from metrics.iou import IoU

args = get_arguments()
device = torch.device(args.device)


def train(model, optimizer, criterion, metric, train_loader, val_loader, class_encoding):
    print("\nTraining...\n")
    print(model)
    trainer = Trainer(model=model, data_loader=train_loader, optimizer=optimizer, criterion=criterion, metric=metric,
                      device=device)
    val = Tester(model=model, data_loader=val_loader, criterion=criterion, metric=metric, device=device)
    if args.resume_training:
        try:
            try:
                _, _, epoch, miou = load_checkpoint(model, optimizer, args.save_dir, BEST_MODE)
                best_val_result = init_best_result(epoch, miou)
            except AssertionError:
                best_val_result = init_best_result(0, 0)
                print("Checkpoint for best model not found")
            model, optimizer, start_epoch, best_miou = load_checkpoint(model, optimizer, args.save_dir, LAST_MODE)
            start_epoch += 1
            print("Resuming from model: Start epoch = {0} "
                  "| Best mean IoU = {1:.4f}".format(start_epoch, best_miou))
        except AssertionError:
            best_miou = 0
            start_epoch = 0
            best_val_result = init_best_result(start_epoch, best_miou)
            print("Checkpoint file not found. Starting from model: Start epoch = {0} "
                  "| Best mean IoU = {1:.4f}".format(start_epoch, best_miou))
    else:
        start_epoch = 0
        best_miou = 0
        best_val_result = init_best_result(start_epoch, best_miou)

    for epoch in tqdm(range(start_epoch, args.epochs)):
        print("[Epoch: {0:d} | Training] Start epoch...".format(epoch))
        loss, (ious, miou) = trainer.run_epoch()
        print("[Epoch: {0:d} | Training] Finish epoch...\n"
              "Results: Avg Loss:{1:.4f} | MIoU: {2:.4f}".format(epoch, loss, miou))
        print(dict_ious(class_encoding, ious))
        ious = dict_ious(class_encoding, ious)
        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print("[Epoch: {0:d} | Validation] Start epoch...".format(epoch))
            loss, (ious, miou) = val.run_epoch()
            print(dict_ious(class_encoding, ious))
            print("[Epoch: {0:d} | Validation] Finish epoch...\n"
                  "Results: Avg loss: {1:.4f} | MIoU: {2:.4f}".format(epoch, loss, miou))
            best_ious = best_val_result['ious']
            if miou > best_val_result['miou']:
                best_val_result['miou'] = miou
                best_val_result['epoch'] = epoch
                best_val_result['ious'] = ious
                save_checkpoint(model, optimizer, epoch, miou, ious, BEST_MODE)
        save_checkpoint(model, optimizer, epoch, miou, ious, LAST_MODE)
    return model


def test(model, criterion, metric, test_loader, class_encoding):
    print("\nTesting...\n")
    tester = Tester(model=model, data_loader=test_loader, criterion=criterion, metric=metric, device=device)
    paths = None
    if args.dataset == 'kitti':
        data = iter(test_loader).__next__()
    elif args.dataset == 'crossir':
        data, paths = iter(test_loader).__next__()
    else:
        loss, (iou, miou) = tester.run_epoch()
        print(dict_ious(class_encoding, iou))
        print("[Test] Avg loss: {0:.4f} | MIoU: {1:.4f}".format(loss, miou))
        data, _ = iter(test_loader).__next__()
    if device.type == 'cuda':
        model.cuda()
    predict(model, data, class_encoding, paths=paths)


def predict(model, images, class_encoding, paths=None):
    images = images.to(device)
    model.eval()
    with torch.no_grad():
        if model.__class__.__name__.lower() == 'fcn':
            predictions = model(images)['out']
        else:
            predictions = model(images)
    _, predictions = torch.max(predictions.data, 1)
    pred_transform = transforms.Compose([
        ext_transforms.LongTensorToRGBPIL(class_encoding),
        transforms.ToTensor()
    ])
    # predictions = batch_transform(predictions.cpu(), label_to_rgb)
    images = images.detach().cpu()
    predictions = predictions.detach().cpu()
    imshow_batch(images, predictions, pred_transform)
    save_results(images, paths, predictions)


def batch_transform(batch, transform):
    transf_slices = [transform(tensor) for tensor in torch.unbind(batch)]
    return torch.stack(transf_slices)


def imshow_batch(images, predictions, pred_transform):
    predictions = batch_transform(predictions, pred_transform)
    images = torchvision.utils.make_grid(images).numpy()
    predictions = torchvision.utils.make_grid(predictions).numpy()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 7))
    ax1.imshow(np.transpose(images, (1, 2, 0)))
    ax2.imshow(np.transpose(predictions, (1, 2, 0)))
    ax3.imshow(np.transpose(images, (1, 2, 0)))
    ax3.imshow(np.transpose(predictions, (1, 2, 0)), alpha=0.5)


def save_results(images, paths, predictions):
    for idx, img in enumerate(images):
        pil_img = transforms.ToPILImage()(img)
        new_img_path = os.path.join(args.results_dir, 'img_' + str(idx) + '.bmp')
        # if not os.path.exists(new_img_path):
        #     open(new_img_path).close()
        pil_img.save(new_img_path)
        # torchvision.utils.save_image(img, new_img_path)
    for idx, img in enumerate(predictions):
        pil_img = transforms.ToPILImage()(img)
        new_img_path = os.path.join(args.results_dir, 'pred_' + str(idx) + '.bmp')
        # if not os.path.exists(new_img_path):
        #     open(new_img_path).close()
        pil_img.save(new_img_path, 'BMP')
        # torchvision.utils.save_image(img, new_img_path)


def get_parameters(num_classes):
    model = model_utils.get_model(num_classes, pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    metric = IoU(num_classes=num_classes, ignore_index=None)
    return model, criterion, optimizer, metric


def dict_ious(class_encoding, ious):
    result = dict()
    for idx, (name, color) in enumerate(class_encoding.items()):
        result[name] = ious[idx]
    return result


def init_best_result(start_epoch, best_miou):
    result = {
        'ious': dict(),
        'miou': best_miou,
        'epoch': start_epoch
    }
    return result
