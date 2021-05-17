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
import transforms as ext_transforms
from commons.arguments import get_arguments
from commons.tester import Tester
from commons.trainer import Trainer
from metrics.iou import IoU
from commons.checkpoint import save_checkpoint, load_checkpoint

args = get_arguments()
device = torch.device(args.device)


def train(model, optimizer, criterion, metric, train_loader, val_loader, class_encoding):
    print("\nTraining...\n")
    print(model)
    trainer = Trainer(model=model, data_loader=train_loader, optimizer=optimizer, criterion=criterion, metric=metric,
                      device=device)
    val = Tester(model=model, data_loader=val_loader, criterion=criterion, metric=metric, device=device)
    if args.resume_training:
        model, optimizer, start_epoch, best_miou = load_checkpoint(
            model, optimizer, args.save_dir, load_best_result=False)
        start_epoch += 1
        print("Resuming from model: Start epoch = {0} "
              "| Best mean IoU = {1:.4f}".format(start_epoch, best_miou))
    else:
        start_epoch = 0
        best_miou = 0
    best_result = {
        'iou': [],
        'miou': best_miou,
        'epoch': start_epoch
    }
    for epoch in tqdm(range(start_epoch, args.epochs)):
        print("[Epoch: {0:d} | Training] Start epoch...".format(epoch))
        loss, (iou, miou) = trainer.run_epoch()
        print("[Epoch: {0:d} | Training] Finish epoch...\n"
              "Results: Avg Loss:{1:.4f} | MIoU: {2:.4f}".format(epoch, loss, miou))
        print(dict_ious(class_encoding, iou))
        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print("[Epoch: {0:d} | Validation] Start epoch...".format(epoch))
            loss, (iou, miou) = val.run_epoch()
            print(dict_ious(class_encoding, iou))
            print("[Epoch: {0:d} | Validation] Finish epoch...\n"
                  "Results: Avg loss: {1:.4f} | MIoU: {2:.4f}".format(epoch, loss, miou))
            if miou > best_result['miou']:
                best_result['miou'] = miou
                best_result['epoch'] = epoch
                best_result['iou'] = iou
                save_checkpoint(model, optimizer, epoch, miou, save_best_result=True)
            save_checkpoint(model, optimizer, epoch, miou, save_best_result=False)
    return model


def test(model, criterion, metric, test_loader, class_encoding):
    print("\nTesting...\n")
    tester = Tester(model=model, data_loader=test_loader, criterion=criterion, metric=metric, device=device)
    if args.dataset != 'kitti':
        loss, (iou, miou) = tester.run_epoch()
        print(dict_ious(class_encoding, iou))
        print("[Test] Avg loss: {0:.4f} | MIoU: {1:.4f}".format(loss, miou))
    data, _ = iter(test_loader).__next__()
    if device.type == 'cuda':
        model.cuda()
    predict(model, data, class_encoding)


def predict(model, images, class_encoding):
    images = images.to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(images)
    _, predictions = torch.max(predictions.data, 1)
    pred_transform = transforms.Compose([
        ext_transforms.LongTensorToRGBPIL(class_encoding),
        transforms.ToTensor()
    ])
#     predictions = batch_transform(predictions.cpu(), label_to_rgb)
    imshow_batch(images.detach().cpu(), predictions.detach().cpu(), pred_transform)
#     save_results(images.detach().cpu(), predictions.detach().cpu())


def batch_transform(batch, transform):
    transf_slices = [transform(tensor) for tensor in torch.unbind(batch)]
    return torch.stack(transf_slices)


def imshow_batch(images, predictions, pred_transform):
    predictions = batch_transform(predictions, pred_transform)
    images = torchvision.utils.make_grid(images).numpy()
    predictions = torchvision.utils.make_grid(predictions).numpy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7))
    ax1.imshow(np.transpose(images, (1, 2, 0)))
    ax2.imshow(np.transpose(predictions, (1, 2, 0)))
    plt.show()


def save_results(images, predictions):
    for idx, img in enumerate(images):
        pil_img = transforms.ToPILImage()(img)
        img_path = os.path.join(args.save_dir, 'Results', 'img_' + str(idx))
        if not os.path.exists(img_path):
            open(img_path).close()
        pil_img.save(img_path, 'PNG')
    for idx, img in enumerate(predictions):
        pil_img = transforms.ToPILImage()(img)
        img_path = os.path.join(args.save_dir, 'results', 'pred_' + str(idx))
        if not os.path.exists(img_path):
            open(img_path).close()
        pil_img.save(img_path, 'PNG')


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
