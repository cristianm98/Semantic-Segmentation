from utils.arguments import get_arguments
from models.pspnet import pspnet
from models.segnet import segnet
from models.unet import unet

args = get_arguments()


def get_model(num_classes, pretrained):
    model = None
    if args.model == 'pspnet':
        model = pspnet.PspNet(num_classes=num_classes, pretrained=pretrained)
    elif args.model == 'segnet':
        model = segnet.SegNet(num_classes=num_classes, pretrained=pretrained)
    elif args.model == 'unet':
        model = unet.Unet(num_classes=num_classes)
    return model
