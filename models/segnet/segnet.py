import torch.nn as nn
from torchvision import models
from .utils import Down2, Down3, Up2, Up3


class SegNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, pretrained=True):
        super(SegNet, self).__init__()

        self.down1 = Down2(in_channels, 64)
        self.down2 = Down2(64, 128)
        self.down3 = Down3(128, 256)
        self.down4 = Down3(256, 512)
        self.down5 = Down3(512, 512)

        self.up1 = Up3(512, 512)
        self.up2 = Up3(512, 256)
        self.up3 = Up3(256, 128)
        self.up4 = Up2(128, 64)
        self.up5 = Up2(64, num_classes)

        vgg16 = models.vgg16_bn(pretrained)
        self.init_vgg16(vgg16)

    def forward(self, x):
        down1, indices_1, out_size_1 = self.down1(x)
        down2, indices_2, out_size_2 = self.down2(down1)
        down3, indices_3, out_size_3 = self.down3(down2)
        down4, indices_4, out_size_4 = self.down4(down3)
        down5, indices_5, out_size_5 = self.down5(down4)
        up1 = self.up1(down5, indices_5, out_size_5)
        up2 = self.up2(up1, indices_4, out_size_4)
        up3 = self.up3(up2, indices_3, out_size_3)
        up4 = self.up4(up3, indices_2, out_size_2)
        up5 = self.up5(up4, indices_1, out_size_1)
        return up5

    def init_vgg16(self, vgg16):
        layers = [self.down1, self.down2, self.down3, self.down4, self.down5]
        features = list(vgg16.features.children())
        vgg_layers = []
        for f in features:
            if isinstance(f, nn.Conv2d):
                vgg_layers.append(f)
        seg_layers = []
        for idx, conv_block in enumerate(layers):
            if idx < 2:
                units = [conv_block.conv1.layer, conv_block.conv2.layer]
            else:
                units = [conv_block.conv1.layer, conv_block.conv2.layer, conv_block.conv3.layer]
            for u in units:
                for layer in u:
                    if isinstance(layer, nn.Conv2d):
                        seg_layers.append(layer)
        assert len(vgg_layers) == len(seg_layers)

        for vgg_layer, seg_layer in zip(vgg_layers, seg_layers):
            if isinstance(vgg_layer, nn.Conv2d) and isinstance(seg_layer, nn.Conv2d):
                assert vgg_layer.weight.size() == seg_layer.weight.size()
                assert vgg_layer.bias.size() == seg_layer.bias.size()
                seg_layer.weight.data = vgg_layer.weight.data
                seg_layer.bias.data = vgg_layer.bias.data
