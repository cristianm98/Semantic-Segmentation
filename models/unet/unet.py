import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, features=None, use_dropout=True):
        super(Unet, self).__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder = self.down(in_channels, features)
        self.decoder = self.up(features)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.conv1x1 = nn.Conv2d(features[0], num_classes, kernel_size=1)
        self.use_dropout = use_dropout

    def down(self, in_channels, features):
        encoder = nn.ModuleList()
        for f in features:
            encoder.append(DoubleConv(in_channels, f))
            in_channels = f
        return encoder

    def up(self, features):
        decoder = nn.ModuleList()
        for f in reversed(features):
            decoder.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            decoder.append(DoubleConv(f * 2, f))
        return decoder

    def forward(self, x):
        skip_connections = []

        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            if self.use_dropout:
                x = nn.Dropout()(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = list(reversed(skip_connections))
        for index in range(0, len(self.decoder), 2):
            x = self.decoder[index](x)
            skip_conn = skip_connections[index // 2]
            if x.shape != skip_conn.shape:
                x = TF.resize(x, size=skip_conn.shape[2:])
            skip_cat = torch.cat((skip_conn, x), dim=1)
            x = self.decoder[index + 1](skip_cat)
        return self.conv1x1(x)
