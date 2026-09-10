# This code is adapted from https://github.com/milesial/Pytorch-UNet/blob/master/unet

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, avgPool):
        super().__init__()
        if avgPool:
            self.maxpool_conv = nn.Sequential(
                nn.AvgPool2d(2),
                DoubleConv(in_channels, out_channels)
            )
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
            )


    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)


    def forward(self, x):
        return self.conv(x)
    

class UNet(nn.Module):
    def __init__(self, config, bilinear=False):
        super(UNet, self).__init__()
        self.config = config
        self.bilinear = bilinear

        self.inc = (DoubleConv(self.config.in_im_c, 64))
        self.down1 = (Down(64, 128, config.avgPool))
        self.down2 = (Down(128, 256, config.avgPool))
        self.down3 = (Down(256, 512, config.avgPool))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, config.avgPool))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, self.config.out_im_c))


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class HPOUNet(nn.Module):

    """
    This class is an adaptation of the UNet class above, which can be modified
    more easily for use in HPO.
    """

    def __init__(self, config, bilinear=False):
        super(HPOUNet, self).__init__()
        self.config = config
        self.bilinear = bilinear
        torch.manual_seed(self.config.seed)

        # Input layer
        self.inc = (DoubleConv(self.config.in_im_c, self.config.im_h))
        factor = 2 if bilinear else 1

        # Down Sampling layers
        self.downLayers = nn.ModuleList()
        for index, layer in enumerate(self.config.layers[:-1]):
            self.downLayers.append(Down(layer, self.config.layers[index+1],
                                        config.avgPool))

        # Up sampling layers
        self.upLayers = nn.ModuleList()
        reverseLayers = [i for i in reversed(self.config.layers)]
        for index, layer in enumerate(reverseLayers[:-2]):
            self.upLayers.append(Up(layer, reverseLayers[index+1] // factor, bilinear))

        self.upLayers.append(Up(reverseLayers[-2], reverseLayers[-1], bilinear))

        self.outc = (OutConv(self.config.im_h, self.config.out_im_c))


    def forward(self, x):
        inc = self.inc(x)
        downLayers = []
        downLayers.append(inc)
        for downLayer in self.downLayers:
            downLayers.append(downLayer(downLayers[-1]))

        
        upLayers = []

        upLayers.append(self.upLayers[0](downLayers[-1], downLayers[-2]))

        for index, upLayer in enumerate(self.upLayers[1:-1]):
            downLayer = downLayers[-(index+3)]

            upLayers.append(upLayer(upLayers[-1], downLayer))

        upLayers.append(self.upLayers[-1](upLayers[-1], inc))

        logits = self.outc(upLayers[-1])
        return logits


    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.downLayers = [torch.utils.checkpoint(self.downLayers[i]) for i in range(len(self.config.layers))]
        self.upLayers = [torch.utils.checkpoint(self.upLayers[i]) for i in range(len(self.config.layers))]
        self.outc = torch.utils.checkpoint(self.outc)


if __name__ == "__main__":
    class config(object):
        im_h = 64
        in_im_c = 3
        out_im_c = 2
        avgPool = True

        seed = 42
        layers = [64, 128, 256, 512, 1024]

    staticModel = UNet(config)
    HPOmodel = HPOUNet(config)
    print(staticModel.eval())
    print(HPOmodel.eval())