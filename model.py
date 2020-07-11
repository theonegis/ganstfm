import torch
import torch.nn as nn
import torch.nn.functional as F
from torchgan.layers import SpectralNorm2d
import enum

from ssim import msssim
from normalization import SwitchNorm2d


class Sampling(enum.Enum):
    UpSampling = enum.auto()
    DownSampling = enum.auto()
    Identity = enum.auto()
    MaxPooling = enum.auto()


NUM_BANDS = 6
PATCH_SIZE = 256
SCALE_FACTOR = 16


class Upsample(nn.Module):
    def __init__(self, scale_factor=2):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, inputs):
        return F.interpolate(inputs, scale_factor=self.scale_factor)


class Conv3X3NoPadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv3X3NoPadding, self).__init__(in_channels, out_channels, 3, stride=stride, padding=1)


class Conv3X3WithPadding(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv3X3WithPadding, self).__init__(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3, stride=stride)
        )


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, sampling=None):
        layers = []
        if sampling == Sampling.UpSampling:
            layers.append(Conv3X3WithPadding(in_channels, out_channels))
            layers.append(Upsample(2))
        elif sampling == Sampling.DownSampling:
            layers.append(Conv3X3WithPadding(in_channels, out_channels, 2))
        elif sampling == Sampling.MaxPooling:
            layers.append(Conv3X3WithPadding(in_channels, out_channels))
            layers.append(nn.MaxPool2d(2))
        else:
            layers.append(Conv3X3WithPadding(in_channels, out_channels))

        layers.append(nn.LeakyReLU(inplace=True))
        super(ConvBlock, self).__init__(*layers)


class ResidulBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidulBlock, self).__init__()
        channels = min(in_channels, out_channels)
        self.residual = nn.Sequential(
            SwitchNorm2d(in_channels),
            nn.LeakyReLU(),
            Conv3X3WithPadding(in_channels, channels),
            SwitchNorm2d(channels),
            nn.LeakyReLU(),
            nn.Conv2d(channels, out_channels, 1)
        )
        self.transform = Conv3X3WithPadding(in_channels, out_channels)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return self.transform(inputs[0]) + self.residual(inputs[1])
        return self.transform(inputs) + self.residual(inputs)


class ReconstructionLoss(nn.Module):
    def __init__(self, model, alpha=1.0, beta=1.0):
        super(ReconstructionLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.model = model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.encoder = nn.Sequential(
            self.model.conv1,
            self.model.conv2,
            self.model.conv3,
            self.model.conv4
        )

    def forward(self, prediction, target):
        loss = (F.l1_loss(prediction, target) +
                self.alpha * F.mse_loss(self.encoder(prediction), self.encoder(target)) +
                self.beta * (1.0 - msssim(prediction, target, normalize=True)))
        return loss


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=NUM_BANDS, out_channels=NUM_BANDS):
        super(AutoEncoder, self).__init__()
        channels = (16, 32, 64, 128)
        self.conv1 = ConvBlock(in_channels, channels[0])
        self.conv2 = ConvBlock(channels[0], channels[1], Sampling.DownSampling)
        self.conv3 = ConvBlock(channels[1], channels[2], Sampling.DownSampling)
        self.conv4 = ConvBlock(channels[2], channels[3], Sampling.DownSampling, )
        self.conv5 = ConvBlock(channels[3], channels[2], Sampling.UpSampling)
        self.conv6 = ConvBlock(channels[2] * 2, channels[1], Sampling.UpSampling)
        self.conv7 = ConvBlock(channels[1] * 2, channels[0], Sampling.UpSampling)
        self.conv8 = nn.Conv2d(channels[0] * 2, out_channels, 1)

    def forward(self, inputs):
        l1 = self.conv1(inputs)
        l2 = self.conv2(l1)
        l3 = self.conv3(l2)
        l4 = self.conv4(l3)
        l5 = self.conv5(l4)
        l6 = self.conv6(torch.cat((l3, l5), 1))
        l7 = self.conv7(torch.cat((l2, l6), 1))
        l8 = self.conv8(torch.cat((l1, l7), 1))
        return l8


class SFFusion(nn.Sequential):
    def __init__(self, in_channels=NUM_BANDS, out_channels=NUM_BANDS):
        channels = (16, 32, 64, 128)
        super(SFFusion, self).__init__(
            ResidulBlock(in_channels, channels[0]),
            ResidulBlock(channels[0], channels[1]),
            ResidulBlock(channels[1], channels[2]),
            ResidulBlock(channels[2], channels[3]),
            ResidulBlock(channels[3], channels[2]),
            ResidulBlock(channels[2], channels[1]),
            ResidulBlock(channels[1], channels[0]),
            nn.Conv2d(channels[0], out_channels, 1)
        )


class ResidulBlockWithSN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidulBlockWithSN, self).__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            SpectralNorm2d(
                Conv3X3NoPadding(in_channels, in_channels, stride=2)),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            SpectralNorm2d(
                nn.Conv2d(in_channels, out_channels, 1)),
        )
        self.transform = SpectralNorm2d(nn.Conv2d(in_channels, out_channels, 1, stride=2))

    def forward(self, inputs):
        return self.transform(inputs) + self.residual(inputs)


# 判别器网络
class Discriminator(nn.Sequential):
    def __init__(self):
        channels = [NUM_BANDS * 2, 32, 32, 64, 64, 128, 128, 256, 256]
        modules = []
        for i in range(1, (len(channels))):
            modules.append(ResidulBlockWithSN(channels[i - 1], channels[i]))
        modules.append(SpectralNorm2d(nn.Conv2d(channels[-1], 1, 1)))
        super(Discriminator, self).__init__(*modules)

    def forward(self, inputs):
        prediction = super(Discriminator, self).forward(inputs)
        return prediction.view(-1, 1).squeeze(1)
