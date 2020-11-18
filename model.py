import torch
import torch.nn as nn
import torch.nn.functional as F

from ssim import msssim
import enum

NUM_BANDS = 6


class Sampling(enum.Enum):
    UPSAMPLING = enum.auto()
    DOWNSAMPLING = enum.auto()
    NOSAMPLING = enum.auto()


class Conv3X3NoPadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv3X3NoPadding, self).__init__(in_channels, out_channels, 3, stride=stride, padding=1)


class Conv3X3WithPadding(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv3X3WithPadding, self).__init__(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3, stride=stride)
        )


class Upsample(nn.Module):
    def __init__(self, scale_factor=2):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, inputs):
        return F.interpolate(inputs, scale_factor=self.scale_factor)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, sampling=None):
        layers = []
        if sampling == Sampling.UpSampling:
            layers.append(Upsample(2))
            layers.append(Conv3X3WithPadding(in_channels, out_channels))
        elif sampling == Sampling.DownSampling:
            layers.append(Conv3X3WithPadding(in_channels, out_channels, 2))

        layers.append(nn.LeakyReLU(inplace=True))
        super(ConvBlock, self).__init__(*layers)


class VisionLoss(nn.Module):
    def __init__(self, alpha=0.84, window_size=11, size_average=True, normalize=True):
        super(VisionLoss, self).__init__()
        self.alpha = alpha
        self.window_size = window_size
        self.size_average = size_average
        self.normalize = normalize

    def forward(self, prediction, target):
        return (F.smooth_l1_loss(prediction, target) + self.alpha * (
                1.0 - msssim(prediction, target, window_size=self.window_size,
                             size_average=self.size_average, normalize=self.normalize)))


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        channels = (NUM_BANDS, 16, 32, 64, 128)
        self.conv1 = ConvBlock(channels[0], channels[1])
        self.conv2 = ConvBlock(channels[1], channels[2], Sampling.DOWNSAMPLING)
        self.conv3 = ConvBlock(channels[2], channels[3], Sampling.DOWNSAMPLING)
        self.conv4 = ConvBlock(channels[3], channels[4], Sampling.DOWNSAMPLING)
        self.conv5 = ConvBlock(channels[4], channels[3], Sampling.UPSAMPLING)
        self.conv6 = ConvBlock(channels[3] * 2, channels[2], Sampling.UPSAMPLING)
        self.conv7 = ConvBlock(channels[2] * 2, channels[1], Sampling.UPSAMPLING)
        self.conv8 = nn.Conv2d(channels[1] * 2, channels[0], 1)

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
