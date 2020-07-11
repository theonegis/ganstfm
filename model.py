import torch
import torch.nn as nn
import torch.nn.functional as F

from ssim import msssim

NUM_BANDS = 6


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.ReplicationPad2d(1),
        nn.Conv2d(in_channels, out_channels, 3, stride=stride)
    )


class CompoundLoss(nn.Module):
    def __init__(self, pretrained, alpha=0.85, normalize=True):
        super(CompoundLoss, self).__init__()
        self.pretrained = pretrained
        self.alpha = alpha
        self.normalize = normalize

    def forward(self, prediction, target):
        return (F.mse_loss(prediction, target) +
                F.mse_loss(self.pretrained(prediction), self.pretrained(target)) +
                self.alpha * (1.0 - msssim(prediction, target,
                                           normalize=self.normalize)))


class FEncoder(nn.Sequential):
    def __init__(self):
        channels = [NUM_BANDS, 32, 64, 128]
        super(FEncoder, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            conv3x3(channels[2], channels[3]),
            nn.ReLU(True)
        )


class REncoder(nn.Sequential):
    def __init__(self):
        channels = [NUM_BANDS * 3, 32, 64, 128]
        super(REncoder, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            conv3x3(channels[2], channels[3])
        )


class Decoder(nn.Sequential):
    def __init__(self):
        channels = [128, 64, 32, NUM_BANDS]
        super(Decoder, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2]),
            nn.ReLU(True),
            nn.Conv2d(channels[2], channels[3], 1)
        )


class Pretrained(nn.Sequential):
    def __init__(self):
        channels = [NUM_BANDS, 32, 64, 128]
        super(Pretrained, self).__init__(
            conv3x3(channels[0], channels[1]),
            nn.ReLU(True),
            conv3x3(channels[1], channels[2], 2),
            nn.ReLU(True),
            conv3x3(channels[2], channels[3], 2),
            nn.ReLU(True)
        )


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        self.encoder = FEncoder()
        self.residual = REncoder()
        self.decoder = Decoder()

    def forward(self, inputs):
        prev_diff = self.residual(torch.cat((inputs[0], inputs[1], inputs[-1]), 1))
        return self.decoder(self.encoder(inputs[1]) + prev_diff)
