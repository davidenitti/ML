"""
WARNING this is beta code!
"""
import torch
from torch import nn
import torch.nn.functional as F
import random


class PixelNorm2d(nn.Module):
    """
    similar to local responce normalization.
    taken from
    Progressive Growing of GANs for Improved Quality, Stability, and Variation
    """

    def __init__(self, chan, affine=True):
        super(PixelNorm2d, self).__init__()
        self.eps = 0.0000001
        self.gain = nn.Parameter(torch.ones((1, chan, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, chan, 1, 1)))

    def forward(self, input):
        out = input / (torch.mean(input ** 2, dim=1, keepdim=True) + self.eps).sqrt()
        out = out * self.gain + self.bias
        return out


class GaussianActivation(nn.Module):
    def __init__(self):
        super(GaussianActivation, self).__init__()
        self.gain = nn.Parameter(torch.ones((1,)))
        self.bias = nn.Parameter(-torch.ones((1,)))
        self.mean = nn.Parameter(torch.zeros((1,)))
        self.std = nn.Parameter(torch.ones((1,)))

    def forward(self, x):
        return torch.exp(-((x - self.mean) ** 2 / (self.std + 0.001) ** 2)) + self.bias


class Noise(nn.Module):
    def __init__(self, noise_std):
        super(Noise, self).__init__()
        self.noise_std = noise_std

    def forward(self, x):
        if self.noise_std > 0 and self.training:
            return x + torch.randn_like(x, requires_grad=False, device=x.device) * self.noise_std
        else:
            return x


class TanhMod(nn.Module):
    def __init__(self, scale):
        super(TanhMod, self).__init__()
        self.scale = scale
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.tanh(x / self.scale) * self.scale


class TanhScaled(nn.Module):
    def __init__(self):
        super(TanhScaled, self).__init__()
        self.gain = nn.Parameter(torch.ones((1,)))
        self.bias = nn.Parameter(torch.zeros((1,)))
        self.bias2 = nn.Parameter(torch.zeros((1,)))
        self.tanh = nn.Tanh()

    def forward(self, x):
        scale = torch.abs(self.gain) + 0.001
        return self.tanh((x + self.bias) / scale) * scale + self.bias2

class ConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, nonlin, batch_norm,
                 noise_std=0.0, affine=True):
        super(ConvBlock2, self).__init__()
        self.conv1 = nn.Sequential(
            *[nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                        padding=padding),
              batch_norm(out_channels, affine=affine),
              Noise(noise_std),
              nonlin()
              ])

    def forward(self, x):
        return self.conv1(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, nonlin, batch_norm,
                 noise_std=0.0, affine=True):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            *[nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                        padding=padding),
              batch_norm(out_channels, affine=affine),
              nonlin()
              ])
        self.noise_std = noise_std

    def forward(self, x):
        if self.noise_std > 0 and self.training:
            x = x + torch.randn_like(x, requires_grad=False, device=x.device) * self.noise_std
        return self.conv1(x)


class Identity(nn.Module):
    def __init__(self, channels=None, affine=None):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        shape = [x.shape[0]] + self.shape
        return x.view(*shape)


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x


class CropPad(nn.Module):
    def __init__(self, height, width):
        super(CropPad, self).__init__()
        self.height = height
        self.width = width

    def forward(self, x):
        self.crop_h = torch.FloatTensor([x.size()[2]]).sub(self.height).div(-2)
        self.crop_w = torch.FloatTensor([x.size()[3]]).sub(self.width).div(-2)
        return F.pad(x, [
            self.crop_w.ceil().int()[0], self.crop_w.floor().int()[0],
            self.crop_h.ceil().int()[0], self.crop_h.floor().int()[0],
        ])


class ResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, nonlin=None, batch_norm=None,
                 noise_std=0.0, affine=True):
        if batch_norm is None:
            batch_norm = nn.BatchNorm2d
        if nonlin is None:
            nonlin = nn.ReLU
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=False)
        self.bn1 = batch_norm(out_channels, affine=affine)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding,
                               bias=False)
        self.bn2 = batch_norm(out_channels, affine=affine)
        self.nonlin = nonlin()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                batch_norm(self.expansion * out_channels, affine=affine)
            )
        self.noise = Noise(noise_std)

    def forward(self, x):
        out = self.nonlin(self.noise(self.bn1(self.conv1(x))))
        out = self.noise(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        out = self.nonlin(out)
        return out
