import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, nonlin):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            *[nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                        padding=padding),
              nn.BatchNorm2d(out_channels),
              nonlin()
              ])

    def forward(self, x):
        return self.conv1(x)
