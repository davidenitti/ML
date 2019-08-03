"""
WARNING this is beta code!
"""
import torch
from torch import nn
import torch.nn.functional as F
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core import modules
from core.modules import View, Interpolate, Identity


class Net(nn.Module):
    def __init__(self, num_out=10, net_params=None, shape=None):
        super(Net, self).__init__()

        self.net_params = net_params
        if hasattr(nn, net_params['non_linearity']):
            self.nonlin = getattr(nn, net_params['non_linearity'])
        else:
            self.nonlin = getattr(modules, net_params['non_linearity'])

        if net_params['batchnorm'] == "BatchNorm2d":
            self.batch_norm = nn.BatchNorm2d
        elif net_params['batchnorm'] == "InstanceNorm2d":
            self.batch_norm = nn.InstanceNorm2d
        else:
            raise NotImplementedError

        conv_block = getattr(modules, net_params['conv_block'])
        kernel = net_params['kernel_size']
        self.base = net_params['base']
        self.last_pool = getattr(F, net_params['last_pool'])

        if net_params['padding']:
            self.padding = (kernel - 1) // 2
        else:
            self.padding = 0

        self.conv1 = nn.Sequential(*[
            conv_block(in_channels=3, out_channels=self.base, kernel_size=kernel, stride=1, padding=self.padding,
                       nonlin=self.nonlin, batch_norm=self.batch_norm, noise_std=net_params['noise']),
            conv_block(in_channels=self.base, out_channels=self.base, kernel_size=kernel, stride=1,
                       padding=self.padding,
                       nonlin=self.nonlin, batch_norm=self.batch_norm, noise_std=net_params['noise']),
            conv_block(in_channels=self.base, out_channels=self.base, kernel_size=kernel, stride=1,
                       padding=self.padding,
                       nonlin=self.nonlin, batch_norm=self.batch_norm, noise_std=net_params['noise'])
        ])
        self.conv2 = nn.Sequential(*[
            conv_block(in_channels=self.base, out_channels=self.base * 2, kernel_size=kernel, stride=1,
                       padding=self.padding,
                       nonlin=self.nonlin, batch_norm=self.batch_norm, noise_std=net_params['noise']),
            conv_block(in_channels=self.base * 2, out_channels=self.base * 2, kernel_size=kernel, stride=1,
                       padding=self.padding, nonlin=self.nonlin, batch_norm=self.batch_norm,
                       noise_std=net_params['noise'])
        ])
        self.conv3 = nn.Sequential(*[
            conv_block(in_channels=self.base * 2, out_channels=self.base * 4, kernel_size=kernel, stride=1,
                       padding=self.padding,
                       nonlin=self.nonlin, batch_norm=self.batch_norm, noise_std=net_params['noise']),
            conv_block(in_channels=self.base * 4, out_channels=self.base * 4, kernel_size=kernel, stride=1,
                       padding=self.padding, nonlin=self.nonlin, batch_norm=self.batch_norm,
                       noise_std=net_params['noise'])
        ])
        self.conv4 = nn.Sequential(
            *[conv_block(in_channels=self.base * 4, out_channels=self.base * 8, kernel_size=kernel, stride=1,
                         padding=self.padding, nonlin=self.nonlin, batch_norm=self.batch_norm,
                         noise_std=net_params['noise']),
              conv_block(in_channels=self.base * 8, out_channels=self.base * 8, kernel_size=kernel, stride=1,
                         padding=self.padding, nonlin=self.nonlin, batch_norm=self.batch_norm,
                         noise_std=net_params['noise'])
              ])
        self.classify = nn.Linear(self.base * 8, num_out)
        self.num_out = num_out
        print('net v 1.1')
        self.debug = False

    def decoding(self, encoding):
        debug = self.debug
        # x = self.upconv1(encoding)
        x = encoding
        if debug:
            print('decoding', x.shape)
        x = self.upconv2(x)
        if debug:
            print('decoding', x.shape)
        x = self.upconv_rec(x)
        if debug:
            print('decoding', x.shape)
        return x

    def forward(self, x, get_shape_enc=False):
        if self.debug:
            print(x.shape)

        x = self.conv1(x)

        if self.debug:
            print(x.shape)

        x = F.max_pool2d(x, 2, 2)

        if self.debug:
            print(x.shape)

        x = self.conv2(x)

        if self.debug:
            print(x.shape)

        x = F.max_pool2d(x, 2, 2)

        if self.debug:
            print(x.shape)

        if self.debug:
            print(x.shape)

        x = self.conv3(x)

        x = F.max_pool2d(x, 2, 2)

        if self.debug:
            print(x.shape)

        x = self.conv4(x)

        if get_shape_enc:
            return x.shape

        if self.debug:
            print(x.shape)

        x = self.last_pool(x, x.shape[1])
        if self.debug:
            print(x.shape)
        x = x.view(x.shape[0], -1)
        if self.debug:
            print(x.shape)
        x = self.classify(x)
        if self.debug:
            print(x.shape)

        return x
