import torch
from torch import nn
import torch.nn.functional as F
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core import modules
from core.modules import View, Interpolate, Identity, StaticBatchNorm2d


class Net(nn.Module):
    def __init__(self, num_out=10, net_params=None, shape=None):
        super(Net, self).__init__()

        self.net_params = net_params
        self.nonlin = getattr(nn, net_params['non_linearity'])

        if net_params['batchnorm'] == "standard":
            self.batch_norm = nn.BatchNorm2d
        elif net_params['batchnorm'] == "static":
            self.batch_norm = StaticBatchNorm2d
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
                       nonlin=self.nonlin, batch_norm=self.batch_norm),
            conv_block(in_channels=self.base, out_channels=self.base, kernel_size=kernel, stride=1,
                       padding=self.padding,
                       nonlin=self.nonlin, batch_norm=self.batch_norm),
            conv_block(in_channels=self.base, out_channels=self.base, kernel_size=kernel, stride=1,
                       padding=self.padding,
                       nonlin=self.nonlin, batch_norm=self.batch_norm)
        ])
        self.conv2 = nn.Sequential(*[
            conv_block(in_channels=self.base, out_channels=self.base * 2, kernel_size=kernel, stride=1,
                       padding=self.padding,
                       nonlin=self.nonlin, batch_norm=self.batch_norm),
            conv_block(in_channels=self.base * 2, out_channels=self.base * 2, kernel_size=kernel, stride=1,
                       padding=self.padding, nonlin=self.nonlin, batch_norm=self.batch_norm)
        ])
        self.conv3 = nn.Sequential(*[
            conv_block(in_channels=self.base * 2, out_channels=self.base * 4, kernel_size=kernel, stride=1,
                       padding=self.padding,
                       nonlin=self.nonlin, batch_norm=self.batch_norm),
            conv_block(in_channels=self.base * 4, out_channels=self.base * 4, kernel_size=kernel, stride=1,
                       padding=self.padding, nonlin=self.nonlin, batch_norm=self.batch_norm)
        ])
        self.conv4 = nn.Sequential(
            *[conv_block(in_channels=self.base * 4, out_channels=self.base * 8, kernel_size=kernel, stride=1,
                         padding=self.padding, nonlin=self.nonlin, batch_norm=self.batch_norm),
              conv_block(in_channels=self.base * 8, out_channels=self.base * 8, kernel_size=kernel, stride=1,
                         padding=self.padding, nonlin=self.nonlin, batch_norm=self.batch_norm)
              ])
        self.classify = nn.Linear(self.base * 8, num_out)
        self.num_out = num_out
        print('net v 1.1')
        self.debug = False

        if net_params['autoencoder']:
            self.eval()
            shape_enc = self.forward(torch.randn(shape), True)
            self.train()
            # self.upconv1 = nn.Sequential(*[
            #     View([-1]), nn.Linear(shape_enc[1]*shape_enc[2]*shape_enc[3], self.upconv_chan * self.upconv_size * self.upconv_size),
            #     View([self.upconv_chan, self.upconv_size, self.upconv_size]),
            #     self.non_lin(),
            #     nn.Conv2d(self.upconv_chan, self.upconv_chan, 3, 1, padding=1),
            #     self.norm(self.upconv_chan, affine=True),
            #     self.non_lin()])
            list_upconv2 = []
            chan = shape_enc[1]
            for i in range(2):
                new_chan = int(chan // 2)
                list_upconv2 += [nn.Upsample(scale_factor=2, mode='nearest'),
                                 conv_block(chan, new_chan, kernel_size=kernel, stride=1, padding=1, nonlin=self.nonlin,
                                            batch_norm=self.batch_norm)]
                list_upconv2 += [
                    conv_block(new_chan, new_chan, kernel_size=kernel, stride=1, padding=1, nonlin=self.nonlin,
                               batch_norm=self.batch_norm)]
                chan = new_chan

            self.upconv2 = nn.Sequential(*list_upconv2)

            self.upconv_rec = nn.Sequential(*[
                conv_block(chan, 3, kernel_size=kernel, stride=1, padding=1, nonlin=Identity,
                           batch_norm=self.batch_norm),
                Interpolate((shape[2], shape[3]), mode='bilinear'),
                nn.Tanh()])

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

        if self.net_params['random_pad']:
            x = torch.cat((torch.randn_like(x), x, torch.randn_like(x)), dim=2)
            x = torch.cat((torch.randn_like(x), x, torch.randn_like(x)), dim=3)
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
        if self.net_params['autoencoder']:
            x_decoded = self.decoding(x)
        if self.debug:
            print(x.shape)
        if self.net_params['random_pad']:
            n_features = x.shape[3]
            x = x[:, :, n_features // 4:-n_features // 4, n_features // 4:-n_features // 4]
            # print(x.shape)
        x = self.last_pool(x, x.shape[1])
        if self.debug:
            print(x.shape)
        x = x.view(x.shape[0], -1)
        if self.debug:
            print(x.shape)
        x = self.classify(x)
        if self.debug:
            print(x.shape)
        if self.net_params['autoencoder']:
            return x, x_decoded
        return x
