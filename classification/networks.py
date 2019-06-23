import torch
from torch import nn
import torch.nn.functional as F

try:
    from .modules import ConvBlock
except:
    from modules import ConvBlock

class Net(nn.Module):
    def __init__(self, num_out=10, net_params=None):
        super(Net, self).__init__()

        self.net_params = net_params
        self.nonlin = getattr(nn, net_params['non_linearity'])
        self.base = net_params['base']
        self.last_pool = getattr(F, net_params['last_pool'])

        if net_params['padding']:
            self.padding = 1
        else:
            self.padding = 0

        self.conv1 = nn.Sequential(*[
            ConvBlock(in_channels=3, out_channels=self.base, kernel_size=3, stride=1, padding=self.padding,
                      nonlin=self.nonlin),
            ConvBlock(in_channels=self.base, out_channels=self.base, kernel_size=3, stride=1, padding=self.padding,
                      nonlin=self.nonlin)
        ])
        self.conv2 = nn.Sequential(*[
            ConvBlock(in_channels=self.base, out_channels=self.base * 2, kernel_size=3, stride=1, padding=self.padding,
                      nonlin=self.nonlin),
            ConvBlock(in_channels=self.base * 2, out_channels=self.base * 4, kernel_size=3, stride=1,
                      padding=self.padding, nonlin=self.nonlin)
        ])
        self.conv3 = ConvBlock(in_channels=self.base * 4, out_channels=self.base * 8, kernel_size=3, stride=1,
                               padding=self.padding, nonlin=self.nonlin)
        self.conv4 = nn.Sequential(
            *[ConvBlock(in_channels=self.base * 8, out_channels=self.base * 16, kernel_size=3, stride=1,
                        padding=self.padding, nonlin=self.nonlin)
              # ,
              # ConvBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=self.padding)
              ])
        self.classify = nn.Linear(self.base * 16, num_out)
        self.num_out = num_out
        print('net v 1.1')
        self.debug = False

    def forward(self, x):
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

        x = self.conv3(x)

        if self.debug:
            print(x.shape)

        x = self.conv4(x)

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
        return x
