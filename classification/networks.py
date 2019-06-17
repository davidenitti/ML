import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_out=10, net_params=None):
        super(Net, self).__init__()

        self.net_params = net_params
        self.nonlin = getattr(nn, net_params['non_linearity'])

        self.last_pool = getattr(F, net_params['last_pool'])

        if net_params['padding']:
            self.padding = 1
        else:
            self.padding = 0

        self.conv1 = nn.Sequential(*[nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=self.padding),
                                     nn.BatchNorm2d(64),
                                     self.nonlin(),
                                     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=self.padding),
                                     nn.BatchNorm2d(64),
                                     self.nonlin()
                                     ])
        self.conv2 = nn.Sequential(*[nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=self.padding),
                                     nn.BatchNorm2d(64),
                                     self.nonlin(),
                                     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=self.padding),
                                     nn.BatchNorm2d(64),
                                     self.nonlin()
                                     ])
        self.conv3 = nn.Sequential(
            *[nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=self.padding),
              nn.BatchNorm2d(64),
              self.nonlin(),
              nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=self.padding),
              nn.BatchNorm2d(64),
              self.nonlin()
              ])
        self.fc1 = nn.Sequential(*[nn.Linear(64, 64), nn.BatchNorm1d(64), self.nonlin()])
        self.fc2 = nn.Linear(64, num_out)

        self.num_out = num_out
        # self.emb = torch.empty(1,num_classes, num_out).uniform_(-1, 1)

    def forward(self, x):
        if self.net_params['random_pad']:
            #if self.training:
            x = torch.cat((torch.randn_like(x), x, torch.randn_like(x)), dim=2)
            x = torch.cat((torch.randn_like(x), x, torch.randn_like(x)), dim=3)
            # else:
            #     x = torch.cat((torch.zeros_like(x), x, torch.zeros_like(x)), dim=2)
            #     x = torch.cat((torch.zeros_like(x), x, torch.zeros_like(x)), dim=3)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv3(x)
        #print(x.shape)
        if self.net_params['random_pad']:
            n_features = x.shape[3]
            x = x[:,:,n_features//4:-n_features//4,n_features//4:-n_features//4]
            #print(x.shape)
        x = self.last_pool(x, x.shape[1])
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        return x