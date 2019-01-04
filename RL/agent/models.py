import torch.nn as nn
import torch.nn.functional as F


def identity(inp):
    return inp


def activ(activation):
    if activation is not None:
        return getattr(F, activation)
    else:
        return identity


class ScaledIdentity(nn.Module):
    def __init__(self, scale=1.0):
        super(ScaledIdentity, self).__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class DenseNet(nn.Module):
    def __init__(self, dense_layers, input_shape, scale=1.0, final_act=False, activation='relu', batch_norm=False):
        super(DenseNet, self).__init__()
        self.dense = []

        self.dense_layers = [input_shape] + dense_layers
        if batch_norm:
            self.bn = [nn.BatchNorm1d(self.dense_layers[0])]
        for i in range(len(self.dense_layers) - 1):
            self.dense.append(nn.Linear(self.dense_layers[i], self.dense_layers[i + 1]))
            if batch_norm:
                self.bn.append(nn.BatchNorm1d(self.dense_layers[i + 1]))
        self.dense = nn.ModuleList(self.dense)
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.ModuleList(self.bn)
        self.scale = scale
        self.final_act = final_act
        self.act = activ(activation)

    def forward(self, x):
        x = x * self.scale
        if self.batch_norm:
            x = self.bn[0](x)
        for i, d in enumerate(self.dense[:-1]):
            x = d(x)
            if self.batch_norm:
                x = self.bn[i + 1](x)
            x = self.act(x)
        x = self.dense[-1](x)
        if self.final_act:
            if self.batch_norm:
                x = self.bn[-1](x)
            x = self.act(x)
        return x


# TODO add batch_norm
class ConvNet(nn.Module):
    def __init__(self, convlayers, input_shape, scale=1.0, activation='relu', batch_norm=False):
        super(ConvNet, self).__init__()
        self.conv = []
        input_channel = input_shape[0]
        for i in range(len(convlayers)):
            self.conv.append(nn.Conv2d(input_channel, convlayers[i][3],
                                       kernel_size=convlayers[i][0], stride=convlayers[i][2]))
            input_channel = convlayers[i][3]
        self.conv = nn.ModuleList(self.conv)
        self.scale = scale
        self.act = activ(activation)
        if batch_norm:
            raise NotImplemented

    def forward(self, x):
        x = x * self.scale
        for c in self.conv:
            x = c(x)
            x = self.act(x)
        x = x.view(x.shape[0], -1)
        return x
