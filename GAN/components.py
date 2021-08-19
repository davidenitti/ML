# Based on https://github.com/pytorch/examples/blob/master/dcgan/main.py
import torch
import math
from torch import nn
from core.modules import View, PixelNorm2d
import core.modules
import random
from GAN.modules import EqualizedConv2d


def weight_formula(i, idx, speed=2):
    if idx > i:
        w = max(0, min(1, 1 - abs(i - idx) * speed))
    else:
        w = max(0, min(1, speed - abs(i - idx) * speed))
    return w


class FirstLayer(nn.Module):

    def __init__(self, latent_dim, size_out, out_channels) -> None:
        """
        Args:
            latent_dim: Dimension of the latent space
            feature_maps: Number of feature maps to use
            image_channels: Number of channels of the images from the dataset
        """
        super().__init__()
        self.size_out = size_out
        self.linear = nn.Linear(latent_dim, size_out * size_out * out_channels)
        self.out_channels = out_channels
        self.bn = nn.Sequential(PixelNorm2d(self.out_channels), nn.LeakyReLU(0.2))

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        l1 = self.linear(noise.view(noise.shape[0], -1))
        l1_view = l1.view(noise.shape[0], self.out_channels, self.size_out, self.size_out)
        return self.bn(l1_view)


class DCGANGenerator(nn.Module):

    def __init__(self, latent_dim: int, feature_maps: int, image_channels: int, version: float, size: int,
                 custom_conv: bool) -> None:
        """
        Args:
            latent_dim: Dimension of the latent space
            feature_maps: Number of feature maps to use
            image_channels: Number of channels of the images from the dataset
        """
        super().__init__()
        self.num_layers = int(math.log2(size)) - 2
        self.version = version
        if version == 1:
            self.gen = nn.Sequential(
                FirstLayer(latent_dim, 4, feature_maps),
                self._make_gen_block(feature_maps, feature_maps, custom_conv=custom_conv),  # 8x8
                self._make_gen_block(feature_maps, feature_maps // 2, custom_conv=custom_conv),  # 16x16
                self._make_gen_block(feature_maps // 2, feature_maps // 4, custom_conv=custom_conv),  # 32x32
                self._make_gen_block(feature_maps // 4, feature_maps // 8, scale=1, custom_conv=custom_conv),  # 32x32
                self._make_gen_block(feature_maps // 8, image_channels, last_block=True, custom_conv=custom_conv)  # 64x64
            )
        elif version == 2.1:
            gen_layers = [
                FirstLayer(latent_dim, 4, feature_maps),
                self._make_gen_block(feature_maps, feature_maps // 2, custom_conv=custom_conv),  # 8x8
                self._make_gen_block(feature_maps // 2, feature_maps // 4, custom_conv=custom_conv),  # 16x16
                self._make_gen_block(feature_maps // 4, feature_maps // 4, scale=1, custom_conv=custom_conv),
                self._make_gen_block(feature_maps // 4, feature_maps // 8, custom_conv=custom_conv),  # 32x32
                self._make_gen_block(feature_maps // 8, feature_maps // 8, scale=1, custom_conv=custom_conv),
                self._make_gen_block(feature_maps // 8, image_channels, last_block=True, custom_conv=custom_conv)  # 64x64
            ]
            self.gen = nn.Sequential(*gen_layers)
        elif version == 3:
            gen_layers = [nn.Sequential(FirstLayer(latent_dim, 4, feature_maps),
                                        self._make_gen_block(feature_maps, feature_maps, scale=1, custom_conv=custom_conv))]
            out_layers = [self._make_gen_block(feature_maps, image_channels, scale=1, last_block=True, custom_conv=custom_conv)]
            num_features = feature_maps
            for layer in range(self.num_layers):
                out_features = num_features if layer <= 3 else num_features // 2
                gen_layers += [nn.Sequential(
                    self._make_gen_block(num_features, out_features, custom_conv=custom_conv),
                    self._make_gen_block(out_features, out_features, scale=1, custom_conv=custom_conv))]
                out_layers += [
                    self._make_gen_block(out_features, image_channels, scale=1, last_block=True, custom_conv=custom_conv)
                ]
                num_features = out_features
            self.gen = nn.ModuleList(gen_layers)
            self.out_layers = nn.ModuleList(out_layers)
        else:
            raise NotImplementedError

    @staticmethod
    def _make_gen_block(
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            scale: int = 2,
            bias: bool = True,
            last_block: bool = False,
            use_tanh=False,
            custom_conv=False
    ) -> nn.Sequential:
        if custom_conv:
            conv = EqualizedConv2d
            if not bias:
                print('_make_gen_block: setting bias to True')
            bias = True
        else:
            conv = nn.Conv2d
        if use_tanh:
            last_act = nn.Tanh()
        else:
            last_act = nn.Identity()
        if scale > 1:
            upscale = nn.Upsample(scale_factor=scale)
        else:
            upscale = nn.Identity()
        if not last_block:
            gen_block = nn.Sequential(
                upscale,
                conv(in_channels, out_channels, kernel_size, 1, kernel_size // 2, bias=bias),
                PixelNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )
        else:
            gen_block = nn.Sequential(
                upscale,
                conv(in_channels, out_channels, kernel_size, 1, kernel_size // 2, bias=bias),
                last_act
            )

        return gen_block

    def forward(self, noise: torch.Tensor, idx: float):
        if self.version < 3:
            return torch.clamp(self.gen(noise), -1, 1)
        else:
            out = None
            layer = noise
            assert self.num_layers == len(self.gen) - 1
            lower_idx = min(self.num_layers, math.floor(idx))
            higher_idx = min(self.num_layers, math.ceil(idx))
            debug = random.random() < 0.002 and False
            sum_w = 0.0
            for l in range(higher_idx + 1):
                if debug:
                    print(l)
                layer = self.gen[l](layer)
                if lower_idx <= l <= higher_idx:
                    w = weight_formula(l, idx)
                    sum_w += w
                    if out is None:
                        out = w * self.out_layers[l](layer)
                    else:
                        new_out = self.out_layers[l](layer)
                        out = torch.nn.functional.interpolate(out, scale_factor=2, mode='nearest') + w * new_out
            assert sum_w > 0
            return out / sum_w


class DCGANDiscriminator(nn.Module):

    def __init__(self, feature_maps: int, image_channels: int, version: float, size: int, use_avg: bool,
                 norm: str, use_std: bool, custom_conv: bool) -> None:
        """
        Args:
            feature_maps: Number of feature maps to use
            image_channels: Number of channels of the images from the dataset
        """
        super().__init__()
        self.num_layers = int(math.log2(size)) - 2
        self.version = version
        self.use_std = use_std
        if self.version < 3:
            num_features = feature_maps // (2 ** (self.num_layers - 1))
            self.disc = [
                self._make_disc_block(image_channels, num_features, use_avg=use_avg, norm=norm, custom_conv=custom_conv)]
            for l in range(self.num_layers - 1):
                # self.disc.append(self._make_disc_block(num_features, num_features, use_avg=use_avg, stride=1,norm=norm))
                self.disc.append(
                    self._make_disc_block(num_features, num_features * 2, use_avg=use_avg, norm=norm, custom_conv=custom_conv))
                num_features *= 2
            assert num_features == feature_maps
            self.disc.append(self._make_disc_block(num_features, 1, kernel_size=4,
                                                   stride=1, padding=0, last_block=True, custom_conv=custom_conv))

            self.disc = nn.Sequential(*self.disc)
        else:
            self.avg2x = nn.AvgPool2d(2)
            if use_std:
                chan_std = 1
            else:
                chan_std = 0
            num_features = feature_maps  # // (2 ** (self.num_layers - 3))
            self.disc = []
            self.from_rgb = [self._make_disc_block(image_channels, feature_maps, stride=1, use_avg=use_avg, norm=norm,
                                                   custom_conv=custom_conv)]

            self.out = nn.Sequential(
                self._make_disc_block(feature_maps + chan_std, feature_maps, stride=1, use_avg=use_avg, norm=norm,
                                      custom_conv=custom_conv),
                self._make_disc_block(feature_maps, 1, kernel_size=4, stride=1, padding=0, last_block=True,
                                      custom_conv=custom_conv))

            for num_l in range(self.num_layers):
                num_features //= 2
                self.from_rgb.append(self._make_disc_block(image_channels, num_features, stride=1, use_avg=use_avg, norm=norm,
                                                           custom_conv=custom_conv))
                single_disc = []
                single_disc.append(self._make_disc_block(num_features, num_features, stride=1, use_avg=use_avg, norm=norm,
                                                         custom_conv=custom_conv))
                single_disc.append(
                    self._make_disc_block(num_features, num_features * 2, use_avg=use_avg, norm=norm, custom_conv=custom_conv))

                self.disc.append(nn.Sequential(*single_disc))

            self.disc = nn.ModuleList(self.disc)
            self.from_rgb = nn.ModuleList(self.from_rgb)

    @staticmethod
    def _make_disc_block(
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 2,
            padding: int = 1,
            bias: bool = True,
            last_block: bool = False,
            use_avg=False,
            custom_conv=False,
            norm=""
    ) -> nn.Sequential:
        if use_avg:
            stride_conv = 1
            if stride > 1:
                downscale = nn.AvgPool2d(stride)
            else:
                downscale = nn.Identity()
        else:
            downscale = nn.Identity()
            stride_conv = stride
        if custom_conv:
            conv = EqualizedConv2d
            if not bias:
                print('_make_gen_block: setting bias to True')
            bias = True
        else:
            conv = nn.Conv2d
        if not last_block:
            if hasattr(nn, norm):
                norm_layer = getattr(nn, norm)
            else:
                norm_layer = getattr(core.modules, norm)
            disc_block = nn.Sequential(
                conv(in_channels, out_channels, kernel_size, stride_conv, padding, bias=bias),
                norm_layer(out_channels),
                nn.LeakyReLU(0.2),
                downscale
            )
        else:
            disc_block = nn.Sequential(
                conv(in_channels, out_channels, kernel_size, stride, padding, bias=bias)  # ,
                # nn.Sigmoid(),
            )

        return disc_block

    def forward(self, x, idx):
        if self.version < 3:
            return self.disc(x).view(x.shape[0], 1)
        else:
            lower_idx = min(self.num_layers, math.floor(idx))
            higher_idx = min(self.num_layers, math.ceil(idx))
            w1 = weight_formula(lower_idx, idx)
            w2 = weight_formula(higher_idx, idx)
            sum_w = w1 + w2
            w1 /= sum_w
            w2 /= sum_w
            if random.random() < 0.001:
                print('idx', lower_idx, idx, higher_idx, 'w', w1, w2)
            if lower_idx == higher_idx:
                o = self.from_rgb[lower_idx](x)
                # print(o.shape)
                if higher_idx > 0:
                    o = self.disc[higher_idx - 1](o)
            else:
                o = w1 * self.from_rgb[lower_idx](self.avg2x(x)) + w2 * self.disc[higher_idx - 1](self.from_rgb[higher_idx](x))
            # print(lower_idx, higher_idx, o.shape)
            for l in range(higher_idx - 2, -1, -1):
                o = self.disc[l](o)
                # print(lower_idx,higher_idx,o.shape)
            if self.use_std:
                o = core.modules.miniBatchStdDev(o)
            o = self.out(o)
            return o.view(x.shape[0], 1)
