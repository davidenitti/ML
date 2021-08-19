from typing import Any
import os, math
import pytorch_lightning as pl
import torch
from torch import nn
from GAN.components import DCGANDiscriminator, DCGANGenerator, weight_formula

from GAN import utils




class DCGAN(pl.LightningModule):
    """
    DCGAN implementation.

    Example::

        from pl_bolts.models.gan import DCGAN

        m = DCGAN()
        Trainer(gpus=2).fit(m)

    Example CLI::

        # mnist
        python dcgan_module.py --gpus 1

        # cifar10
        python dcgan_module.py --gpus 1 --dataset cifar10 --image_channels 3
    """

    def __init__(
            self,
            beta1: float = 0.5,
            feature_maps_disc: int = 64,
            image_channels: int = 1,
            latent_dim: int = 100,
            lambda_gp: float = 10,
            decay: float = 0.0,
            loss: str = "",
            length: int = 2,
            version: float = None,
            l2_loss_weight: float = None,
            speed_transition = 40000,
            **kwargs: Any,
    ) -> None:
        """
        Args:
            beta1: Beta1 value for Adam optimizer
            feature_maps_gen: Number of feature maps to use for the generator
            feature_maps_disc: Number of feature maps to use for the discriminator
            image_channels: Number of channels of the images from the dataset
            latent_dim: Dimension of the latent space
            learning_rate: Learning rate
        """
        super().__init__()
        self.save_hyperparameters()
        self.generator = self._get_generator()
        self.discriminator = self._get_discriminator()
        self.loss = loss
        self.version = version
        self.args = kwargs
        self.img_dim = (image_channels, kwargs['image_size'], kwargs['image_size'])
        if loss == "rals":
            self.dic_loss_func = self._get_disc_loss_lsregan
            self.gen_loss_func = self._get_gen_loss_lsregan
        elif loss == "wgangp":
            self.dic_loss_func = self._get_disc_loss_wgangp
            self.gen_loss_func = self._get_gen_loss_wgangp
        elif loss == "dcgan":
            self.dic_loss_func = self._get_disc_loss
            self.gen_loss_func = self._get_gen_loss
        self.length = length
        self.decay = decay
        self.l2_loss_weight = l2_loss_weight
        self.lambda_gp = lambda_gp
        self.speed_transition = speed_transition

    @property
    def automatic_optimization(self) -> bool:
        return False

    def _get_generator(self) -> nn.Module:
        generator = DCGANGenerator(self.hparams.latent_dim, self.hparams.feature_maps_gen,
                                   self.hparams.image_channels, self.hparams.version, self.hparams.image_size, self.hparams.custom_conv)
        if not self.hparams.custom_conv:
            generator.apply(self._weights_init)
        return generator

    def _get_discriminator(self) -> nn.Module:
        discriminator = DCGANDiscriminator(self.hparams.feature_maps_disc, self.hparams.image_channels, self.hparams.version,
                                           self.hparams.image_size, self.hparams.use_avg, self.hparams.norm_disc,
                                           self.hparams.use_std, self.hparams.custom_conv)
        if not self.hparams.custom_conv:
            discriminator.apply(self._weights_init)
        return discriminator

    @staticmethod
    def _weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)

    def configure_optimizers(self):
        lr = self.hparams.learning_rate
        betas = (self.hparams.beta1, 0.99)
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas, weight_decay=self.decay)
        opt_gen = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas, weight_decay=self.decay)
        return opt_disc, opt_gen

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Generates an image given input noise

        Example::

            noise = torch.rand(batch_size, latent_dim)
            gan = GAN.load_from_checkpoint(PATH)
            img = gan(noise)
        """
        noise = noise.view(*noise.shape, 1, 1)
        return self.generator(noise, self.get_idx(self.num_scales))

    def training_step(self, batch, batch_idx):
        d_opt, g_opt = self.optimizers()
        ratio = 1
        real, _ = batch
        self.num_scales = 1
        if self.version >= 3:
            self.num_scales = int(math.log2(real.shape[-1])) - 2
            idx = self.get_idx(self.num_scales)
            lower_idx = min(self.num_scales, math.floor(idx))
            higher_idx = min(self.num_scales, math.ceil(idx))
            w1 = weight_formula(lower_idx, idx)
            w2 = weight_formula(higher_idx, idx)
            sum_w = w1 + w2
            w1 /= sum_w
            w2 /= sum_w

            real = real[:real.shape[0]//max(1,higher_idx)]
            size = real.shape[-1] // (2 ** (self.num_scales-higher_idx))
            real = torch.nn.functional.interpolate(real, size=(size, size), mode="area")
            if lower_idx != higher_idx:
                low_res_real = torch.nn.functional.avg_pool2d(real, (2, 2))
                low_res_real = torch.nn.functional.upsample(low_res_real, scale_factor=2, mode='nearest')
                real = w1 * low_res_real + w2 * real

            if batch_idx % 200 == 0:
                print('current size real', real.shape,'idx',idx,'w',w1,w2)
        if batch_idx % ratio == 0:
            d_opt.zero_grad()
            d_x = self._disc_step(real)
            if self.args['use_tpu']:
                self.manual_backward(d_x,d_opt)
            else:
                self.manual_backward(d_x)
            d_opt.step()

        g_opt.zero_grad()
        g_x = self._gen_step(real)
        if self.args['use_tpu']:
            self.manual_backward(g_x,g_opt)
        else:
            self.manual_backward(g_x)
        g_opt.step()
        if batch_idx % ratio == 0:
            self.log_dict({'g_loss': g_x, 'd_loss': d_x}, prog_bar=True)

    def _disc_step(self, real: torch.Tensor) -> torch.Tensor:
        disc_loss = self.dic_loss_func(real)
        self.log("loss/disc", disc_loss, on_step=True, on_epoch=True)
        return disc_loss

    def _gen_step(self, real: torch.Tensor) -> torch.Tensor:
        gen_loss = self.gen_loss_func(real)
        self.log("loss/gen", gen_loss, on_step=True, on_epoch=True)
        return gen_loss

    def _get_disc_loss(self, real: torch.Tensor) -> torch.Tensor:
        # Train with real
        real_pred = self.discriminator(real, self.get_idx(self.num_scales))
        real_gt = torch.ones_like(real_pred)
        real_loss = torch.nn.functional.binary_cross_entropy_with_logits(real_pred, real_gt)

        # Train with fake
        fake_pred = self._get_fake_pred(real)
        fake_gt = torch.zeros_like(fake_pred)
        fake_loss = torch.nn.functional.binary_cross_entropy_with_logits(fake_pred, fake_gt)

        disc_loss = real_loss + fake_loss

        return disc_loss

    def get_idx(self, max_val):
        idx = self.global_step / self.speed_transition
        if idx >= max_val:
            idx = max_val
        return idx

    def _get_disc_loss_wgangp(self, real: torch.Tensor) -> torch.Tensor:
        # Train with real
        idx = self.get_idx(self.num_scales)
        real_pred = self.discriminator(real, idx)
        # Train with fake
        fake_pred, fake = self._get_fake_pred(real, True)

        gradient_penalty = utils.compute_gradient_penalty(self.discriminator, real, fake, idx)
        disc_loss = (-torch.mean(real_pred) + torch.mean(fake_pred) + self.lambda_gp * gradient_penalty)

        return disc_loss

    def _get_gen_loss_wgangp(self, real: torch.Tensor) -> torch.Tensor:
        # Train with real
        idx = self.get_idx(self.num_scales)
        real_pred = self.discriminator(real, idx)
        # Train with fake
        fake_pred = self._get_fake_pred(real)

        self.log("loss/realfake_diff", torch.mean(real_pred) - torch.mean(fake_pred),
                 on_step=True, on_epoch=True)
        self.log("loss/real_mean", torch.mean(real_pred),
                 on_step=True, on_epoch=True)
        self.log("loss/fake_mean", torch.mean(fake_pred),
                 on_step=True, on_epoch=True)
        gen_loss = (-torch.mean(fake_pred))
        return gen_loss

    def _get_disc_loss_lsregan(self, real: torch.Tensor) -> torch.Tensor:
        if self.version>=3:
            num_layers = self.discriminator.num_layers
            assert num_layers==self.num_scales

        idx = self.get_idx(self.num_scales)
        # Train with real
        real_pred = self.discriminator(real,idx)
        # Train with fake
        fake_pred = self._get_fake_pred(real)
        disc_loss = torch.mean((real_pred - torch.mean(fake_pred) - 1) ** 2) + \
                        torch.mean((fake_pred - torch.mean(real_pred) + 1) ** 2)
        if self.l2_loss_weight > 0:
            if isinstance(real_pred, list):
                l2_loss = 0.0
                sum_w = 0.0
                for i in range(len(real_pred)):
                    if fake_pred[i] is None:
                        assert real_pred[i] is None
                        continue
                    w = weight_formula(i, idx)
                    sum_w += w
                    l2_loss += w * self.l2_loss_weight * (torch.mean(fake_pred[i] ** 2) + torch.mean(real_pred[i] ** 2))
                l2_loss /= sum_w
            else:
                l2_loss = self.l2_loss_weight * (torch.mean(fake_pred ** 2) + torch.mean(real_pred ** 2))
            return disc_loss / 2 + l2_loss
        else:
            return disc_loss / 2

    def _get_gen_loss_lsregan(self, real: torch.Tensor) -> torch.Tensor:
        idx = self.get_idx(self.num_scales)

        # Train with real
        real_pred = self.discriminator(real,idx)
        # Train with fake
        fake_pred = self._get_fake_pred(real)
        if isinstance(real_pred, list):
            gen_loss = 0.0
            sum_w = .0
            for i in range(len(real_pred)):

                if fake_pred[i] is None:
                    assert real_pred[i] is None
                    continue
                w = weight_formula(i, idx)
                sum_w += w
                if self.global_step % 100 == 0:
                    print(2 ** (i + 2), 'idx', idx, 'w', w)
                gen_loss += w * (torch.mean((real_pred[i] - torch.mean(fake_pred[i]) + 1) ** 2) +
                                 torch.mean((fake_pred[i] - torch.mean(real_pred[i]) - 1) ** 2))
                self.log("loss/realfake_diff" + str(2 ** (i + 2)), torch.mean(real_pred[i]) - torch.mean(fake_pred[i]),
                         on_step=True, on_epoch=True)
            gen_loss /= sum_w

        else:
            self.log("loss/realfake_diff", torch.mean(real_pred) - torch.mean(fake_pred),
                     on_step=True, on_epoch=True)
            self.log("loss/real_mean", torch.mean(real_pred),
                     on_step=True, on_epoch=True)
            self.log("loss/fake_mean", torch.mean(fake_pred),
                     on_step=True, on_epoch=True)
            gen_loss = torch.mean((real_pred - torch.mean(fake_pred) + 1) ** 2) + \
                       torch.mean((fake_pred - torch.mean(real_pred) - 1) ** 2)
        if self.l2_loss_weight > 0 and False:  # disabled for generator
            if isinstance(real_pred, list):
                l2_loss = 0.0
                if self.global_step % 50 == 0:
                    print()
                sum_w = .0
                for i in range(len(real_pred)):
                    if fake_pred[i] is None:
                        assert real_pred[i] is None
                        continue
                    w = weight_formula(i, idx)
                    sum_w += w

                    l2_loss += w * self.l2_loss_weight * (torch.mean(fake_pred[i] ** 2) + torch.mean(real_pred[i] ** 2))
                l2_loss /= sum_w
            else:
                l2_loss = self.l2_loss_weight * (torch.mean(fake_pred ** 2) + torch.mean(real_pred ** 2))
            self.log("loss/l2_loss", l2_loss, on_step=True, on_epoch=True)
            return gen_loss / 2 + l2_loss
        else:
            return gen_loss / 2

    def _get_gen_loss(self, real: torch.Tensor) -> torch.Tensor:
        # Train with fake
        fake_pred = self._get_fake_pred(real)
        fake_gt = torch.ones_like(fake_pred)
        gen_loss = torch.nn.functional.binary_cross_entropy_with_logits(fake_pred, fake_gt)

        return gen_loss

    def _get_fake_pred(self, real: torch.Tensor, return_fake: bool = False) -> torch.Tensor:
        if isinstance(real, list):
            for r in real:
                if r is not None:
                    batch_size = r.shape[0]
                    break
        else:
            batch_size = real.shape[0]
        noise = self._get_noise(batch_size, self.hparams.latent_dim)
        fake = self(noise)
        fake_pred = self.discriminator(fake,self.get_idx(self.num_scales))
        if return_fake:
            return fake_pred, fake
        else:
            return fake_pred

    def _get_noise(self, n_samples: int, latent_dim: int) -> torch.Tensor:
        return utils.sampler(n_samples, latent_dim, device=self.device, length=self.length)
