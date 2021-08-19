from typing import Optional, Tuple, List
import numpy as np
import os
import torch
from torch import Tensor
from pytorch_lightning import Callback, LightningModule, Trainer
import time,random
import torchvision

def sampler(batch_size, dim, device, length):
    return torch.randn((batch_size, dim), device=device)
#    return torch.rand(batch_size, dim, device=device) * 2 * length - length

class TensorboardGenerativeModelImageSampler(Callback):
    """
    Generates images and logs to tensorboard.
    Your model must implement the ``forward`` function for generation

    Requirements::

        # model must have img_dim arg
        model.img_dim = (1, 28, 28)

        # model forward must work for sampling
        z = torch.rand(batch_size, latent_dim)
        img_samples = your_model(z)

    Example::

        from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler

        trainer = Trainer(callbacks=[TensorboardGenerativeModelImageSampler()])
    """

    def __init__(
        self,
        num_samples: int = 3,
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        norm_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
        length: int = 2,
    ) -> None:
        """
        Args:
            num_samples: Number of images displayed in the grid. Default: ``3``.
            nrow: Number of images displayed in each row of the grid.
                The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
            padding: Amount of padding. Default: ``2``.
            normalize: If ``True``, shift the image to the range (0, 1),
                by the min and max values specified by :attr:`range`. Default: ``False``.
            norm_range: Tuple (min, max) where min and max are numbers,
                then these numbers are used to normalize the image. By default, min and max
                are computed from the tensor.
            scale_each: If ``True``, scale each image in the batch of
                images separately rather than the (min, max) over all images. Default: ``False``.
            pad_value: Value for the padded pixels. Default: ``0``.
        """

        super().__init__()
        self.num_samples = num_samples
        self.nrow = nrow
        self.padding = padding
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value
        self.length = length

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule,outputs,
        batch,        batch_idx: int,        dataloader_idx: int) -> None:
        if trainer.global_step % 500 != 0 or trainer.global_step==0:
            return
        z = sampler(self.num_samples, pl_module.hparams.latent_dim, pl_module.device, self.length)

        # generate images
        with torch.no_grad():
            pl_module.eval()
            img = pl_module(z)
            if isinstance(img,list):
                images = []
                for img_i in img:
                    if img_i is None:
                        images.append(None)
                    else:
                        images.append(torch.nn.functional.interpolate(img_i, size=(256, 256)))
            else:
                images = torch.nn.functional.interpolate(img, size=(256,256))
            pl_module.train()

        if isinstance(img, list):
            for i, images_i in enumerate(images):
                if images_i is None:
                    continue
                if len(images_i.size()) == 2:
                    img_dim = pl_module.img_dim
                    images_i = images_i.view(self.num_samples, *img_dim)
                grid = torchvision.utils.make_grid(
                    tensor=images_i,
                    nrow=self.nrow,
                    padding=self.padding,
                    normalize=self.normalize,
                    range=self.norm_range,
                    scale_each=self.scale_each,
                    pad_value=self.pad_value,
                )
                str_title = f"{pl_module.__class__.__name__}_images{2**(i+2)}"
                trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)

        else:
            if len(images.size()) == 2:
                img_dim = pl_module.img_dim
                images = images.view(self.num_samples, *img_dim)

            grid = torchvision.utils.make_grid(
                tensor=images,
                nrow=self.nrow,
                padding=self.padding,
                normalize=self.normalize,
                range=self.norm_range,
                scale_each=self.scale_each,
                pad_value=self.pad_value,
            )
            str_title = f"{pl_module.__class__.__name__}_images"
            trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)

            grid_real = torchvision.utils.make_grid(
                tensor=batch[0],
                nrow=self.nrow,
                padding=self.padding,
                normalize=self.normalize,
                range=self.norm_range,
                scale_each=self.scale_each,
                pad_value=self.pad_value
            )
            str_title = f"{pl_module.__class__.__name__}_real_images"
            trainer.logger.experiment.add_image(str_title, grid_real, global_step=trainer.global_step)

        time.sleep(random.random())
        if not os.path.exists(os.path.join(trainer.log_dir,'images')):
            os.makedirs(os.path.join(trainer.log_dir,'images'),exist_ok=True)
        torchvision.utils.save_image(grid.cpu(), os.path.join(trainer.log_dir,
                                                              'images', 'sampled{:07d}.png'.format(trainer.global_step)))

class LatentDimInterpolator(Callback):
    """
    Interpolates the latent space for a model by setting all dims to zero and stepping
    through the first two dims increasing one unit at a time.

    Default interpolates between [-5, 5] (-5, -4, -3, ..., 3, 4, 5)

    Example::

        from pl_bolts.callbacks import LatentDimInterpolator

        Trainer(callbacks=[LatentDimInterpolator()])
    """

    def __init__(
        self,
        interpolate_epoch_interval: int = 20,
        range_start: int = -1,
        range_end: int = 1,
        steps: int = 11,
        num_samples: int = 2,
        normalize: bool = True,
        callback=None
    ):
        """
        Args:
            interpolate_epoch_interval: default 20
            range_start: default -5
            range_end: default 5
            steps: number of step between start and end
            num_samples: default 2
            normalize: default True (change image to (0, 1) range)
        """
        super().__init__()
        self.interpolate_epoch_interval = interpolate_epoch_interval
        self.range_start = range_start
        self.range_end = range_end
        self.num_samples = num_samples
        self.normalize = normalize
        self.steps = steps
        self.callback=callback

    def on_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.callback is not None and trainer.global_step>10:
            self.callback(False)

    def on_batch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.global_step % 500 != 0 or trainer.global_step==0:
            return
        if (trainer.current_epoch + 1) % self.interpolate_epoch_interval == 0:
            images = self.interpolate_latent_space(
                pl_module,
                latent_dim=pl_module.hparams.latent_dim  # type: ignore[union-attr]
            )
            images = torch.cat(images, dim=0)  # type: ignore[assignment]

            num_rows = self.steps
            grid = torchvision.utils.make_grid(images, nrow=num_rows, normalize=self.normalize)
            str_title = f'{pl_module.__class__.__name__}_latent_space'
            trainer.logger.experiment.add_image(str_title, grid, global_step=trainer.global_step)
            if not os.path.exists(os.path.join(trainer.log_dir,'images')):
                os.makedirs(os.path.join(trainer.log_dir,'images'))
            torchvision.utils.save_image(grid.cpu(), os.path.join(trainer.log_dir,
                                                          'images', 'latent{:07d}.png'.format(trainer.global_step)))
    def interpolate_latent_space(self, pl_module: LightningModule, latent_dim: int) -> List[Tensor]:
        images = []
        with torch.no_grad():
            pl_module.eval()
            for z1 in np.linspace(self.range_start, self.range_end, self.steps):
                for z2 in np.linspace(self.range_start, self.range_end, self.steps):
                    # set all dims to zero
                    z = torch.zeros(self.num_samples, latent_dim, device=pl_module.device)

                    # set the fist 2 dims to the value
                    z[:, 0] = torch.tensor(z1)
                    z[:, 1] = torch.tensor(z2)

                    # sample
                    # generate images
                    img = pl_module(z)
                    if isinstance(img, list):
                        idx = -1
                        img_tmp = img[idx]
                        while img_tmp is None and abs(idx)<len(img):
                            idx -= 1
                            img_tmp = img[idx]
                        img = img_tmp
                        if img is None:
                            img = torch.zeros((self.num_samples, *pl_module.img_dim))
                        else:
                            img = torch.nn.functional.interpolate(img, size=(pl_module.img_dim[-2],pl_module.img_dim[-1]))
                    if len(img.size()) == 2:
                        img = img.view(self.num_samples, *pl_module.img_dim)

                    img = img[0]
                    img = img.unsqueeze(0)
                    images.append(img)

        pl_module.train()
        return images


def compute_gradient_penalty(D, real_samples, fake_samples, idx=None):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=real_samples.device)# Tensor(np.random.random((real_samples.size(0), 1, 1, 1)),device=real_samples.device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    d_interpolates = D(interpolates, idx)
    fake = torch.ones(real_samples.shape[0], 1, device=real_samples.device)

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
