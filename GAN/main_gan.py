from argparse import ArgumentParser
from typing import Any
import time
import os, math
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from GAN.gan import DCGAN
from torchvision import transforms as transform_lib
from torchvision.datasets import LSUN, MNIST, ImageFolder

from GAN import utils
#import os
#os.environ['CUDA_VISIBLE_DEVICES']='0'


def get_args(args_list=None):
    parser = ArgumentParser(description='GAN')
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--dataset_type", default="face", type=str, choices=["face", "lsun", "mnist"])
    parser.add_argument("--image_size", default=128, type=int)
    parser.add_argument("--num_workers", default=0 if __name__ == "__main__" else 8, type=int)

    parser.add_argument('--dataset', default="/mnt/teradisk/davide/datasets/celeba/",
                        help='dataset path e.g. https://drive.google.com/open?id=0BxYys69jI14kYVM3aVhKS1VhRUk')
    parser.add_argument('--res_dir', default='./', help='result dir')
    parser.add_argument('--net_params', default={}, type=dict, help='net_params')

    parser.add_argument("--beta1", default=0.0, type=float)
    parser.add_argument("--latent_dim", default=512, type=int)
    parser.add_argument("--loss", default="rals", type=str, choices=["rals", "dcgan",'wgangp'])
    parser.add_argument("--length", default=1, type=float)
    parser.add_argument("--weight_decay", default=0.00000, type=float)
    parser.add_argument("--l2_loss_weight", default=1, type=float)

    parser.add_argument("--use_tpu", action="store_true")
    parser.add_argument("--use_std", action="store_true")

    parser.add_argument("--custom_conv", action="store_true")

    parser.add_argument("--use_avg", action="store_true")
    parser.add_argument("--norm_disc", default="Identity")  # e.g., Identity PixelNorm2d BatchNorm2d

    parser.add_argument("--version", default=3, type=float)
    parser.add_argument("--name", default="newdiscr")
    parser.add_argument("--speed_transition", default=50000, type=float)

    args = parser.parse_args(args_list)

    return args


def main(args=None, callback=None, upload_checkpoint=False):
    args.feature_maps_gen = args.latent_dim
    args.feature_maps_disc = args.latent_dim
    pl.seed_everything(1234)

    if args.dataset_type == "face":
        dataset = ImageFolder(root=args.dataset,
                              transform=transform_lib.Compose([
                                  transform_lib.Resize(args.image_size),
                                  transform_lib.CenterCrop(args.image_size),
                                  transform_lib.ToTensor(),
                                  transform_lib.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ]))
        image_channels = 3
    elif args.dataset_type == "lsun":
        transforms = transform_lib.Compose([
            transform_lib.Resize(args.image_size),
            transform_lib.CenterCrop(args.image_size),
            transform_lib.ToTensor(),
            transform_lib.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = LSUN(root=args.dataset, classes=["bedroom_train"], transform=transforms)
        image_channels = 3
    elif args.dataset_type == "mnist":
        transforms = transform_lib.Compose([
            transform_lib.Resize(args.image_size),
            transform_lib.ToTensor(),
            transform_lib.Normalize((0.5,), (0.5,)),
        ])
        dataset = MNIST(root=args.dataset, download=True, transform=transforms)
        image_channels = 1

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    version = 0
    if args.use_tpu:
        tpu_cores = 8
        gpus = None
        use_tpu_string = ""
    else:
        tpu_cores = None
        gpus = 1
        use_tpu_string = ""
    if args.use_avg:
        use_avg = "avg"
    else:
        use_avg = "noavg"
    use_std = "with_std" if args.use_std else "no_std"
    dirpath = "{}_loss_{}_latent_{}_decay{}_v{}{}_l2_loss_weight{}_{}_{}_b{}_{}".format(args.name, args.loss, args.latent_dim,
                                                                                    args.weight_decay,
                                                                                    args.version,
                                                                                    use_tpu_string, args.l2_loss_weight,
                                                                                    use_avg, args.norm_disc,
                                                                                    args.beta1,use_std)
    print(dirpath)
    resume_from_checkpoint = os.path.join(args.res_dir, dirpath,
                                          "version_{}".format(version), 'checkpoints', 'last.ckpt')
    if not os.path.exists(resume_from_checkpoint):
        resume_from_checkpoint = None
    if args.loss == "rals":
        args.learning_rate = 0.0004
    elif args.loss == "dcgan":
        args.learning_rate = 0.0002
    elif args.loss == "wgangp":
        if args.custom_conv:
            args.learning_rate = 0.001
        else:
            args.learning_rate = 0.00025
    print(vars(args))

    model = DCGAN(**vars(args), image_channels=image_channels)

    callbacks = [
        ModelCheckpoint(filename='last', save_last=True),
        utils.TensorboardGenerativeModelImageSampler(length=args.length, num_samples=9, normalize=True, nrow=3),
        utils.LatentDimInterpolator(range_start=-args.length, range_end=args.length,
                                    interpolate_epoch_interval=1,
                                    normalize=True, callback=callback)
    ]
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.res_dir, name=dirpath, version=version)
    print('starting trainer')
    trainer = pl.Trainer(tpu_cores=tpu_cores, gpus=gpus, logger=tb_logger, resume_from_checkpoint=resume_from_checkpoint,
                         callbacks=callbacks, checkpoint_callback=True, max_epochs=100)
    trainer.fit(model, dataloader)

def get_experiment(name):
    experiments = {
        'ral0':"--batch_size 48 --l2_loss_weight 0.1 --version 2.1 --loss rals --image_size 64"
               "--weight_decay 0.000001 --beta1 0.5 --use_std --use_avg --norm_disc Identity",
        'ral3':"--batch_size 64 --l2_loss_weight 0. --version 2.1 --loss rals --image_size 64"
               "--weight_decay 0.000001 --beta1 0. --use_avg --norm_disc BatchNorm2d",
        'ral4':"--batch_size 64 --l2_loss_weight 0. --version 2.1 --loss rals --image_size 64"
               "--weight_decay 0.000001 --beta1 0. --use_avg --norm_disc Identity",
        'ral1':"--batch_size 64 --l2_loss_weight 0.2 --version 3 --loss rals"
               "--weight_decay 0.0 --beta1 0.0 --use_std --use_avg --norm_disc Identity",
        'ral2': "--batch_size 64 --l2_loss_weight 0 --version 3 --loss rals"
                "--weight_decay 0.0 --beta1 0.0 --use_std --use_avg --norm_disc Identity",
        'wgan1_new': "--batch_size 32 --l2_loss_weight 0 --version 3 --loss wgangp"
                " --weight_decay 0.0 --beta1 0. --use_std --use_avg --norm_disc Identity --speed_transition 25000",
        'wgan1_slow': "--batch_size 40 --l2_loss_weight 0 --version 3 --loss wgangp"
                " --weight_decay 0.0 --beta1 0.0 --use_std --use_avg --norm_disc Identity",
        'wgan1_custom_big': "--batch_size 32 --l2_loss_weight 0 --version 3 --loss wgangp"
                " --weight_decay 0.0 --beta1 0.0 --use_std --use_avg --norm_disc Identity --custom_conv --speed_transition 60000",
        'wgan1_custom': "--batch_size 48 --l2_loss_weight 0 --version 3 --loss wgangp"
            " --weight_decay 0.0 --beta1 0.0 --use_std --use_avg --norm_disc Identity --custom_conv --speed_transition 50000",
        'wgan_custom': "--batch_size 80 --l2_loss_weight 0 --version 3 --loss wgangp"
                " --weight_decay 0.0 --beta1 0 --use_std --use_avg --norm_disc Identity --custom_conv --speed_transition 20000"
    }
    str_exp = experiments[name] + ' --name ' + name
    return str_exp.split(' ')

if __name__ == "__main__":
    base_dir_res = "/home/davide/results/GAN_face"#"/home/davide/Dropbox/Apps/davide_colab/results/GAN_face_gpu/output/"#
    base_dir_dataset = "/mnt/teradisk/davide/datasets/celeba/"
    list_args = ['--dataset', base_dir_dataset,
                 '--res_dir', base_dir_res]
    list_args += get_experiment('wgan_custom')
    args = get_args(list_args)
    main(args)
