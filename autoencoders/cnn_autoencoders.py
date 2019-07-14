'''
CNN autoencoder applied to faces (for example)
Author: Davide Nitti
'''
import torch
from torch import nn
from torchvision import transforms, datasets
import argparse
import torch.optim as optim
import matplotlib
import time
import json
import torch.nn.functional as F
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.modules import View, CropPad, Identity, Interpolate, ConvBlock, TanhMod, PixelNorm2d
from core.utils import get_lr, load_model, save_model, set_lr

try:
    matplotlib.use("TkAgg")
except:
    print('WARNING: TkAgg not loaded')
import matplotlib.pyplot as plt
import random, os
import numpy as np
from multiprocessing import Process


def start_process(func, args):
    p = Process(target=func, args=args)
    p.start()
    return p


STD = 0.50
try:
    import IPython.display  # for colab compatibility
except:
    plt.ion()


def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return


class Encoder(nn.Module):
    def __init__(self, net_params, size_input):
        super(Encoder, self).__init__()
        self.debug = False

        if net_params['norm'] == 'PixelNorm2d':
            self.norm = PixelNorm2d
        elif net_params['norm'] == 'BatchNorm2d':
            self.norm = nn.BatchNorm2d
        elif net_params['norm'] == 'LocalResponseNorm':
            self.norm = nn.LocalResponseNorm
        elif net_params['norm'] == None:
            self.norm = Identity
        elif net_params['norm'] == 'InstanceNorm2d':
            self.norm = nn.InstanceNorm2d
        else:
            raise NotImplementedError
        self.base = net_params['base']
        self.multiplier_chan = net_params['multiplier_chan']
        self.non_lin = getattr(nn, net_params['non_linearity'])

        self.base_enc = net_params['num_features_encoding']
        self.upconv_chan = net_params['upconv_chan']
        self.upconv_size = net_params['upconv_size']
        max_hw = max(size_input[-2:])
        if size_input[-1] != size_input[-2]:
            self.pad_inp = CropPad(max_hw, max_hw)
            self.crop_inp = CropPad(size_input[-2], size_input[-1])
        else:
            print('no pad/crop')
            self.pad_inp = Identity()
            self.crop_inp = Identity()

        conv_block = ConvBlock
        self.conv1 = conv_block(in_channels=3, out_channels=self.base, kernel_size=3, stride=1,
                                padding=1, nonlin=self.non_lin, batch_norm=self.norm, affine=net_params['affine'])

        list_conv2 = []

        chan = self.base
        for i in range(6):
            new_chan = min(net_params['max_chan'], int(chan * self.multiplier_chan))
            list_conv2 += [conv_block(in_channels=chan, out_channels=chan, kernel_size=3, stride=1,
                                      padding=1, nonlin=self.non_lin, batch_norm=self.norm),
                           conv_block(in_channels=chan, out_channels=chan, kernel_size=3, stride=1,
                                      padding=1, nonlin=self.non_lin, batch_norm=self.norm),
                           conv_block(in_channels=chan, out_channels=new_chan, kernel_size=3, stride=2,
                                      padding=1, nonlin=self.non_lin, batch_norm=self.norm)]
            chan = new_chan
        self.conv2 = nn.ModuleList(list_conv2)

        pre_encoding_shape = self.encoding(torch.zeros(size_input)).shape
        print('pre_encoding_shape', pre_encoding_shape)

        self.conv_enc = nn.Sequential(
            *[View([-1]),
              nn.Linear(pre_encoding_shape[1] * pre_encoding_shape[2] * pre_encoding_shape[3], self.base_enc),
              TanhMod(net_params['scale_tanh'])])

        self.upconv1 = nn.ModuleList([
            View([-1]), nn.Linear(self.base_enc, self.upconv_chan * self.upconv_size * self.upconv_size),
            View([self.upconv_chan, self.upconv_size, self.upconv_size]),
            self.non_lin(),
            conv_block(in_channels=self.upconv_chan, out_channels=self.upconv_chan, kernel_size=3, stride=1,
                       padding=1, nonlin=self.non_lin, batch_norm=self.norm)])
        list_upconv2 = []
        chan = self.upconv_chan
        new_chan = chan
        for i in range(net_params['upscale_blocks']):
            if i >= 3:
                new_chan = int(chan // self.multiplier_chan)
            list_upconv2 += [nn.Upsample(scale_factor=2, mode='nearest'),
                             conv_block(in_channels=chan, out_channels=new_chan, kernel_size=3,
                                        stride=1, padding=1, nonlin=self.non_lin, batch_norm=self.norm),
                             conv_block(in_channels=new_chan, out_channels=new_chan, kernel_size=3,
                                        stride=1, padding=1, nonlin=self.non_lin, batch_norm=self.norm)]
            chan = new_chan

        self.upconv2 = nn.ModuleList(list_upconv2)

        self.upconv_rec = nn.ModuleList([
            self.norm(chan),
            nn.Conv2d(chan, 3, 3, 1, padding=1),
            # nn.Tanh(),
            Interpolate((max_hw, max_hw), mode='bilinear')])

    def decoding(self, encoding):
        debug = self.debug
        x = encoding
        for layer in self.upconv1:
            x = layer(x)
            if debug:
                print(x.shape, layer)
        for layer in self.upconv2:
            x = layer(x)
            if debug:
                print(x.shape, layer)

        for layer in self.upconv_rec:
            x = layer(x)
            if debug:
                print(x.shape, layer)

        x = self.crop_inp(x)
        if debug:
            print(x.shape)
        if not self.training:
            x = torch.clamp(x, -1, 1)
        return x

    def encoding(self, x):
        debug = self.debug
        if debug:
            print(x.shape, 'encoding start')
        x = self.pad_inp(x)
        if debug:
            print(x.shape, 'pad_inp')
        x = self.conv1(x)
        if debug:
            print(x.shape)
        for layer in self.conv2:
            x = layer(x)
            if debug:
                print(x.shape, layer)
        return x

    def forward(self, x):
        x = self.encoding(x)
        encoding = self.conv_enc(x)

        if self.debug:
            print('encoding', encoding.shape)

        x = self.decoding(encoding)

        return encoding, x


def renorm(inp):
    img = inp.permute(1, 2, 0)
    return torch.clamp(img * STD + 0.5, 0, 1)


def renorm_batch(inp):
    # print(inp.shape)
    img = inp.permute(0, 2, 3, 1)
    # print(img.shape)
    return torch.clamp(img * STD + 0.5, 0, 1)


def var_loss(pred, gt):
    var_pred = nn.functional.avg_pool2d((pred - nn.functional.avg_pool2d(pred, 5, stride=1, padding=2)) ** 2,
                                        kernel_size=5, stride=1)
    var_gt = nn.functional.avg_pool2d((gt - nn.functional.avg_pool2d(gt, 5, stride=1, padding=2)) ** 2, kernel_size=5,
                                      stride=1)
    loss = torch.mean((var_pred - var_gt) ** 2)
    return loss


def train(args, model, device, train_loader, optimizer, epoch, upload_checkpoint, callback, process_upload, scheduler):
    stats_enc = {'mean': 0, 'sum_var': 0, 'n': 0, 'min': torch.tensor(100000000.), 'max': torch.zeros(1)}
    mean_image = 0.0
    model.train()
    total_loss = 0.
    num_loss = 0
    image_first_batch = None
    if args.local:
        fig, ax = plt.subplots(9, figsize=(18, 10))
    num_baches = 0.0
    total_time_batches = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        time.sleep(args.sleep)
        start = time.time()
        model.train()

        data, target = data.to(device), target.to(device)
        if image_first_batch is None:
            image_first_batch = data
        optimizer.zero_grad()
        with torch.autograd.detect_anomaly():
            encoding, output = model(data)
            stats_enc['min'] = torch.min(stats_enc['min'], encoding.min().cpu().detach())
            stats_enc['max'] = torch.max(stats_enc['max'], encoding.max().cpu().detach())
            for b in range(encoding.shape[0]):
                stats_enc['n'] += 1
                mean_image += (data[b].cpu().detach() - mean_image) / stats_enc['n']
                mean_old = stats_enc['mean']
                stats_enc['mean'] += (encoding[b].cpu().detach() - stats_enc['mean']) / stats_enc['n']
                stats_enc['sum_var'] += (encoding[b].cpu().detach() - mean_old) * (
                        encoding[b].cpu().detach() - stats_enc['mean'])
            stats_enc['var'] = (stats_enc['sum_var'] / stats_enc['n'])
            stats_enc['std'] = stats_enc['var'] ** 0.5
            loss_encoding = 0.1 * args.net_params['reg'] * torch.mean(encoding ** 2)
            loss_encoding += args.net_params['reg'] * torch.mean(encoding.mean(0) ** 2)
            #loss_encoding += 0.1*args.net_params['reg'] * torch.mean((1.0 - (encoding).var(0)) ** 2)

            if 'dist_reg' in args.net_params:
                pass  # loss_encoding -= args.net_params['dist_reg']*(encoding.view(encoding.shape[0],-1,1)-encoding.view(1,encoding.shape[0],-1))**2
            loss_mse = torch.mean((data - output) ** 2)
            # loss_aer = torch.mean(torch.abs(data - output))
            loss = loss_mse + loss_encoding  # + 0.01 * loss_aer

            total_loss += loss.item()
            num_loss += 1
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)

        optimizer.step()
        time_batch = time.time() - start
        total_time_batches += time_batch
        num_baches += 1
        if batch_idx % args.log_interval == 0 or batch_idx == len(train_loader) - 1:
            if batch_idx > 0 and batch_idx % (args.log_interval * 3) == 0 and upload_checkpoint:
                if process_upload is not None:
                    process_upload.join()
                save_model(args.checkpoint, epoch, model, optimizer, scheduler)
                if callback is not None:
                    callback(False)
                    process_upload = start_process(callback, (True,))
            if not args.local:
                fig, ax = plt.subplots(9, figsize=(18, 10))
            model.eval()
            img1 = renorm(image_first_batch[0])
            mean_image_norm = renorm(mean_image)
            if epoch == 1:
                matplotlib.image.imsave(os.path.join(args.res_dir, 'mean_image.png'),
                                        mean_image_norm.cpu().detach().numpy(), vmin=0.0, vmax=1.0)
            with torch.no_grad():
                if batch_idx == 0:
                    model.debug = True
                encod1, output1 = model(image_first_batch)
                if batch_idx == 0:
                    model.debug = False
                img1_recon = renorm(output1[0])
                all_img = torch.cat((img1.cpu(), img1_recon.cpu()), 1).detach().numpy()
                matplotlib.image.imsave(
                    os.path.join(args.res_dir, 'img1_reconstruction_epoch_{0:03d}.png'.format(epoch)), all_img,
                    vmin=0.0,
                    vmax=1.0)
                img2 = renorm(data[0])
                img_rec2 = renorm(output[0])
                all_img2 = torch.cat((img2, img_rec2), 1).cpu().detach().numpy()
                matplotlib.image.imsave(
                    os.path.join(args.res_dir, 'img2_reconstruction_epoch_{0:03d}.png'.format(epoch)), all_img2,
                    vmin=0.0,
                    vmax=1.0)

                zero_enc = stats_enc['mean'].view(1, -1).cuda()  #torch.zeros_like(encod1[:1])
                rand_enc = torch.clamp(
                    torch.randn_like(encod1[:1]) * stats_enc['std'].cuda() + stats_enc['mean'].cuda(),
                    -args.net_params['scale_tanh'], args.net_params['scale_tanh'])
                enc_to_show = torch.cat((zero_enc, rand_enc, encod1[:1]), 0)
                rand_img = model.decoding(enc_to_show)
                rand_img_list = []
                width_img = rand_img.shape[2]
                show_every = 3
                for i in range(30 * show_every):
                    if i % show_every == 0:
                        rand_img_list.append(rand_img)
                    _, rand_img = model(rand_img)
                rand_img = renorm_batch(torch.cat(rand_img_list, 3))
                matplotlib.image.imsave(
                    os.path.join(args.res_dir, 'zero_encode_evolution_epoch_{0:03d}.png'.format(epoch)),
                    rand_img[0, :, :width_img * 5].cpu().detach().numpy(), vmin=0.0, vmax=1.0)
                matplotlib.image.imsave(
                    os.path.join(args.res_dir, 'rand_encode_evolution_epoch_{0:03d}.png'.format(epoch)),
                    rand_img[1, :, :width_img * 5].cpu().detach().numpy(), vmin=0.0, vmax=1.0)
                matplotlib.image.imsave(os.path.join(args.res_dir, 'encode_evolution_epoch_{0:03d}.png'.format(epoch)),
                                        rand_img[2, :, :width_img * 5].cpu().detach().numpy(),
                                        vmin=0.0, vmax=1.0)
                rand_img = torch.cat([r for r in rand_img], 0).cpu().detach().numpy()
                img_list = []
                for i in range(6):
                    alpha = i / 5.0
                    blend_enc = alpha * encod1[:1] + (1 - alpha) * encoding[:1]
                    img_list.append(renorm(model.decoding(blend_enc)[0]))
                all_img_blend = torch.cat(img_list, 1).cpu().detach().numpy()

            plt.figure(1)
            ax[0].imshow(np.hstack((all_img, all_img2)))
            ax[1].imshow(all_img_blend)
            matplotlib.image.imsave(os.path.join(args.res_dir, 'img1_to_img2_morph_epoch_{0:03d}.png'.format(epoch)),
                                    all_img_blend, vmin=0.0,
                                    vmax=1.0)

            h = rand_img.shape[0]
            for row in range(2):
                ax[row + 2].imshow(rand_img[:h // 3, row * width_img * 15:(row + 1) * width_img * 15])
            for row in range(2):
                ax[row + 2 + 2].imshow(rand_img[h // 3:2 * h // 3, row * width_img * 15:(row + 1) * width_img * 15])
            for row in range(2):
                ax[row + 2 + 2 + 2].imshow(rand_img[2 * h // 3:, row * width_img * 15:(row + 1) * width_img * 15])
            plt.tight_layout()
            for a in ax:
                a.axis('off')
            plt.subplots_adjust(wspace=0, hspace=0)
            if args.local:
                mypause(0.01)
            else:
                # clear_output()
                plt.draw()
                plt.pause(0.01)

            print('data {:.3f} {:.3f} {:.3f}'.format(data.min().item(), data.max().item(), data.mean().item()))
            print('output {:.3f} {:.3f} {:.3f}'.format(output.min().item(), output.max().item(), output.mean().item()))
            # print(img1_recon.min().item(), img1_recon.max().item(), img1_recon.mean().item())
            print('stats_enc')
            for s in stats_enc:
                if s not in ['sum_var', 'var']:
                    if isinstance(stats_enc[s], int):
                        print("{} = {}".format(s, stats_enc[s]))
                    else:
                        print("{} = {:.3f} {:.3f} {:.3f} shape {}".format(
                            s, stats_enc[s].min().item(), stats_enc[s].mean().item(), stats_enc[s].max().item(),
                            stats_enc[s].shape))
            print('non_lin', model.non_lin)
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f} loss_mse: {:.4f}  loss_enc {:.4f} time_batch {:.2f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), (total_loss / num_loss),
                    loss_mse, loss_encoding.item(), total_time_batches / num_baches))
            model.train()
    plt.close()
    return process_upload


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_args(args_list=None):
    parser = argparse.ArgumentParser(description='PyTorch  Example')
    parser.add_argument('--batch_size', type=int, default=26, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local', action='store_true', default=False,
                        help='local')
    parser.add_argument('--seed', type=int, default=-1, metavar='S',
                        help='random seed (default: -1)')
    parser.add_argument('--optimizer', default='adam',
                        help='optimizer')
    parser.add_argument('--sleep', type=float, default=0.001,
                        help='sleep')
    parser.add_argument('--decay_lr', type=float, default=0.995,
                        help='sleep')
    parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--checkpoint', default='tmp.pth',
                        help='checkpoint')
    parser.add_argument('--save_raw', action='store_true', default=False,
                        help='For Saving the current Model (raw)')
    parser.add_argument('--dataset', default='/home/davide/datasets/', help='dataset path '
                                                                            'e.g. https://drive.google.com/open?id=0BxYys69jI14kYVM3aVhKS1VhRUk')
    parser.add_argument('--res_dir', default='./', help='result dir')
    parser.add_argument('--net_params', default={'non_linearity': "PReLU",
                                                 'norm': 'InstanceNorm2d',
                                                 'base': 32,
                                                 'num_features_encoding': 256,
                                                 'upconv_chan': 256,
                                                 'upconv_size': 4,
                                                 'multiplier_chan': 2,
                                                 'max_chan': 512,
                                                 'upscale_blocks': 6,
                                                 'reg': 0.05,
                                                 'dist_reg': 0.1,
                                                 'crop': -1,
                                                 'scale_tanh': 4,
                                                 'affine': True
                                                 }, type=dict, help='net_params')
    args = parser.parse_args(args_list)
    return args


def main(args, callback=None, upload_checkpoint=False):
    print(vars(args))
    print('upload_checkpoint', upload_checkpoint)
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)
    with open(os.path.join(args.res_dir, 'params.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    if not os.path.exists(os.path.dirname(args.checkpoint)):
        os.makedirs(os.path.dirname(args.checkpoint))
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if args.seed >= 0:
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    transform_list = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[STD, STD, STD])
    ]
    if args.net_params['crop'] > 0:
        transform_list = [transforms.CenterCrop(args.net_params['crop'])] + transform_list
    data_transform = transforms.Compose(transform_list)
    face_dataset_train = datasets.ImageFolder(root=args.dataset,
                                              transform=data_transform)
    # face_dataset_test = datasets.ImageFolder(root='test',
    #                                           transform=data_transform)
    train_loader = torch.utils.data.DataLoader(face_dataset_train,
                                               batch_size=args.batch_size, shuffle=True,
                                               num_workers=0)

    # test_loader = torch.utils.data.DataLoader(face_dataset_test,
    #     batch_size=args.test_batch_size, shuffle=True, **kwargs)
    # args.checkpoint = "cnn3.pth"

    model = Encoder(args.net_params, next(iter(train_loader))[0].shape).to(device)
    if args.optimizer == 'sgd':
        if args.lr is None:
            args.lr = 0.07
        optimizer = optim.SGD(model.parameters(), lr=args.lr, nesterov=True, momentum=0.8, weight_decay=0)
    elif args.optimizer == 'adam':
        if args.lr is None:
            args.lr = 0.00075
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    else:
        raise NotImplementedError
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.decay_lr)
    if os.path.exists(args.checkpoint):
        epoch_start = load_model(args.checkpoint, model, optimizer, scheduler) + 1
    else:
        epoch_start = 1
    if False:
        set_lr(optimizer, args.lr)
    process_upload = None
    for epoch in range(epoch_start, args.epochs + 1):
        print('learning rate {:.5f}'.format(get_lr(optimizer)))
        process_upload = train(args, model, device, train_loader, optimizer, epoch, upload_checkpoint, callback,
                               process_upload, scheduler)
        if process_upload is not None:
            process_upload.join()
        save_model(args.checkpoint, epoch, model, optimizer, scheduler)
        if callback is not None:
            callback(False)
            if upload_checkpoint:
                process_upload = start_process(callback, (True,))

        scheduler.step()
        # no test at the moment
        # test(args, model, device, test_loader)



if __name__ == '__main__':
    base_dir_res = "/home/davide/results/cnn_autoencoders_local"
    base_dir_dataset = '/home/davide/datasets/faces'
    list_args = ['--sleep', '0.5', '--local', '--batch_size', '18', '--dataset', base_dir_dataset,
                 '--res_dir', base_dir_res,
                 '--checkpoint', os.path.join(base_dir_res, 'checkpoint.pth')]
    args = get_args(list_args)
    main(args)
