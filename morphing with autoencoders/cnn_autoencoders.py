import torch
from torch import nn
from torchvision import transforms, datasets
import argparse
import torch.optim as optim
import matplotlib
import time

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import random, os
import numpy as np

STD = 0.505
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


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        shape = [x.shape[0]] + self.shape
        return x.view(*shape)


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x


class Encoder(nn.Module):
    def __init__(self, non_lin=nn.ELU, norm=nn.InstanceNorm2d):
        super(Encoder, self).__init__()

        self.norm = norm
        self.base = 72
        self.non_lin = non_lin

        self.base_enc = 48
        self.upconv_chan = 256
        self.upconv_size = 16
        self.conv1 = nn.Sequential(
            *[nn.Conv2d(3, self.base, 5, 2, padding=2), self.norm(self.base, affine=True), non_lin()])

        list_conv2 = []

        chan = self.base
        for i in range(5):
            list_conv2 += [nn.Conv2d(chan, chan, 3, 1, padding=1),
                           self.norm(chan, affine=True),
                           non_lin()]

            list_conv2 += [nn.Conv2d(chan, chan, 3, 1, padding=1),
                           self.norm(chan, affine=True),
                           non_lin()]
            list_conv2 += [nn.Conv2d(chan, chan, 3, 2, padding=1),
                           self.norm(chan, affine=True),
                           non_lin()]
        self.conv2 = nn.Sequential(*list_conv2)

        self.conv_enc = nn.Sequential(
            *[nn.Conv2d(chan, self.base_enc, 4, 1, padding=0),
              self.norm(self.base_enc, affine=True),
              nn.Tanh()])

        self.upconv1 = nn.Sequential(*[
            View([-1]), nn.Linear(self.base_enc, self.upconv_chan * self.upconv_size * self.upconv_size),
            View([self.upconv_chan, self.upconv_size, self.upconv_size]),
            non_lin(),
            nn.Conv2d(self.upconv_chan, chan, 3, 1, padding=1),
            self.norm(chan, affine=True),
            non_lin()])
        list_upconv2 = []
        for i in range(4):
            # if i>=4:
            #     chan2 = chan//2
            # else:
            #     chan2 = chan
            chan2 = chan
            list_upconv2 += [nn.Upsample(scale_factor=2, mode='nearest'),
                             nn.Conv2d(chan, chan2, 3, 1, padding=1),
                             self.norm(chan2, affine=True),
                             non_lin()]
            list_upconv2 += [nn.Conv2d(chan2, chan2, 3, 1, padding=1), self.norm(chan2, affine=True),
                             non_lin()]
            list_upconv2 += [nn.Conv2d(chan2, chan2, 3, 1, padding=1), self.norm(chan2, affine=True),
                             non_lin()]
            chan = chan2

        self.upconv2 = nn.Sequential(*list_upconv2)

        self.upconv_rec = nn.Sequential(*[
            self.norm(chan, affine=True),
            nn.Conv2d(chan, 3, 3, 1, padding=1),
            Interpolate((200, 200), mode='bilinear'),
            nn.Tanh()])

        self.debug = False

    def decoding(self, encoding):
        debug = self.debug  # and random.random() < 0.001
        x = self.upconv1(encoding)
        if debug:
            print(x.shape)
        x = self.upconv2(x)
        if debug:
            print(x.shape)
        x = self.upconv_rec(x)
        if debug:
            print(x.shape)
        return x

    def forward(self, x):
        debug = self.debug  # and random.random() < 0.001

        if debug:
            print(x.shape)
        x = self.conv1(x)
        if debug:
            print(x.shape)
        x = self.conv2(x)
        if debug:
            print(x.shape)
        encoding = self.conv_enc(x)

        if debug:
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


def train(args, model, device, train_loader, optimizer, epoch):
    stats_enc = {'mean': 0, 'sum_var': 0, 'n': 0, 'min': torch.tensor(100000000.), 'max': torch.zeros(1)}
    mean_image = 0.0
    model.train()
    total_loss = 0.
    num_loss = 0
    image = None
    fig, ax = plt.subplots(7)
    for batch_idx, (data, target) in enumerate(train_loader):
        time.sleep(0.25)  # fixme
        model.train()
        data, target = data.to(device), target.to(device)
        if image is None:
            image = data
        # data = image #remove
        optimizer.zero_grad()
        with torch.autograd.detect_anomaly():
            encoding, output = model(data)

            tmp = encoding.min(dim=0)
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
            loss_encoding = 0.001 * torch.mean(encoding ** 2) + 0.001 * torch.mean(encoding) ** 2
            # square = (data - output) ** 2 + 0.1 * torch.abs(data - output)
            # mask = 1.0*(square>(square.mean()+square.std()))
            # mask = mask.float()
            # mask_perc = mask.mean()*100
            # loss1 = torch.mean(square*mask)
            # loss2 = torch.mean(square*(1-mask))
            # loss = 0.1*loss1 + 0.9*loss2
            loss_mse = torch.mean((data - output) ** 2)
            loss_aer = torch.mean(torch.abs(data - output))
            loss = 0.99 * loss_mse + 0.01 * loss_aer + loss_encoding

            total_loss += loss.item()
            num_loss += 1
            loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            if batch_idx > 1 and batch_idx % args.log_interval * 5 == 0:
                print('saving')
                if os.path.exists(args.checkpoint):
                    os.rename(args.checkpoint, args.checkpoint + '.old')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, args.checkpoint)
                print('saved')

            model.eval()
            img1 = renorm(image[0])
            with torch.no_grad():
                encod1, output1 = model(image)
                img_rec1 = renorm(output1[0])
                all_img = torch.cat((renorm(mean_image), img1.cpu(), img_rec1.cpu()), 1).detach().numpy()
                img2 = renorm(data[0])
                img_rec2 = renorm(output[0])
                all_img2 = torch.cat((img2, img_rec2), 1).cpu().detach().numpy()

                rand_enc = 0. * torch.clamp(torch.randn_like(encod1[:1]) * stats_enc['std'].cuda(), -1, 1)  # fixme
                rand_enc = torch.cat((rand_enc, encod1[:1]), 0)
                rand_img = model.decoding(rand_enc)
                rand_img_list = []
                width_img = rand_img.shape[2]
                show_every = 3
                for i in range(30 * show_every):
                    if i % show_every == 0:
                        rand_img_list.append(rand_img)
                    _, rand_img = model(rand_img)
                rand_img = renorm_batch(torch.cat(rand_img_list, 3))
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

            h = rand_img.shape[0]
            for row in range(2):
                ax[row + 2].imshow(rand_img[:h // 2, row * width_img * 15:(row + 1) * width_img * 15])
            for row in range(2):
                ax[row + 2 + 2].imshow(rand_img[h // 2:, row * width_img * 15:(row + 1) * width_img * 15])
            plt.tight_layout()
            for a in ax:
                a.axis('off')
            plt.subplots_adjust(wspace=0, hspace=0)
            mypause(0.01)

            print('data {:.3f} {:.3f} {:.3f}'.format(data.min().item(), data.max().item(), data.mean().item()))
            print('output {:.3f} {:.3f} {:.3f}'.format(output.min().item(), output.max().item(), output.mean().item()))
            # print(img_rec1.min().item(), img_rec1.max().item(), img_rec1.mean().item())
            for s in stats_enc:
                if isinstance(stats_enc[s], int):
                    print("{} = {}".format(s, stats_enc[s]))
                else:
                    print("{} = {:.3f} {:.3f} {:.3f}".format(
                        s, stats_enc[s].min().item(), stats_enc[s].mean().item(), stats_enc[s].max().item()))
            print('non_lin', model.non_lin)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f} loss_mse: {:.4f}  loss_aer {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), (total_loss / num_loss),
                loss_mse, loss_aer))
            model.train()
    plt.close()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch  Example')
    parser.add_argument('--batch-size', type=int, default=24, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0009, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--instance', action='store_true', default=False,
                        help='instance norm')
    parser.add_argument('--seed', type=int, default=-1, metavar='S',
                        help='random seed (default: -1)')
    parser.add_argument('--optimizer', default='adam',
                        help='optimizer')
    parser.add_argument('--log-interval', type=int, default=25, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--checkpoint', default='tmp.pth',
                        help='checkpoint')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--dataset', default='datasets', help='dataset path')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if args.seed >= 0:
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[STD, STD, STD])
    ])
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
    if args.instance:
        norm = nn.InstanceNorm2d
    else:
        norm = nn.BatchNorm2d
    print(norm)
    model = Encoder(non_lin=nn.ELU, norm=norm).to(device)
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, nesterov=True, momentum=0.8)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    else:
        raise NotImplementedError
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    print('learning rate', get_lr(optimizer))
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step()

        # test(args, model, device, test_loader)

        print('learning rate', get_lr(optimizer))


if __name__ == '__main__':
    main()
