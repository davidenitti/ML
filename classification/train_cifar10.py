from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import json
import os, sys, time

num_classes = 10
if __name__ == '__main__':
    import utils, networks
else:
    from . import utils, networks


def train(args, model, device, train_loader, optimizer, epoch):
    if epoch == 1:
        model.debug = True
    model.train()
    # print('num batches', len(train_loader))
    print(args)
    print(args.net_params)
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        time.sleep(args.sleep)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if args.net_params['autoencoder']:
            output, decoded = model(data)
        else:
            output = model(data)
        if epoch == 1:
            model.debug = False
        loss = F.cross_entropy(output, target)
        if args.net_params['autoencoder']:
            loss += 0.4 * torch.mean((data - decoded) ** 2)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0 or batch_idx == len(train_loader) - 1:
            if (args.checkpoint != ""):
                utils.save_model(args.checkpoint, epoch, model, optimizer)
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                epoch,
                min(len(train_loader.dataset), (batch_idx + 1) * train_loader.batch_size),
                len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader),
                (total_loss / (batch_idx + 1))))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    num = 0
    with torch.no_grad():
        for data, target in test_loader:
            num += 1
            data, target = data.to(device), target.to(device)
            if args.net_params['autoencoder']:
                output, decoded = model(data)
            else:
                output = model(data)
            assert output.shape[1] > target.max().item()
            test_loss += F.cross_entropy(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= num
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)


def main(args, callback=None, upload_checkpoint=False):
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print('device', device)
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(os.path.join(args.dataset, 'cifar10'), train=True, download=True,
                         transform=transforms.Compose([
                             transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(os.path.join(args.dataset, 'cifar10'), train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    num_out = num_classes
    print(vars(args))
    print(args.net_params)
    model = networks.Net(num_out, args.net_params, next(iter(train_loader))[0].shape).to(device)
    if args.optimizer.lower() == "adam":
        if args.lr is None:
            args.lr = 0.001
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "sgd":
        if args.lr is None:
            args.lr = 0.1
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    else:
        raise NotImplementedError
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
    utils.load_model(args.checkpoint, model, optimizer)
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print('learning rate {:.5f}'.format(utils.get_lr(optimizer)))
        train(args, model, device, train_loader, optimizer, epoch)
        acc = test(args, model, device, test_loader)
        best_acc = max(best_acc, acc)
        scheduler.step()

    results_path = os.path.join(args.res_dir, 'results_cifar10.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        results = {'res': []}
    results['res'].append([vars(args), {'best': best_acc, 'last': acc}])
    if not os.path.exists(os.path.dirname(results_path)):
        os.makedirs(os.path.dirname(results_path))
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)


def get_args(args_list):
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate')
    parser.add_argument('--sleep', type=float, default=0.01,
                        help='sleep')
    parser.add_argument('--weight_decay', type=float, default=2e-6,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=500,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--optimizer', default="sgd",
                        help='sleep')
    parser.add_argument('--checkpoint', default="",
                        help='checkpoint path')
    parser.add_argument('--dataset', default='./dataset',
                        help='dataset path')

    parser.add_argument('--net_params', default={'non_linearity': "ReLU",
                                                 'random_pad': False,
                                                 'last_pool': 'avg_pool2d',
                                                 'padding': True,
                                                 'base': 32,
                                                 'autoencoder': False,
                                                 'batchnorm': 'standard',
                                                 'conv_block': 'ResNetBlock',
                                                 'kernel_size': 3
                                                 }, type=dict, help='net_params')
    parser.add_argument('--res_dir', default='./',
                        help='results directory')
    args = parser.parse_args(args_list)
    return args


def hyper_tune(args, callback=None, upload_checkpoint=False):
    # args = get_args(list_args)
    for autoencoder in [False, True]:
        for non_lin in ['ReLU', 'ReLU6', "PReLU"]:
            for pool in ['avg_pool2d', 'max_pool2d']:
                for base in [64, 128]:
                    for conv_block in ['ResNetBlock']:
                        for kernel in [3, 5]:
                            for opt in ['adam']:
                                args.optimizer = opt
                                args.net_params = {'non_linearity': non_lin,
                                                   'random_pad': False,
                                                   'last_pool': pool,
                                                   'padding': True,
                                                   'base': base,
                                                   'autoencoder': autoencoder,
                                                   'batchnorm': 'standard',
                                                   'conv_block': conv_block,
                                                   'kernel_size': kernel
                                                   }
                                args.checkpoint = ""  # checkpoint not needed
                                main(args)
                                if callback is not None:
                                    callback()


if __name__ == '__main__':
    base_dir_res = "../../../results"
    base_dir_dataset = '/home/davide/datasets'
    list_args = ['--dataset', base_dir_dataset,
                 '--res_dir', base_dir_res,
                 '--sleep', '0.1']
    args = get_args(list_args)
    main(args)
