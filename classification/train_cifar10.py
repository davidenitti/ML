from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import json
import os, sys, time

num_classes = 10
import utils, networks


def train(args, model, device, train_loader, optimizer, epoch):
    if epoch == 1:
        model.debug = True
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        time.sleep(args.sleep)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if epoch == 1:
            model.debug = False
        loss = F.cross_entropy(output, target)
        loss_val = loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            if (args.checkpoint != ""):
                utils.save_model(args.checkpoint, epoch, model, optimizer)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss_val))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    num = 0
    with torch.no_grad():
        for data, target in test_loader:
            num += 1
            data, target = data.to(device), target.to(device)
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


def main(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print('device', device)
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(os.path.join(args.base_dir_dataset, 'cifar10'), train=True, download=True,
                         transform=transforms.Compose([
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(os.path.join(args.base_dir_dataset, 'cifar10'), train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    num_out = num_classes
    print(vars(args))
    print(args.net_params)
    model = networks.Net(num_out, args.net_params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
    utils.load_model(args.checkpoint, model, optimizer)
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print('learning rate {:.5f}'.format(utils.get_lr(optimizer)))
        train(args, model, device, train_loader, optimizer, epoch)
        acc = test(args, model, device, test_loader)
        best_acc = max(best_acc, acc)
        scheduler.step()
    if os.path.exists(args.results):
        with open(args.results, 'r') as f:
            results = json.load(f)
    else:
        results = {'res': []}
    results['res'].append([vars(args), {'best': best_acc, 'last': acc}])
    if not os.path.exists(os.path.dirname(args.results)):
        os.makedirs(os.path.dirname(args.results))
    with open(args.results, 'w') as f:
        json.dump(results, f, indent=4)


def get_args(args_list):
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10')
    parser.add_argument('--batch-size', type=int, default=40,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=40,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0009,
                        help='learning rate')
    parser.add_argument('--sleep', type=float, default=0.01,
                        help='sleep')
    parser.add_argument('--weight_decay', type=float, default=3e-6,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=500,
                        help='how many batches to wait before logging training status')

    parser.add_argument('--checkpoint', default="",
                        help='checkpoint path')
    parser.add_argument('--base_dir_dataset', default='./dataset',
                        help='base_dir_dataset path')

    parser.add_argument('--net_params', default={'non_linearity': "PReLU",
                                                 'random_pad': False,
                                                 'last_pool': 'avg_pool2d',
                                                 'padding': True,
                                                 'base': 32
                                                 }, type=dict, help='net_params')
    parser.add_argument('--results', default='./results_cifar10.json',
                        help='results')
    args = parser.parse_args(args_list)
    return args


def hyper_tune(args, update_func=None):
    # args = get_args(list_args)
    for non_lin in ['ReLU', "PReLU"]:
        for pool in ['avg_pool2d', 'max_pool2d']:
            for base in [64, 128, 256, 32]:
                args.net_params = {'non_linearity': non_lin,
                                   'random_pad': False,
                                   'last_pool': pool,
                                   'padding': True,
                                   'base': base
                                   }
                main(args)
                if update_func is not None:
                    update_func()


if __name__ == '__main__':
    base_dir_res = "../../../results"
    base_dir_dataset = '/home/davide/datasets'
    list_args = ['--base_dir_dataset', base_dir_dataset,
                 '--results', os.path.join(base_dir_res, 'results_cifar10.json')]
    args = get_args(list_args)
    main(args)
