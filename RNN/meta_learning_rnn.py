"""
Meta-learner for supervised task
inspired by the paper 'Learning to learn using gradient descent. Hochreiter, Younger, and Conwell. 2001'

"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

plt.ion()


def sample_func(sample_size, params, batch_size=64):
    X = np.random.random(size=(sample_size, batch_size, 1)) * 2 - 1.0
    Y = params[0] * X + params[1] * X * X
    z = np.zeros((1, batch_size, 1))
    Yshift = np.concatenate((z, Y), axis=0)
    X = np.concatenate((X, Yshift[:-1]), axis=2)
    return X, Y


class MetaLearner(nn.Module):
    def __init__(self, inp_layer_size, hidden_size):
        super(MetaLearner, self).__init__()
        self.hidden_size = hidden_size

        self.inp = nn.Linear(2, inp_layer_size)
        self.rnn = nn.LSTM(inp_layer_size, hidden_size, 2, dropout=0.0)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, input, hidden=None):
        input = self.inp(input)
        input = F.elu(input)
        output, hidden = self.rnn(input, hidden)
        output = self.out(output)
        return output, hidden


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    n_epochs = 1000
    n_iters = 50
    hidden_size = 20
    inp_layer_size = 20
    batch_size = 64
    seq_len = 15
    lr = 0.01
    model = MetaLearner(inp_layer_size, hidden_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = np.zeros(n_epochs)  # For plotting

    for epoch in range(n_epochs):
        lr *= 0.98
        adjust_learning_rate(optimizer, lr)

        for iter in range(n_iters):
            params = (np.random.random(size=(3, 1, batch_size, 1)) - 0.5) * 2.
            X, Y = sample_func(seq_len, params, batch_size=batch_size)
            inputs = Variable(torch.from_numpy(X).float())
            targets = Variable(torch.from_numpy(Y).float())
            model.train()
            outputs, hidden = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            losses[epoch] += loss.item()
            loss.backward()
            optimizer.step()

        # plotting
        if epoch % 1 == 0:
            model.eval()
            mX = []
            mY = []
            for i in range(1):
                params = (np.random.random(size=(3, 1, 1, 1)) - 0.5) * 2.
                X, Y = sample_func(seq_len, params, batch_size=1)
                mX.append(X)
                mY.append(Y)
            mX = np.concatenate(mX, 0)
            mY = np.concatenate(mY, 0)
            inputs = Variable(torch.from_numpy(mX).float())
            targets = Variable(torch.from_numpy(mY).float())
            outputs, hidden = model(inputs)

            loss_val = criterion(outputs, targets)
            print("epoch {:2d} loss epoch {:0.3f} lr {:0.4f} hidden size {}".format(epoch, losses[epoch], lr,
                                                                                    hidden_size))
            plt.figure('function predictions (red) and targets (blue)')
            plt.clf()
            plt.plot(inputs[:seq_len, 0, 0].data.numpy(), targets[:seq_len, 0, 0].data.numpy(), 'ro', color='b')
            plt.plot(inputs[:seq_len, 0, 0].data.numpy(), outputs[:seq_len, 0, 0].data.numpy(), 'ro', color='r')
            plt.figure('loss')
            plt.clf()
            err = torch.abs(targets - outputs)
            plt.plot(err[:, 0, 0].data.numpy(), color='b')
            plt.draw()
            plt.pause(.1)
