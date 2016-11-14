'''
Created on Jun 13, 2016

@author: davide
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
import scipy.optimize


plt.ion()
fig, ax = plt.subplots()
np.random.seed(1)

N = 20
X = np.concatenate((np.random.normal(0, 1, 0.3 * N),
                    np.random.normal(5, 1, 0.7 * N)))[:, np.newaxis]
X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
bins = np.linspace(-5, 10, 10)

X_plot = np.linspace(0.00, 1., 5000)

# f(x) to integrate
def target(X):
    return 1. * (X > 0.) * (X < 1.) * (1 + np.sin(1 / X))

def normtarget(X):
    return target(X) / 1.504067061906928371989856117741148229624985028212639170871


def proposal(X, W, Centroids, bandwidth):
    d = np.zeros_like(X)
    for c in xrange(Centroids.shape[0]):
        d += norm(Centroids[c], bandwidth).pdf(X) * W[c]
    d /= np.sum(W)
    return d


def RBFfeatures(X, Centroids, bandwidth):
    d = norm(Centroids[0], bandwidth).pdf(X)
    d = d.reshape(-1, 1)
    for c in xrange(Centroids.shape[0] - 1):
        d = np.concatenate((d, (norm(Centroids[c + 1], bandwidth).pdf(X)).reshape(-1, 1)), 1)
    return d


def samplefromp(num, W, Centroids, bandwidth):
    indeces = np.random.choice(Centroids.shape[0], num, replace=True, p=W / np.sum(W))
    return np.array([np.random.normal(Centroids[indeces[i]], bandwidth) for i in xrange(num)])


def weights(X, oldX, oldW):
    WX = np.minimum(np.abs(target(X)) / proposal(X, oldW, oldX, bandwidth), 100.)  # limit!
    return WX


count = 0
estimate = 0.

bandwidth = 0.01
W = np.ones(10) / 10.
Centroids = np.linspace(0.000000001, 3, 10)
learnrate = 1.

while True:
    count += 1
    learnrate *=0.97

    X = samplefromp(300, W, Centroids, bandwidth)

    WX = weights(X, Centroids, W)

    estimate = estimate * (1 - learnrate) + np.mean(WX) * learnrate
    print 'estimate', estimate,'error',abs(estimate-1.504067061906928371989856117741148229624985028212639170871)

    NewX = np.concatenate((X, Centroids))
    features = RBFfeatures(NewX, NewX, bandwidth)

    (W, err) = scipy.optimize.nnls(features, target(NewX) / estimate)
    Centroids = NewX[W > 0.0000000001]  # prune points with low weights
    W = W[W > 0.0000000001]

    if count % 1 == 0:

        ax.plot(X_plot, normtarget(X_plot), label='normalized f(x) = optimal proposal')
        ax.plot(X_plot, target(X_plot), label='f(x) to integrate')
        pr = proposal(Centroids, W, Centroids, bandwidth)
        ax.scatter(X, WX / 10., s=10, marker='x', color='red', label="weight")
        ax.scatter(Centroids, pr, s=10, label="proposal")

        ax.legend(loc='upper left')
        ax.plot(Centroids, -0.01 - 0.1 * np.random.random(Centroids.shape[0]), '+k')

        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.2, 3)
        plt.draw()
        plt.pause(0.1)
        ax.clear()
