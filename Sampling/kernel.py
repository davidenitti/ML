'''
Created on Jun 13, 2016

@author: davide
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity


np.random.seed(1)
N = 20
X = np.concatenate((np.random.normal(0, 1, 0.3 * N),
                    np.random.normal(5, 1, 0.7 * N)))[:, np.newaxis]
X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
bins = np.linspace(-5, 10, 10)

# ----------------------------------------------------------------------
# Plot a 1D density example
N = 100
np.random.seed(1)
plt.ion()
fig, ax = plt.subplots()

X_plot = np.linspace(-5, 5, 1000)


def target(X):
    return np.abs(np.sin(X)) * 1. * (X > -np.pi / 2) * (
    X < np.pi / 2)


def normtarget(X, W):
    nrm = np.sum(W) / W.shape[0]
    return target(X) / nrm




def proposal(X, W, Centroids, bandwidth):
    d = np.zeros_like(X)
    for c in xrange(Centroids.shape[0]):
        d += norm(Centroids[c], bandwidth).pdf(X) * W[c]
    d /= np.sum(W)
    return d


def samplefromp(num, W, Centroids, bandwidth):
    indeces = np.random.choice(Centroids.shape[0], num, replace=True, p=W / np.sum(W))
    return np.array([np.random.normal(Centroids[indeces[i]], bandwidth) for i in xrange(num)])


def weights(X, oldX, oldW):
    WX = np.minimum(np.abs(target(X)) / proposal(X, oldW, oldX, bandwidth), 10.)  # limit!
    return X,WX


count = 0
estimate = 0.
bandwidth = 0.06 # standard deviation of the Gaussian kernel
learnrate = 1.

W = np.ones(5) / 5.
Centroids = np.linspace(-3, 3, 5)

while True:
    count += 1
    learnrate *=0.97
    X = samplefromp(1000, W, Centroids, bandwidth)

    Centroids, W = weights(X, Centroids, W)
    estimate = estimate * (1-learnrate) + np.mean(W) * learnrate
    print 'estimate', estimate,'error',abs(estimate-2.)
    if count % 1 == 0:
        ax.plot(X_plot, normtarget(X_plot, W), label='normalized f(x)')
        ax.plot(X_plot, target(X_plot), label='f(x) to integrate')
        pr = proposal(Centroids, W, Centroids, bandwidth)

        ax.scatter(Centroids, W / 10., s=10, marker='x', color='red', label="weight")
        ax.scatter(Centroids, pr, s=10, label="proposal")

        ax.legend(loc='upper left')
        ax.plot(Centroids, -0.01 - 0.1 * np.random.random(Centroids.shape[0]), '+k')

        ax.set_xlim(-2, 2)
        ax.set_ylim(-0.2, 1.2)
        plt.draw()
        plt.pause(0.1)
        ax.clear()
