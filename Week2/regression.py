# author: Michael Hüppe
# date: 10.04.2024
# project: /regression.py
import warnings

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def load_data(name, m=None, rng=None):
    data = np.load(name)
    x = data[:, :-1]
    y = data[:, -1]

    if not m is None:
        if rng is None: rng = np.default_rng(seed=66)
        idx = rng.choice(m, size=len(x), replace=False)
        x = x[idx]
        y = y[idx]

    return (x, y)


def plot(x, y, w=None, sigma=None):
    '''
    only for plotting 2D data
    '''

    plt.plot(x, y, '.r', markersize=8, label='Samples')

    # also plot the prediction
    if not w is None:
        deg = w.shape[0]
        x_plot = np.linspace(np.min(x), np.max(x), 100)
        X_plot = np.vander(x_plot, deg)

        # set plotting range properly
        plt.ylim((np.min(y) * 1.2, np.max(y) * 1.2))

        plt.plot(x_plot, np.dot(X_plot, w), linewidth=5, color='tab:blue', label="Model")

        # also plot confidence intervall
        if not sigma is None:
            plt.plot(x_plot, np.dot(X_plot, w) + sigma, linewidth=2, color='tab:cyan')
            plt.plot(x_plot, np.dot(X_plot, w) - sigma, linewidth=2, color='tab:cyan')

    plt.tight_layout()
    plt.savefig('fig.pdf')

    plt.show()

def backtracking_line_search(fg, x, d):
    alpha = 0.2
    beta = 0.8
    t = 1
    f, g = fg(x)
    while fg(x + t*d)[0] > f + alpha * t * np.dot(g, d):
        t *= beta
    return x + t * d

def gradient_descent(fg, x, niter=100):
    for i in range(niter):
        f, g = fg(x)
        x = backtracking_line_search(fg, x, -g)
    return x

def regression(x, y):

    X = np.block([[ x, np.ones((len(x), 1)) ]])
    def fg(x0):
        '''
        returns the function value and gradient at x0
        '''
        v = X.dot(x0) - y
        gx = 2 * X.T.dot(v)
        fx = v.dot(v)
        return fx, gx

    x0 = np.zeros(X.shape[1])
    w = gradient_descent(fg, x0)

    return w




if __name__ == '__main__':
    x, y = load_data('dataset0.npy')
    w = gradient_descent(x, y)
    plot(x, y, w)


    x, y = load_data('dataset1.npy')
    w = gradient_descent(x, y)
    plot(x, y, w)

    x, y = load_data('dataset2.npy')
    w = gradient_descent(x, y)
    plot(x, y, w)

    x, y = load_data('dataset3.npy')
    w = gradient_descent(x, y)
    plot(x, y, w)

    x, y = load_data('dataset4.npy')
    w = gradient_descent(x, y)
    print(w)
    print(np.linalg.norm(x @ w[:-1] + w[-1] - y))
