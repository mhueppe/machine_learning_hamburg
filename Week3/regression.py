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



def regression(x, y):
    # assemble matrix A
    # Add vector for bias
    X = np.vstack([x.T, np.ones(x.shape[0])]).T

    # turn y into a column vector
    y = y[:, np.newaxis]

    # calculate weights:
    # w = (XᵀX)⁻¹Xᵀy
    # Compute X^TX and X^Ty
    XTX = np.dot(X.T, X)
    XTy = np.dot(X.T, y)

    # Define regularization parameter lambda and identity matrix I
    lambda_val = 0.01  # Your regularization parameter
    n = X.shape[0]  # Number of samples
    I = np.identity(XTX.shape[0])  # Identity matrix of appropriate size

    # Add regularization term lambda*n*I to XTX
    XTX_reg = XTX + lambda_val * n * I

    # Solve the linear equations for w
    w = np.linalg.solve(XTX_reg, XTy)
    return w

if __name__ == '__main__':
    x, y = load_data('dataset0.npy')
    w = regression(x, y)
    plot(x, y, w)

    x, y = load_data('dataset1.npy')
    w = regression(x, y)
    plot(x, y, w)

    x, y = load_data('dataset2.npy')
    w = regression(x, y)
    plot(x, y, w)

    x, y = load_data('dataset3.npy')
    w = regression(x, y)
    plot(x, y, w)

    x, y = load_data('dataset4.npy')
    w = regression(x, y)
    print(w)
    print(np.linalg.norm(x @ w[:-1] + w[-1] - y))
