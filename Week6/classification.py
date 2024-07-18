# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def load_data(name, m=None):
    data = np.load(name)
    x = data[:,:-1]
    y = data[:,-1]

    return (x, y)


if __name__ == '__main__':

    print('Dataset O')
    x_train, y_train = load_data('dataset_O_train.npy')
    x_test, y_test = load_data('dataset_O_test.npy')

    print('Dataset U')
    x_train, y_train = load_data('dataset_U_train.npy')
    x_test, y_test = load_data('dataset_U_test.npy')

    print('Dataset V')
    x_train, y_train = load_data('dataset_V_train.npy')
    x_test, y_test = load_data('dataset_V_test.npy')
