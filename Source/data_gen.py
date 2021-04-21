import torch
import numpy as np


class Data():

    def __init__(self, data):

        lg = data[:, -5:-1]
        for i in range(0, len(lg[:, 0])):
                if lg[i, 0]== max(lg[i, :]): lg[i, :] = [1,0,0,0]
                if lg[i, 1]== max(lg[i, :]): lg[i, :] = [0,1,0,0]
                if lg[i, 2]== max(lg[i, :]): lg[i, :] = [0,0,1,0]
                if lg[i, 3]== max(lg[i, :]): lg[i, :] = [0,0,0,1]
                #print(lg[i, :], 'index ',i)

        err_synd_train = data[:, :-5]
        logical_err_train = np.array(lg)
        err_synd_train = np.array(err_synd_train, dtype=np.float32)
        logical_err_train = np.array(logical_err_train, dtype=np.float32)
        #print(logical_err_train, '\n')
        #print(err_synd_train.shape, err_synd_train.dtype, logical_err_train.shape, logical_err_train.dtype)

        self.err_synd_train = err_synd_train
        self.logical_err_train = logical_err_train


if __name__ == '__main__':

    data = np.loadtxt('../Datasets/d=5.txt')
    for_training = Data(data)
    x, y = for_training.err_synd_train, for_training.logical_err_train

    print('\nsyndrome measurements for training: \n', x)
    print('\n',x.shape, x.dtype)

    print('\nlogical gates in array of 0s and 1s: \n', y)
    print('\n',y.shape, y.dtype,'\n')
