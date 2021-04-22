import torch
import numpy as np


class Data():

    def __init__(self, data_train, data_test):

        lg_train = data_train[:, -5:-1]
        for i in range(len(lg_train[:, 0])):
                if lg_train[i, 0]== max(lg_train[i, :]): lg_train[i, :] = [1,0,0,0]
                if lg_train[i, 1]== max(lg_train[i, :]): lg_train[i, :] = [0,1,0,0]
                if lg_train[i, 2]== max(lg_train[i, :]): lg_train[i, :] = [0,0,1,0]
                if lg_train[i, 3]== max(lg_train[i, :]): lg_train[i, :] = [0,0,0,1]
                #print(lg[i, :], 'index ',i)

        err_synd_train = data_train[:, :-5]
        logical_err_train = np.array(lg_train)
        err_synd_train = np.array(err_synd_train, dtype=np.int)
        logical_err_train = np.array(logical_err_train, dtype=np.int)
        #print(logical_err_train, '\n')
        #print(err_synd_train.shape, err_synd_train.dtype, logical_err_train.shape, logical_err_train.dtype)
        print(logical_err_train)
        self.err_synd_train = err_synd_train
        self.logical_err_train = logical_err_train
        print(lg_train)
        lg_test = data_test[:49, 0:2]
        lg_test.tolist()
        N = len(np.array(lg_test)[:, 0])
        print(N)
        for i in range(N):
            print(lg_test)
            if lg_test[i, 0] and lg_test[i, 1] == 0:
                lg_test[i, :] = [1,0,0,0]
            print(lg_test)
            if lg_test[i, 0] == 1 and lg_test[i, 1] == 0:
                lg_test[i, :] = [0,1,0,0]
            if lg_test[i, 0] == 0 and lg_test[i, 1] == 1:
                lg_test[i, :] = [0,0,1,0]
            if lg_test[i, 0] and lg_test[i, 1] == 1:
                lg_test[i, :] = [0,0,0,1]


        err_synd_test = data_test[:49, -8:]
        logical_err_test = np.array(lg_test)
        err_synd_test = np.array(err_synd_test, dtype=np.int)
        logical_err_test = np.array(logical_err_test, dtype=np.int)
        print(err_synd_test)
        #print(logical_err_train, '\n')
        #print(err_synd_train.shape, err_synd_train.dtype, logical_err_train.shape, logical_err_train.dtype)

        self.err_synd_test = err_synd_test
        self.logical_err_test = logical_err_test

if __name__ == '__main__':

    data_train = np.loadtxt('../Datasets/d=3_train.txt')
    data_test = np.loadtxt('../Datasets/d=3_test.txt')
    for_training = Data(data_train, data_test)
    x_train, y_train = for_training.err_synd_train, for_training.logical_err_train
    x_test, y_test = for_training.err_synd_test, for_training.logical_err_test

    #print('\nsyndrome measurements for training: \n', x)
    print('\n',x.shape, x.dtype)

    #print('\nlogical gates in array of 0s and 1s: \n', y)
    print('\n',y.shape, y.dtype,'\n')
