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

        self.err_synd_train = err_synd_train
        self.logical_err_train = logical_err_train
        lg_test = data_test[:49, 0:2]
        N = len(np.array(lg_test)[:, 0])

        def logic(L):

            M=np.zeros(4, dtype=int)

            if (L[0]==0 and L[1]==0):
                M[0]=1
            if (L[0]==0 and L[1]==1):
                M[1]=1
            if (L[0]==1 and L[1]==0):
                M[2]=1
            if (L[0]==1 and L[1]==1):
                M[3]=1

            return np.array(M)


        new_lg_test = np.zeros((N, 4), dtype=int)
        for i in range(N):
            new_lg_test[i] = (logic(lg_test[i,:]))
        #print(new_lg_test)


        err_synd_test = data_test[:49, -8:]
        logical_err_test = np.array(new_lg_test)
        err_synd_test = np.array(err_synd_test, dtype=np.int)
        logical_err_test = np.array(logical_err_test, dtype=np.int)
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
    print('\n',x_train.shape, x_train.dtype)

    #print('\nlogical gates in array of 0s and 1s: \n', y)
    print('\n',y_train.shape, y_train.dtype,'\n')
