#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import math, torch, sys, argparse, json
from torch import nn
from torch.autograd import Variable
sys.path.append('Source')
from nn_gen import RNN
from data_gen import Data
from tqdm import tqdm

def demo(num_epochs, train_in, test_in, loss, optimizer, verbosity):
    cross_vals = []
    obj_vals = []
    train_accuracy = []
    test_accuracy = []

    #Training Loop
    correct_pred = 0
    for epoch in range(1, num_epochs+1):
        train_val, train_acc = model.backprop(train_in, train_out, loss, optimizer, correct_pred)
        obj_vals.append(train_val)
        #migh have to call accuracy function
        train_accuracy.append(train_acc)
        #########model.test??
        test_val, test_acc = model.test(test_in, test_out, loss, correct_pred)
        print(test_acc)
        #else:
        #    test_val, test_acc = model.test(x_validate, y_validate, loss)
        cross_vals.append(test_val)
        test_accuracy.append(test_acc)
        if verbosity >=2:
            if (epoch + 1)% int(0.1*num_epochs) == 0:
                print('Epoch [{}/{}]'.format(epoch+1, num_epochs)+\
                '\tTraining Loss: {:.4f}'.format(train_val)+\
                '\tTraining Accuracy: {:.2f}%'.format(train_acc))


    return obj_vals, cross_vals, train_accuracy, test_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ML Project Arguments')
    """

    parser.add_argument('--res-path', metavar='results',
                        help='path of results')
    """
    parser.add_argument('--data_train', default='Datasets/d=3_train.txt',
                        help='training dataset file path')
    parser.add_argument('--data_test', default='Datasets/d=3_test.txt',
                        help='testing dataset file path')
    parser.add_argument('--param', default='param/param.json',
                        help='parameter file name')
    parser.add_argument('-v', type=int, default=1, metavar='N',
                        help='verbosity (default: 1)')
    args = parser.parse_args()


#######################Data###############################################
    data_train = np.loadtxt('%s' %args.data_train)
    data_test = np.loadtxt('%s' %args.data_test)
    data = Data(data_train, data_test)

    error_synd_train, logical_err_train, error_synd_test, logical_err_test = data.err_synd_train, data.logical_err_train, data.err_synd_test, data.logical_err_test,
    train_in_row = len(error_synd_train[0, :])
    train_in_col = len(error_synd_train[:, 0])
    train_out_row = len(logical_err_train[0, :])
    train_out_col = len(logical_err_train[:, 0])

    test_in_row = len(error_synd_test[0, :])
    test_in_col = len(error_synd_test[:, 0])
    test_out_row = len(logical_err_test[0, :])
    test_out_col = len(logical_err_test[:, 0])

    with open(args.param) as paramfile:
        param = json.load(paramfile)

    if train_in_row < 256:
        layer_dim = 3
        lr = param['learning_rate_d=3']
        seq_len = param['seq_len_d=3']
        num_epochs = param['num_epochs_d=3']
    else:
        layer_dim = 6
        lr = param['learning_rate_d=5']
        seq_len = param['seq_len_d=5']
        num_epochs = param['num_epochs_d=5']



######################################################################################################
    train_in = torch.from_numpy(error_synd_train.reshape(-1, seq_len, train_in_row).astype(np.float32))
    train_out = torch.from_numpy(logical_err_train.reshape(-1, seq_len, train_out_row).astype(np.float32))
    train_out = torch.reshape(train_out, (train_out_col, train_out_row))
    print('\n',train_in.shape, train_out.shape,'\n')

    test_in = torch.from_numpy(error_synd_test.reshape(-1, seq_len, test_in_row).astype(np.float32))
    test_out = torch.from_numpy(logical_err_test.reshape(-1, seq_len, test_out_row).astype(np.float32))
    test_out = torch.reshape(test_out, (test_out_col, test_out_row))
    print('\n',test_in.shape, test_out.shape,'\n')


    #define data partitions
    input_dim, hidden_dim, layer_dim, output_dim = train_in_row, train_in_col, layer_dim, train_out_row
    #Genarating the Model
    model = RNN(input_dim, hidden_dim, layer_dim, output_dim)
    #Definint Optimizer and Loss function
    #optimizer = optim.SGD(model.parameters(), lr=parameters['learning_rate'])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.MSELoss(reduction = 'mean')#Using mean squared error between targets and output

    obj_vals, cross_vals, train_accuracy, test_accuracy = demo(num_epochs, train_in, test_in, loss, optimizer, args.v)

    if args.v:
        print('Final training loss: {:.4f}'.format(obj_vals[-1]))
        print('Final test loss: {:.4f}'.format(cross_vals[-1]))
        print('Final train accuracy: {:.2f}%'.format(train_accuracy[-1]))
        print('Final test accuracy: {:.2f}%'.format(test_accuracy[-1]))



    #net = RNN(input_dim, hidden_dim, layer_dim, output_dim)
    #criterion = nn.MSELoss(reduction = 'mean')
    #optimizer = torch.optim.Adam(net.parameters(), lr=lr)


#     loss_vals = []
#     obj_vals = []
#     train_accuracy = []
#     test_accuracy = []

#     for epoch in range(num_epochs):

#         # Clear gradients w.r.t. parameters
#         #optimizer.zero_grad()
#         # Forward pass to get output/logits
#         outputs = net(train_in) ####outputs not forwarded not forwarded

#         # Calculate Loss: softmax --> cross entropy loss
#         loss = criterion(outputs, train_out)

#         # Getting gradients w.r.t. parameters
#         #loss.backward() ####applied to obj vals

#         # Updating parameters
#         ##should pass in loss

#         #optimizer.step()
#         #Difference
#         #obj_vals.append(loss.item()) #########main dif

#         #test_val= net.test(test_in, test_out, loss)
#         #cross_vals.append(test_val)

#         #train_acc = get_accuracy(train_in, train_out) #maindifffffffff accuracy alculation
#         #test_acc = get_accuracy(test_in, test_out)

#         #train_accuracy.append(train_acc)
#         #test_accuracy.append(test_acc)
#         if args.v >=2:
#             if (epoch + 1)% int(0.1*num_epochs) == 0:
#                 print('Epoch [{}/{}]'.format(epoch+1, num_epochs)+                        '\tTraining Loss: {:.4f}'.format(loss.item()+                        '\tTraining Accuracy: {:.2f}%'.format(train_acc * 100)+                        '\tTest Accuracy: {:.2f}%'.format(test_acc * 100)))


#     if args.v:
#         print('Final training loss: {:.4f}'.format(object_vals[-1]))
#         print('Final test loss: {:.4f}'.format(cross_vals[-1]))
#         print('Final train accuracy: {:.2f}%'.format(train_accuracy[-1] * 100))
#         print('Final test accuracy: {:.2f}%'.format(test_accuracy[-1] * 100))
