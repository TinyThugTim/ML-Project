import numpy as np
import matplotlib.pyplot as plt
import math, torch, sys, argparse, json
from torch import nn
from torch.autograd import Variable
sys.path.append('Source')
from nn_gen import RNN
from data_gen import Data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ML Project Arguments')
    """

    parser.add_argument('--res-path', metavar='results',
                        help='path of results')
    """
    parser.add_argument('--data', default='Datasets/d=3.txt',
                        help='training dataset file path')
    parser.add_argument('--param', default='param/param.json',
                        help='parameter file name')
    parser.add_argument('-v', type=int, default=1, metavar='N',
                        help='verbosity (default: 1)')
    args = parser.parse_args()

    data = np.loadtxt('%s' %args.data)
    for_training = Data(data)

    error_synd_train, logical_err_train = for_training.err_synd_train, for_training.logical_err_train
    train_in_row = len(error_synd_train[0, :])
    train_in_col = len(error_synd_train[:, 0])
    train_out_row = len(logical_err_train[0, :])
    train_out_col = len(logical_err_train[:, 0])

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


    train_in = torch.from_numpy(error_synd_train.reshape(-1, seq_len, train_in_row).astype(np.float32))
    train_out = torch.from_numpy(logical_err_train.reshape(-1, seq_len, train_out_row).astype(np.float32))
    train_out = torch.reshape(train_out, (train_out_col, train_out_row))
    print('\n',train_in.shape, train_out.shape,'\n')

    input_dim, hidden_dim, layer_dim, output_dim = train_in_row, train_in_col, layer_dim, train_out_row
    net = RNN(input_dim, hidden_dim, layer_dim, output_dim)
    criterion = nn.MSELoss(reduction = 'mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    """
    for e in range(1000):
        out = net(train_in)
        loss = criterion(out, train_out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (e + 1)% 100 == 0:
            print(loss)
    """

    iter = 0
    loss_vals = []
    for epoch in range(num_epochs):
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = net(train_in)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, train_out)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        loss_vals.append(loss.item())
        iter += 1

        if args.v >=2:
            if (epoch + 1)% int(0.1*num_epochs) == 0:
                print('Epoch [{}/{}]'.format(epoch+1, num_epochs)+\
                          '\tTraining Loss: {:.4f}'.format(loss.item()))
    if args.v:
        print('Final training loss: {:.4f}'.format(loss_vals[-1]))
