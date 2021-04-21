import numpy as np
import matplotlib.pyplot as plt
import math, torch, sys
from torch import nn
from torch.autograd import Variable
sys.path.append('Source')
from nn_gen import RNN
from data_gen import Data

data = np.loadtxt('Datasets/d=3.txt')
for_training = Data(data)
error_synd_train, logical_err_train = for_training.err_synd_train, for_training.logical_err_train

seq_len = 1

train_in = torch.from_numpy(error_synd_train.reshape(-1, seq_len, 8).astype(np.float32))
train_out = torch.from_numpy(logical_err_train.reshape(-1, seq_len, 4).astype(np.float32))
print('\n',train_in.shape, train_out.shape,'\n')
# so far only training but will get to testing once noise model syndrome measurements are obtained

input_dim, hidden_dim, layer_dim, output_dim = 8, 128, 2, 4
learning_rate = 2e-3
net = RNN(input_dim, hidden_dim, layer_dim, output_dim)
criterion = nn.MSELoss(reduction = 'mean')
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

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

num_epochs = 5000
iter = 0
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

    iter += 1

    if (epoch + 1)% 500 == 0:
        print(loss.item())
