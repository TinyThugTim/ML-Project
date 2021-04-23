from torch import nn
import torch
import numpy as np

class RNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNN, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out

    # Reset function for the training weights
    # Use if the same network is trained multiple times
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def backprop(self, train_in, train_out, loss, optimizer):
        self.train()
        acc = self.get_accuracy(train_in, train_out)
        obj_val = loss(self.forward(train_in), train_out)

        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
        return obj_val.item(), acc

    def get_accuracy(self, data_in, data_out):
        """
        Given the output of the nn and the target,
        compute the accuracy which is the percent of correct predictions over all
        :param inputs: torch.tensor, input
        :param targets: torch.tensor, target
        :return: accuracy: float, persentage of correct predictions
        """
        correct_pred = 0
        with torch.no_grad():
            vect_inp = self.forward(data_in).detach().numpy()
            vect_tar = data_out.detach().numpy()

            total_pred = len(vect_inp)
            for inp, tar in zip(vect_inp, vect_tar):
                if np.where(inp == np.max(inp)) == np.where(tar == 1.):
                    correct_pred += 1

        accuracy = correct_pred / total_pred * 100
        return accuracy


    def test(self, x_test, y_test, loss):
        self.eval()

        with torch.no_grad():
            cross_val = loss(self.forward(x_test), y_test)
            acc = self.get_accuracy(x_test, y_test)

        return cross_val.item(), acc
