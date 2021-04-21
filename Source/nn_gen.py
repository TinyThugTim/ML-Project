from torch import nn
import torch

class RNN(nn.Module):
    """
    def __init__(self, n_dim, seq_len, n_hidden, n_layers=1):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(n_dim, n_hidden, n_layers, batch_first=True)
        self.fc = nn.Linear(n_hidden * seq_len, n_dim)

    def forward(self, x, times=1):
        x, h = self.rnn(x)
        x = self.fc(x)
        outs = []
        outs.append(x)
        for i in range(times-1):
            x, h = self.rnn(x, h)
            x = self.fc(x)
            outs.append(x)
        if times > 1:
            return outs
        b, s, h = x.shape
        return x
    """
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
