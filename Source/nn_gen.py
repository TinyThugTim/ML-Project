class RNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.hid_fc = nn.Linear(185, 128)
        self.out_fc = nn.Linear(185, n_classes)
        self.softmax = nn.LogSoftmax()
    
    def forward(self, inputs, hidden):
        inputs = inputs.view(1,-1)
        combined = torch.cat([inputs, hidden], dim=1)
        hid_out = self.hid_fc(combined)
        out = self.out_fc(combined)
        out = self.softmax(out)
        return out, hid_out


