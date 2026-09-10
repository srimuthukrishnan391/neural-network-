import torch
import torch.nn as nn

class RNNModel(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.rnn = nn.RNN(
            input_size,
            64,
            batch_first=True
        )

        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        output, hidden = self.rnn(x)
        return self.fc(hidden[-1])