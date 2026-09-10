import torch
import torch.nn as nn

class GRUModel(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.gru = nn.GRU(
            input_size,
            64,
            batch_first=True
        )

        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        output, hidden = self.gru(x)
        return self.fc(hidden[-1])