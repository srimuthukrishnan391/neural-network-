import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 13 * 13, 10)
        )

    def forward(self, x):
        return self.network(x)