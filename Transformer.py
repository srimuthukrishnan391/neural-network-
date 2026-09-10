import torch
import torch.nn as nn

class TransformerModel(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=2,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=2
        )

    def forward(self, x):
        return self.encoder(x)