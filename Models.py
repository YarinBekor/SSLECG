import torch

import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, dim):
        super(AutoEncoder, self).__init__()

                # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(dim // 2),
            nn.Linear(dim // 2, dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(dim // 4),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(dim // 4, dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(dim // 2),
            nn.Linear(dim // 2, dim),
            # nn.Sigmoid()  # Sigmoid activation for reconstruction in the range [0, 1]
        )

    def forward(self, x):
        # Forward pass through encoder
        x = self.encoder(x)

        # Forward pass through decoder
        x = self.decoder(x)

        return x

    
# create new model
def init_model(dim_size):
    model = AutoEncoder(dim=dim_size)
    return model

# load model checkpoint
def load_model(path):
    pass
