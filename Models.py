import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, dim, drop_out):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(dim // 2),
            nn.Linear(dim // 2, dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(dim // 4)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(dim // 4, dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(dim // 2),
            nn.Linear(dim // 2, dim)
        )

    def forward(self, x):
        x = x.squeeze()
        # Forward pass through encoder
        x = self.encoder(x)

        # Forward pass through decoder
        x = self.decoder(x)

        return x.unsqueeze(1)

class ConvAutoEncoder(nn.Module):
    def __init__(self, dim, drop_out):
        super(ConvAutoEncoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1),
        )

        # Decoder
        self.decoder = nn.Sequential(
            # Add transposed convolutional layers to upsample
            nn.ConvTranspose1d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose1d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose1d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        # Add noise to the input
        x = self.dropout(x)

        # Forward pass through encoder
        x = self.encoder(x)

        # Forward pass through decoder
        x = self.decoder(x)

        return x
    
class ResAutoEncoder(nn.Module):
    def __init__(self, dim):
        super(ConvAutoEncoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1),
        )

        # Decoder
        self.decoder = nn.Sequential(
            # Add transposed convolutional layers to upsample
            nn.ConvTranspose1d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose1d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose1d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # Add noise to the input
        x = self.dropout(x)

        # Forward pass through encoder
        x = self.encoder(x)

        # Forward pass through decoder
        x = self.decoder(x)

        return x




model_classes = {
        "AutoEncoder": AutoEncoder,
        "ConvAutoEncoder": ConvAutoEncoder,
        "ResAutoEncoder": ResAutoEncoder,
    }

# create new model
def init_model(model_name, dim_size, drop_out):    
    model_class = model_classes[model_name]
    model = model_class(dim=dim_size, drop_out=drop_out)    
    return model

# load model checkpoint
def load_model(path):
    pass
