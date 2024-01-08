import torch.nn as nn
import torch.nn.functional as F
import torch

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
        x = x.squeeze(1)
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
    
    def encode(self, x):
        return self.encoder(x)

import torch.nn as nn

class BasicTuner(nn.Module):
    def __init__(self, dim, drop_out):
        super(BasicTuner, self).__init__()

        self.layer1 = nn.Linear(dim, dim // 2)
        self.layer2 = nn.Linear(dim // 2, dim // 4)
        self.layer3 = nn.Linear(dim // 4, dim // 8)
        self.layer4 = nn.Linear(dim // 8, 1)
        self.beta = nn.Linear(1,1)
        
        # Other 
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.layer1(x)
        x = self.relu(x)        
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        
        return self.beta(x)


model_classes = {
        "AutoEncoder": AutoEncoder,
        "ConvAutoEncoder": ConvAutoEncoder,
        # "ResAutoEncoder": ResAutoEncoder,
    }

tuner_classes = {
    'BasicTuner': BasicTuner
    }

# create new model
def init_model(model_name, dim_size, drop_out):    
    model_class = model_classes[model_name]
    model = model_class(dim=dim_size, drop_out=drop_out)    
    return model

def init_tuner(tuner_name, encoder, dim_size, drop_out):
    # Get the output size of the encoder
    dummy_input = torch.randn(1, 1, dim_size)
    encoder_output_size = encoder(dummy_input).view(1, -1).size(1)

    # Create tuner
    tuner_class = model_classes[tuner_name]
    model = tuner_class(encoder, dim=encoder_output_size, drop_out=drop_out)    
    return model