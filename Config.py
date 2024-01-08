import torch.nn as nn
import torch.optim as optim

model_params ={
    'AutoEncoder' : {
        'lr' : 0.001 ,
        'drop out' : 0.2 ,
        'epochs' : 20 ,
        'batch size' : 32 ,
        'splits' : [0.8, 0.1, 0.1],
        'criterion' : nn.MSELoss(),
        'optimizer' : optim.Adam
                    },
    'ConvAutoEncoder' : {
        'lr' : 0.001 ,
        'drop out' : 0 ,
        'epochs' : 100 ,
        'batch size' : 128 ,
        'splits' : [0.8, 0.1, 0.1],
        'criterion' : nn.MSELoss(),
        'optimizer' : optim.Adam
                    }
}