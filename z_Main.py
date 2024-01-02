from Dataset import *
from Trainer import *
from Models import *
from Utils import *
from Config import *

import torch
import torch.nn as nn

def main():

    ecg_tracings_path = '/MLAIM/databases/CODE/relevant_samples_data.pkl'
    pre_trained_model_path = 'best_model.ckpt'
    device='cuda'

    model_name = 'ConvAutoEncoder'
    params = model_params[model_name]
    
    intro_printer(model_name, params)

    batch_size = params['batch size']
    criterion = params['criterion']
    optimizer = params['optimizer']
    data_splits = params['splits']
    drop_out = params['drop out']
    epochs = params['epochs']
    lr = params['lr']

    train_loader, val_loader, test_loader, num_features = get_data_loaders(data_path=ecg_tracings_path,
                                                                           splits=data_splits,
                                                                           batch_size=batch_size,
                                                                           )

    model = init_model(model_name, num_features, drop_out)
    
    optimizer = optimizer(model.parameters(), lr=lr)

    training_output = train(model=model,
                        optimizer=optimizer,
                        criterion=criterion,
                        epochs=epochs,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        device=device)
    
    model.eval()
    evaluate_model(training_output, test_loader, criterion, device)
        



if __name__ == '__main__':
    main()