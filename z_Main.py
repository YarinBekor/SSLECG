import torch
import numpy as np
import matplotlib.pyplot as plt

from Dataset import *
from Trainer import *
from Models import *
from Utils import *

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def main():
    mode = 'train'
    # mode = 'test'
    ecg_tracings_path = '/MLAIM/databases/CODE/relevant_samples_data.pkl'
    pre_trained_model_path = 'best_model.ckpt'

    epochs = 20
    batch_size = 32
    data_splits = [0.8, 0.1, 0.1]
    elements_in_group = 8
    mask_p = 0.3
    device='cuda'

    train_loader, val_loader, test_loader, num_features = get_data_loaders(data_path=ecg_tracings_path,
                                                                           splits=data_splits,
                                                                           batch_size=batch_size,
                                                                           )
    if mode == 'train':
        model = init_model(num_features)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)


        train_outputs = train(model=model,
                            optimizer=optimizer,
                            criterion=criterion,
                            epochs=epochs,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            device=device)
        
        last_model, best_model_path = train_outputs
        # best_model = load_model(best_model_path)
        
        test_acc = evaluate_model(model=last_model,
                                loader=test_loader,
                                criterion=criterion,
                                device=device)
        


        print(f'test error: {test_acc}')

    model = init_model(num_features)
    model.load_state_dict(torch.load(pre_trained_model_path))
    model.eval()  # Set the model to evaluation mode

    # Plot the results for one test sample
    plot_test_sample(model, test_loader)
        



if __name__ == '__main__':
    main()