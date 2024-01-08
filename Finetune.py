from Dataset import *
from Trainer import *
from Models import *
from Utils import *
from Config import *
from FineTune import *


def main_fine_tuner(encoder):

    downstrim_data_path = '/MLAIM/databases/CODE/relevant_samples_data.pkl'

    device='cuda'

    tuner_name = 'FineTuner'
    params = model_params[tuner_name]
    
    intro_printer(tuner_name, params, device)

    batch_size = params['batch size']
    criterion = params['criterion']
    optimizer = params['optimizer']
    data_splits = params['splits']
    drop_out = params['drop out']
    epochs = params['epochs']
    lr = params['lr']

    train_loader, val_loader, test_loader, num_features = get_data_loaders(data_path=downstrim_data_path,
                                                                           splits=data_splits,
                                                                           batch_size=batch_size,
                                                                           )
    

    pre_trained_model_path = 'best_model.ckpt'
    foundational_model = init_model(num_features)
    foundational_model.load_state_dict(torch.load(pre_trained_model_path))
    model.eval()

    model = init_model(tuner_name, model.encoder(), num_features, drop_out)
    
    optimizer = optimizer(model.parameters(), lr=lr)
    
    training_output = train_tuner(model=model,
                        encoder=encoder,
                        optimizer=optimizer,
                        criterion=criterion,
                        epochs=epochs,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        device=device)
    
    evaluate_model(training_output, test_loader, criterion, device)
        


