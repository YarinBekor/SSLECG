import torch
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt  # Import the plotting library

def train(model, optimizer, criterion, epochs, train_loader, val_loader, device, save_path='best_model.ckpt'):

    model.to(device)
    loss_history = {'train': [], 'validation': []}

    for epoch in tqdm(range(epochs)):
        model.train()
        running_train_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

            # Print statistics
            if i % 100 == 99:
                print(f'Train Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_train_loss / 100:.3f}')
                running_train_loss = 0.0

        average_train_loss = running_train_loss / len(train_loader)
        loss_history['train'].append(average_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = val_data
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)

                running_val_loss += val_loss.item()

            # Store the average validation loss for the epoch in the loss history
            average_val_loss = running_val_loss / len(val_loader)
            loss_history['validation'].append(average_val_loss)

            print(f'Validation Epoch: {epoch + 1}, Loss: {average_val_loss:.3f}')

    torch.save(model.state_dict(), save_path)
    
    print(f'Trained model saved. Model path: {save_path}')

    return model, loss_history