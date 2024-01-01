import torch
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt  # Import the plotting library

def train(model, optimizer, criterion, epochs, train_loader, val_loader, device, save_path='best_model.ckpt'):
    
    model.to(device)
    loss_history = []

    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the specified device

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        # Store the average loss for the epoch in the loss history
        average_loss = running_loss / len(train_loader)
        loss_history.append(average_loss)

    print('Training Finished')

    # Plot the loss history
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.savefig('TrainingLoss.png')

    torch.save(model.state_dict(), save_path)
    print(f'Trained models saved. Model path: {save_path}')

    return model, save_path
