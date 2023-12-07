import torch

from tqdm import tqdm

from copy import deepcopy


def train(model, optimizer, criterion,
          epochs, train_loader, val_loader,
          save_path='best_model.ckpt'):
    
    loss_history = []

    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

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

    print('Training Finished')

    torch.save(model.state_dict(), save_path)

    return model, save_path


    

