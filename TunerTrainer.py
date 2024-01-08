import torch
from tqdm import tqdm

def train_tuner(model, encoder, optimizer, criterion, epochs, train_loader, val_loader, device, save_path='best_model1.ckpt', patience=5):
    model.to(device)
    encoder.to(device)
    loss_history = {'train': [], 'validation': []}

    best_val_loss = float('inf')
    counter = 0 

    for epoch in tqdm(range(epochs)):
        model.train()
        running_train_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.no_grad:
                encoded_inputs = encoder(inputs)
            
            optimizer.zero_grad()
            outputs = model(encoded_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

            # Print statistics
            if i % 1000 == 999:
                print(f'Train Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_train_loss / 1000:.3f}')
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

                with torch.no_grad:
                    encoded_val_inputs = encoder(val_inputs)
            
                val_outputs = model(encoded_val_inputs)
                val_loss = criterion(val_outputs, val_labels)

                running_val_loss += val_loss.item()

            # Store the average validation loss for the epoch in the loss history
            average_val_loss = running_val_loss / len(val_loader)
            loss_history['validation'].append(average_val_loss)

            print(f'Validation Epoch:: {epoch + 1}, Loss: {average_val_loss:.3f}')

            # Early stopping check
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f'Early stopping! No improvement for {patience} epochs.')
                    break

    torch.save(model.state_dict(), save_path)
    
    print(f'Trained model saved. Model path: {save_path}')

    return model, loss_history
