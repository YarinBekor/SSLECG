import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

def evaluate_model(model, loader, criterion):
    total_loss = 0.0
    total_samples = 0

    # set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        for data in loader:
            masked, raw = data
            outputs = model(masked)
            loss = criterion(outputs, raw)

            total_loss += loss.item()
            total_samples += raw.size(0)

    average_loss = total_loss / total_samples
    return average_loss

def plot_test_sample(model, test_loader):
    # Get one batch from the test loader
    bert_signal, raw_signal = next(iter(test_loader))
    
    # Forward pass to get model predictions
    generated_signal = model(bert_signal)
    
    # Convert torch tensors to numpy arrays
    bert_signal = bert_signal.numpy()
    raw_signal = raw_signal.numpy()
    generated_signal = generated_signal.detach().numpy()
    
    # Plot the signals
    save_path = 'test_sample_result.png'
    plot_output(raw_signal[0], bert_signal[0], generated_signal[0], save_path)


def plot_output(raw_signal, bert_signal, generated_signal, save_path):    
    # Plot the signals
    plt.figure(figsize=(10, 6))
    plt.plot(raw_signal, label='Expected Signal', color = 'red')
    plt.plot(generated_signal, label='Generated Signal', color = 'green')
    # plt.plot(bert_signal, label='Original Signal', color = 'blue')
    
    # Customize the plot
    plt.title('Comparison of Signals')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    
    # Save the plot
    plt.savefig(save_path)
