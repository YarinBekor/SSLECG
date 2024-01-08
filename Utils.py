import torch
import matplotlib.pyplot as plt

def evaluate_model(training_output, test_loader, criterion, device):
    model, loss_history = training_output
    model.eval()

    calculate_acc(model, test_loader, criterion, device)
    plot_loss(loss_history)
    plot_test_sample(model, test_loader, device)
    
   
def calculate_acc(model, test_loader, criterion, device):
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for data in test_loader:
            masked, raw = data
            masked, raw = masked.to(device=device), raw.to(device=device)

            outputs = model(masked)
            loss = criterion(outputs, raw)

            total_loss += loss.item()
            total_samples += raw.size(0)

    average_loss = total_loss / total_samples
    
    print()
    print(f'test error using the {criterion} metric: {average_loss}')


def plot_loss(loss_history):
    train_loss, val_loss = loss_history['train'], loss_history['validation']
    
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.savefig('TrainingLoss.png')

    print('Loss plot saved.')


def plot_test_sample(model, test_loader, device):
    for i in range(3):
        for test_masked, test_raw in test_loader:  
            masked_signal = test_masked[i]
            raw_signal = test_raw[i]
        
        masked_signal = masked_signal.to(device=device)
        generated_signal = model(masked_signal)
        
        # Convert torch tensors to numpy arrays
        raw_signal = raw_signal.numpy()
        generated_signal = generated_signal.detach().cpu().numpy()
        
        # Plot the signals
        save_path = f'test_sample_result_{i}.png'
        plot_output(raw_signal[0], generated_signal[0], save_path)
    
    print('All plots saved.')


def plot_output(raw_signal, generated_signal, save_path):    

    plt.figure(figsize=(10, 6))
    plt.plot(raw_signal, label='Expected Signal', color = 'red')
    plt.plot(generated_signal, label='Generated Signal', color = 'green')
    plt.title('Comparison of Signalss')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()    
    plt.savefig(save_path)


def intro_printer(model_name, params, device):
    highlight_start = "\033[93m"
    highlight_end = "\033[0m"
    model_for_print = f'{highlight_start}{model_name}{highlight_end}'
    device_for_print = f'{highlight_start}{device}{highlight_end}'
    print()
    print()
    print(f'Running model {model_for_print} on {device_for_print} with the following configs:')
    print()
    print(params)
    print()
