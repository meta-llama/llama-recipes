import json
import matplotlib.pyplot as plt
import sys
import os

def plot_metrics(file_path):

    # Read the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Get directory and filename information
    directory = os.path.dirname(file_path)
    filename_prefix = os.path.basename(file_path).split('.')[0]

    # Plotting metrics for training and validation step loss
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(data['train_step_loss'], label='Train Step Loss', color='b')
    plt.plot(data['val_step_loss'], label='Validation Step Loss', color='r')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Train and Validation Step Loss')
    plt.legend()

    # Plotting metrics for training and validation epoch loss
    plt.subplot(1, 2, 2)
    plt.plot(data['train_epoch_loss'], label='Train Epoch Loss', color='b')
    plt.plot(data['val_epoch_loss'], label='Validation Epoch Loss', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Epoch Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"{filename_prefix}_train_and_validation_loss.png"))
    plt.close()

    # Plotting perplexity
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(data['train_step_perplexity'],
             label='Train Step Perplexity', color='g')
    plt.plot(data['val_step_perplexity'],
             label='Validation Step Perplexity', color='m')
    plt.xlabel('Step')
    plt.ylabel('Perplexity')
    plt.title('Train and Validation Step Perplexity')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(data['train_epoch_perplexity'],
             label='Train Epoch Perplexity', color='g')
    plt.plot(data['val_epoch_perplexity'],
             label='Validation Epoch Perplexity', color='m')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Train and Validation Epoch Perplexity')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"{filename_prefix}_train_and_validation_perplexity.png"))
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_metrics_json>")
        sys.exit(1)

    file_path = sys.argv[1]
    plot_metrics(file_path)