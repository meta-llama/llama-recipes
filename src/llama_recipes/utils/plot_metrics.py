# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import matplotlib.pyplot as plt
import argparse
import os

def plot_metric(data, metric_name, x_label, y_label, title, colors):
    plt.figure(figsize=(7, 6))
    
    plt.plot(data[f'train_epoch_{metric_name}'], label=f'Train Epoch {metric_name.capitalize()}', color=colors[0])
    plt.plot(data[f'val_epoch_{metric_name}'], label=f'Validation Epoch {metric_name.capitalize()}', color=colors[1])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'Train and Validation Epoch {title}')
    plt.legend()
    plt.tight_layout()

def plot_single_metric_by_step(data, metric_name, x_label, y_label, title, color):
    plt.plot(data[f'{metric_name}'], label=f'{title}', color=color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

def plot_metrics_by_step(data, metric_name, x_label, y_label, colors):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plot_single_metric_by_step(data, f'train_step_{metric_name}', x_label, y_label, f'Train Step {metric_name.capitalize()}', colors[0])
    plt.subplot(1, 2, 2)
    plot_single_metric_by_step(data, f'val_step_{metric_name}', x_label, y_label, f'Validation Step {metric_name.capitalize()}', colors[1])
    plt.tight_layout()

    
def plot_metrics(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return

    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("Invalid JSON file.")
            return

    directory = os.path.dirname(file_path)
    filename_prefix = os.path.basename(file_path).split('.')[0]

    plot_metric(data, 'loss', 'Epoch', 'Loss', 'Loss', ['b', 'r'])
    plt.savefig(os.path.join(directory, f"{filename_prefix}_train_and_validation_loss.png"))
    plt.close()

    plot_metric(data, 'perplexity', 'Epoch', 'Perplexity', 'Perplexity', ['g', 'm'])
    plt.savefig(os.path.join(directory, f"{filename_prefix}_train_and_validation_perplexity.png"))
    plt.close()

    plot_metrics_by_step(data, 'loss', 'Step', 'Loss', ['b', 'r'])
    plt.savefig(os.path.join(directory, f"{filename_prefix}_train_and_validation_loss_by_step.png"))
    plt.close()

    plot_metrics_by_step(data, 'perplexity', 'Step', 'Loss', ['g', 'm'])
    plt.savefig(os.path.join(directory, f"{filename_prefix}_train_and_validation_perplexity_by_step.png"))
    plt.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot metrics from JSON file.')
    parser.add_argument('--file_path', required=True, type=str, help='Path to the metrics JSON file.')
    args = parser.parse_args()

    plot_metrics(args.file_path)
