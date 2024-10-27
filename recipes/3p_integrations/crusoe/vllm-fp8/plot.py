import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def extract_info_from_filename(filename):
    pattern = r'(?P<backend>[^-]+)-(?P<qps>\d+\.\d+)qps-(?P<model>.+)-(?P<date>\d{8}-\d{6})\.json'
    match = re.match(pattern, filename)
    if match:
        return {
            'qps': float(match.group('qps')),
            'model': match.group('model')
        }
    return None

def read_json_files(directory):
    data_tpot = defaultdict(list)
    data_ttft = defaultdict(list)
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            file_info = extract_info_from_filename(filename)
            if file_info:
                with open(filepath, 'r') as file:
                    json_data = json.load(file)
                    median_tpot = json_data.get('median_tpot_ms')
                    std_tpot = json_data.get('std_tpot_ms')
                    median_ttft = json_data.get('median_ttft_ms')
                    std_ttft = json_data.get('std_ttft_ms')
                    if all(v is not None for v in [median_tpot, std_tpot, median_ttft, std_ttft]):
                        data_tpot[file_info['model']].append((file_info['qps'], median_tpot, std_tpot))
                        data_ttft[file_info['model']].append((file_info['qps'], median_ttft, std_ttft))
    return {
        'tpot': {model: sorted(points) for model, points in data_tpot.items()},
        'ttft': {model: sorted(points) for model, points in data_ttft.items()}
    }

def create_chart(data, metric, filename):
    plt.figure(figsize=(12, 6))
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(data)))
    for (model, points), color in zip(data.items(), colors):
        qps_values, median_values, std_values = zip(*points)
        plt.errorbar(qps_values, median_values, yerr=std_values, fmt='o-', capsize=5, capthick=2, label=model, color=color)
        plt.fill_between(qps_values, 
                         np.array(median_values) - np.array(std_values),
                         np.array(median_values) + np.array(std_values),
                         alpha=0.2, color=color)

    plt.xlabel('QPS (Queries Per Second)')
    plt.ylabel(f'Median {metric.upper()} (ms)')
    plt.title(f'Median {metric.upper()} vs QPS with Standard Deviation')
    plt.grid(True)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    directory = './'
    data = read_json_files(directory)
    if data['tpot'] and data['ttft']:
        create_chart(data['tpot'], 'tpot', 'tpot_vs_qps_chart.png')
        create_chart(data['ttft'], 'ttft', 'ttft_vs_qps_chart.png')
        print("Charts have been saved as 'tpot_vs_qps_chart.png' and 'ttft_vs_qps_chart.png'")
    else:
        print("No valid data found in the specified directory.")

if __name__ == "__main__":
    main()