import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def load_feature_labels(config_path):
    feature_labels = {}
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found.")
        return feature_labels
    
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '-' not in line:
                continue
            try:
                parts = line.split('-', 1)
                idx = int(parts[0].strip())
                name = parts[1].strip()
                feature_labels[idx] = name
            except ValueError:
                continue
    return feature_labels

def plot_comparison(files, labels=None):
    data_list = []
    for f in files:
        if not os.path.exists(f):
            print(f"Error: File {f} not found.")
            return
        try:
            data = np.load(f)
            if data.shape[0] == 1:
                data = data.squeeze(0) 
            data_list.append(data)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            return

    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(files))]
    
    if not data_list:
        print("No data loaded.")
        return

    # Load feature labels
    feature_labels = load_feature_labels('config/feature_labels.yml')

    # Determine dimensions to plot
    if len(data_list) == 1:
        total_dims = data_list[0].shape[1]
    else:
        total_dims = min(data_list[0].shape[1], data_list[1].shape[1])
    
    total_dims = 43
    indices = list(range(total_dims)) # 0 to 7
    
    # Remove duplicates and sort
    indices = sorted(list(set(indices)))
    
    num_plots = len(indices)
    cols = 4
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 3 * rows))
    axes = axes.flatten()
    
    for idx, dim_i in enumerate(indices):
        if idx >= len(axes): break
        ax = axes[idx]
        
        for j, data in enumerate(data_list):
            if dim_i < data.shape[1]:
                ax.plot(data[:, dim_i], label=labels[j], alpha=0.7)
        
        title = feature_labels.get(dim_i, f'Feature {dim_i}')
        ax.set_title(f'{dim_i}: {title}')
        if idx == 0:
            ax.legend()
        ax.grid(True)
        
    # Hide unused subplots
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig('comparison_plot.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare trajectory files (.npy)")
    parser.add_argument("files", nargs='+', help="Paths to .npy files to compare")
    parser.add_argument("--labels", nargs='+', help="Labels for the runs")
    args = parser.parse_args()
    
    if args.labels and len(args.labels) != len(args.files):
        print("Warning: Number of labels does not match number of files. Using default labels.")
        args.labels = None
        
    plot_comparison(args.files, args.labels)
