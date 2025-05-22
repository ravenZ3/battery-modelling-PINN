import numpy as np
import matplotlib.pyplot as plt
import os

def plot_soh_results(experiment_dir, save_dir):
    """
    Plot predicted vs. actual SOH for an experiment.
    :param experiment_dir: Path to experiment folder (e.g., results/MIT results/Experiment1)
    :param save_dir: Where to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load predicted and actual SOH
    pred_file = os.path.join(experiment_dir, 'pred_label.npy')
    true_file = os.path.join(experiment_dir, 'true_label.npy')
    
    if not os.path.exists(pred_file) or not os.path.exists(true_file):
        print(f"Missing pred_label.npy or true_label.npy in {experiment_dir}")
        return

    pred_soh = np.load(pred_file).flatten()
    true_soh = np.load(true_file).flatten()
    cycles = np.arange(len(true_soh))  # Generate cycle indices

    # Plot SOH
    plt.figure(figsize=(10, 6))
    plt.plot(cycles, true_soh, label='Actual SOH', color='blue')
    plt.plot(cycles, pred_soh, label='Predicted SOH', color='orange', linestyle='--')
    plt.xlabel('Cycle Index')
    plt.ylabel('State of Health (SOH)')
    plt.title(f'SOH Prediction - {os.path.basename(experiment_dir)}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{os.path.basename(experiment_dir)}_soh_plot.png'))
    plt.close()

def plot_loss(experiment_dir, save_dir):
    """
    Plot training loss from logging.txt, if available.
    """
    log_file = os.path.join(experiment_dir, 'logging.txt')
    if not os.path.exists(log_file):
        print(f"No logging.txt in {experiment_dir}")
        return

    # Parse log file for loss (format unknown, assuming "loss: value" per line)
    epochs = []
    losses = []
    with open(log_file, 'r') as f:
        for line in f:
            if 'loss:' in line:
                try:
                    epoch = int(line.split('epoch:')[1].split(',')[0].strip())
                    loss = float(line.split('loss:')[1].split(',')[0].strip())
                    epochs.append(epoch)
                    losses.append(loss)
                except (IndexError, ValueError):
                    continue

    if not epochs or not losses:
        print(f"No loss data found in {log_file}")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, label='Training Loss', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss - {os.path.basename(experiment_dir)}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{os.path.basename(experiment_dir)}_loss_plot.png'))
    plt.close()

def main():
    result_root = 'results/MIT results'
    plot_save_dir = 'results/MIT_plots'
    os.makedirs(plot_save_dir, exist_ok=True)

    for exp in range(1, 11):
        exp_dir = os.path.join(result_root, f'Experiment{exp}')
        if os.path.exists(exp_dir):
            plot_soh_results(exp_dir, plot_save_dir)
            plot_loss(exp_dir, plot_save_dir)
            print(f"Plotted {exp_dir}")
        else:
            print(f"{exp_dir} not found")

if __name__ == '__main__':
    main()