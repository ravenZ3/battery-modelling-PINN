# MIT Battery PINN for SOH Prediction

This repository uses a Physics-Informed Neural Network (PINN) to predict battery State of Health (SOH) from the MIT Battery Dataset. SOH is the ratio of current capacity to nominal capacity (1.1 Ah), indicating battery health.

## Directory Structure
- `data/MIT_data/`: MIT dataset CSVs (e.g., `2017-05-12/2017-05-12_battery-1.csv`).
- `datasets/dataloader.py`: Loads and processes MIT dataset.
- `models/model.py`: Defines PINN and MLP models.
- `models/compare_models.py`: Trains and compares MLP and CNN models against PINN.
- `scripts/main_mit.py`: Trains PINN on MIT dataset.
- `utils/util.py`: Logging and metrics utilities.
- `plot/plot_MIT.py`: Plots SOH predictions and training loss.
- `results/`: Training outputs (e.g., `MIT results/ExperimentX`).

## Setup
1. Activate conda environment:
   ```bash
   conda activate pinn4soh
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place MIT dataset in `data/MIT_data/` (batches: `2017-05-12`, `2017-06-30`, `2018-04-12`).
4. Run PINN training:
   ```bash
   python scripts/main_mit.py
   ```
5. Compare MLP and CNN models:
   ```bash
   python models/compare_models.py
   ```
6. Plot results:
   ```bash
   python plot/plot_MIT.py
   ```

## Requirements
- Python 3.7.10
- torch
- pandas
- numpy
- scikit-learn
- matplotlib

## Dataset
The MIT Battery Dataset includes CSV files with features like voltage, current, temperature, and capacity over cycles. Example features: `voltage mean`, `current std`, `CC Q`, `capacity`.

## Outputs
- PINN training saves to `results/MIT results/ExperimentX`:
  - `pred_label.npy`: Predicted SOH values.
  - `true_label.npy`: Actual SOH values.
  - `logging.txt`: Training logs (e.g., loss per epoch).
  - `model.pth`: Trained model weights.
- Model comparison saves to `results/comparison/`:
  - `mlp_predictions.npy`, `cnn_predictions.npy`: MLP and CNN predictions.
  - `rmse_comparison.txt`: RMSE for MLP, CNN, and PINN.
- Plots (e.g., SOH vs. cycle, training loss) are saved to `results/MIT_plots/`.