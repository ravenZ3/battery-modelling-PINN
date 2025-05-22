# MIT Battery PINN for SOH Prediction

This repository uses a Physics-Informed Neural Network (PINN) to predict battery State of Health (SOH) using the MIT Battery Dataset. SOH measures a battery’s current capacity relative to its nominal capacity, helping assess battery health.

## Directory Structure
- `data/MIT_data/`: MIT dataset CSVs (e.g., `2017-05-12/2017-05-12_battery-1.csv`).
- `datasets/dataloader.py`: Loads and processes MIT dataset.
- `models/model.py`: Defines PINN and MLP models.
- `models/compare_models.py`: Compares MLP and CNN models.
- `scripts/main_mit.py`: Trains PINN on MIT dataset.
- `utils/util.py`: Logging and metrics utilities.
- `plot/plot_MIT.py`: Plots SOH predictions and results.
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
4. Run training:
   ```bash
   python scripts/main_mit.py
   ```
5. Plot results:
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
The MIT Battery Dataset contains CSV files with measurements (e.g., voltage, current, temperature, capacity) over charge/discharge cycles. Data is organized in batches (`2017-05-12`, etc.).

## Outputs
- Training saves SOH predictions and logs to `results/MIT results/ExperimentX`.
- Plots (SOH vs. cycle index) are saved to `results/MIT_plots/`.