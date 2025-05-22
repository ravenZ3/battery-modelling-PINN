# MIT Battery Dataset PINN Repository

This repository trains a Physics-Informed Neural Network (PINN) to predict battery State of Health (SOH) using the MIT Battery Dataset.

## Directory Structure
- `data/MIT_data/`: MIT dataset CSVs (e.g., `2017-05-12/2017-05-12_battery-1.csv`).
- `datasets/dataloader.py`: Loads and processes MIT dataset.
- `models/model.py`: Defines PINN and MLP models.
- `models/compare_models.py`: Compares MLP and CNN models.
- `scripts/main_mit.py`: Trains PINN on MIT dataset.
- `utils/util.py`: Logging and metrics utilities.
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
3. Place MIT dataset in `data/MIT_data/`.
4. Run training:
   ```bash
   python scripts/main_mit.py
   ```

## Requirements
- Python 3.7.10
- torch
- pandas
- numpy
- scikit-learn