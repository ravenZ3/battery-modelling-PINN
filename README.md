# MIT Battery Dataset Physics-Informed Neural Network (PINN) Repository

This repository trains a Physics-Informed Neural Network (PINN), a type of neural network that incorporates physical laws as constraints during training, to predict battery State of Health (SOH) using the MIT Battery Dataset.

## Directory Structure
- `data/MIT_data/`: MIT dataset CSVs (e.g., `2017-05-12/2017-05-12_battery-1.csv`).
- `datasets/dataloader.py`: Loads and processes MIT dataset.
- `models/model.py`: Defines PINN and MLP models.
- `models/compare_models.py`: Compares MLP and PINN models.
- `scripts/main_mit.py`: Trains PINN on MIT dataset.
- `utils/util.py`: Logging and metrics utilities.
- `results/`: Training outputs, including metrics, plots, and logs (e.g., `MIT results/ExperimentX`).

## Setup
1. Activate conda environment (if you are new to Conda, see [Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)):
   ```bash
   conda activate pinn4soh
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the MIT dataset from [MIT Battery Dataset](https://data.mit.edu/) and place it in `data/MIT_data/`.
4. Run training:
   ```bash
   python scripts/main_mit.py
   ```

## Requirements==0.24.2