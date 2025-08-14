# LSTM ADT Forecast

Modular repo for forecasting Annual Daily Traffic (ADT) using an LSTM.

## Objectives

1. **Clean & Reusable Pipeline** – Provide a clear separation between data loading, modeling, evaluation, and plotting so components are easy to replace or extend.
2. **Reproducible Training & Inference** – One-command scripts for training and prediction, plus notebooks for quick exploration.
3. **Meaningful Evaluation** – Standard regression metrics (MAE, MAPE, MDAPE, RMSE) and comparison plots (train/test/predictions/residuals).
4. **Extensibility** – Support future improvements such as Keras Core / PyTorch backends, walk-forward backtesting, hyperparameter search, exogenous covariates, uncertainty intervals, and CI/CD.


## Structure
```
│
├── code/                      # All main scripts
│   ├── main.py                 # Main pipeline runner
│   ├── train.py                # Model training logic
│   ├── predict.py              # Inference logic
│   └── __init__.py
│
├── utils/                      # Utility/helper functions
│   ├── data_loader.py          # Loading & preprocessing
│   ├── plot_utils.py           # Plotting functions
│   ├── metrics.py              # Metrics computation
│   └── __init__.py
│
├── input_data/                 # Raw and processed data
│   ├── raw/                    # Raw datasets
│   └── processed/              # Processed features
│
├── output/                     # Results & artifacts
│   ├── models/                 # Saved models
│   ├── figures/                # Plots
│   └── logs/                   # Training logs
│
├── notebooks/                  # Jupyter notebooks
│
├── README.md
├── requirements.txt
└── .gitignore

```
## Quickstart
```bash
"Create virtual environment" - .venv\Scripts\activate
pip install -r requirements.txt

# run training
python code/train.py --csv input_data/raw/sample_adt.csv --sequence_length 12

# run predict on the saved model (example path printed after training)
python code/predict.py --csv input_data/raw/sample_adt.csv --model_path output/models/latest_lstm.keras
```
