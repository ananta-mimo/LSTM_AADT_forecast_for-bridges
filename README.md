# LSTM AADT Forecast

A modular deep learning pipeline for forecasting **Annual Average Daily Traffic (AADT)** using a stacked LSTM network. Built as part of PhD research on bridge traffic demand modeling, this repo demonstrates an end-to-end time-series forecasting workflow from raw count data to multi-year predictions.

---

## Problem Statement

Bridge asset managers need reliable multi-year traffic forecasts to plan maintenance schedules, estimate load demand, and prioritize capital investments. Traditional trend extrapolation methods fail to capture non-linear growth patterns. This project applies a stacked LSTM to annual traffic count data to produce data-driven AADT forecasts with quantified error metrics.

---

## Model Architecture

A three-layer stacked LSTM followed by two dense layers:

```
Input sequence (n timesteps × features)
        ↓
LSTM layer 1  (return_sequences=True)
        ↓
LSTM layer 2  (return_sequences=True)
        ↓
LSTM layer 3  (return_sequences=False)
        ↓
Dense (ReLU activation)
        ↓
Dense (linear output → AADT forecast)
```

- Optimizer: Adam
- Loss: Mean Squared Error
- Early stopping on validation loss (patience configurable)
- MinMax scaling applied before training, inverse-transformed for evaluation

---

## Repository Structure

```
lstm-aadt-forecast/
│
├── code/
│   ├── main.py           # Entry point — calls train pipeline
│   ├── train.py          # Full training loop: load → scale → sequence → fit → evaluate
│   ├── predict.py        # Inference on saved model: outputs next-year AADT forecast
│   └── __init__.py
│
├── utils/
│   ├── data_loader.py    # CSV loading and preprocessing
│   ├── metrics.py        # MAE, MAPE, MDAPE, RMSE computation
│   ├── plot_utils.py     # Training history, prediction vs actual, residual plots
│   └── __init__.py
│
├── notebooks/
│   ├── EDA.ipynb         # Exploratory analysis: trends, seasonality, stationarity
│   └── Model_Training.ipynb  # Interactive training with visualizations
│
├── input_data/
│   └── raw/
│       └── sample_aadt.csv   # Sample dataset (synthetic — real data not included)
│
├── output/
│   ├── models/           # Saved .keras model files
│   ├── figures/          # Output plots
│   └── logs/             # Training logs
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart

**1. Clone and set up environment:**

```bash
git clone https://github.com/ananta-mimo/LSTM_AADT_forecast_for-bridges.git
cd LSTM_AADT_forecast_for-bridges
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

**2. Train the model:**

```bash
python code/train.py \
  --csv input_data/raw/sample_aadt.csv \
  --sequence_length 12 \
  --train_split 0.7 \
  --lstm_units 64 \
  --max_epochs 300 \
  --patience 25 \
  --save_model
```

**3. Forecast next year:**

```bash
python code/predict.py \
  --csv input_data/raw/sample_aadt.csv \
  --model_path output/models/latest_lstm.keras \
  --sequence_length 12
```

---

## Input Data Format

The pipeline expects a CSV with at least two columns: a year column and an AADT column.

| year1 | totadt1 |
|-------|---------|
| 2000  | 12400   |
| 2001  | 13100   |
| 2002  | 13750   |
| ...   | ...     |

A synthetic sample file is provided at `input_data/raw/sample_aadt.csv` to run the pipeline out of the box. Real NBI (National Bridge Inventory) traffic count data is not included due to size.

Column names are configurable in `utils/data_loader.py`.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE    | Mean Absolute Error |
| RMSE   | Root Mean Squared Error |
| MAPE   | Mean Absolute Percentage Error |
| MDAPE  | Median Absolute Percentage Error (robust to outliers) |

---

## Key Parameters

| Argument | Default | Description |
|---|---|---|
| `--sequence_length` | 12 | Number of past years used as input features |
| `--train_split` | 0.7 | Fraction of data used for training |
| `--lstm_units` | = sequence_length | Hidden units per LSTM layer |
| `--dense_units` | 25 | Units in the dense layer before output |
| `--batch_size` | 8 | Training batch size |
| `--max_epochs` | 300 | Maximum training epochs |
| `--patience` | 25 | Early stopping patience (val_loss) |
| `--save_model` | False | Save trained model to `output/models/` |

---

## Data Note

Real AADT data used in the original research comes from the **National Bridge Inventory (NBI)**, maintained by the Federal Highway Administration (FHWA). It is publicly available at [FHWA NBI](https://www.fhwa.dot.gov/bridge/nbi.cfm). The synthetic sample provided here mirrors the format of NBI bridge-level traffic count records.

---

## Requirements

See `requirements.txt`. Built and tested with Python 3.10, TensorFlow 2.13.

---

## License

MIT License. See `LICENSE` for details.
