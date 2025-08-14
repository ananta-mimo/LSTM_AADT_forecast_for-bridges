# LSTM_AADT_forecast_for-bridges
A simple LSTM model of to forecast Average Annual Daily Traffic and Average Weights on a Bridge  
``````lstm-adt-forecast/
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
