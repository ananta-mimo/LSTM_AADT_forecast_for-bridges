"""
evaluate.py
-----------
Standalone evaluation script. Loads a saved LSTM model and a CSV,
runs inference on the test split, and prints a metrics report
with an optional residual plot.

Usage:
    python evaluate.py --csv input_data/raw/sample_aadt.csv \
                       --model_path output/models/latest_lstm.keras

    python evaluate.py --csv input_data/raw/sample_aadt.csv \
                       --model_path output/models/latest_lstm.keras \
                       --plot
"""

import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

from utils.data_loader import load_adt_csv
from utils.metrics import regression_metrics


def to_sequences(data: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(seq_len, data.shape[0]):
        X.append(data[i - seq_len:i, :])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def evaluate(csv_path: str, model_path: str,
             sequence_length: int, train_split: float,
             plot: bool) -> None:

    df     = load_adt_csv(csv_path)
    values = df.values

    scaler        = MinMaxScaler((0, 1))
    values_scaled = scaler.fit_transform(values)

    train_len  = math.ceil(len(values_scaled) * train_split)
    test_data  = values_scaled[train_len - sequence_length:, :]
    X_test, y_test = to_sequences(test_data, sequence_length)

    model         = load_model(model_path)
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred        = scaler.inverse_transform(y_pred_scaled)
    y_true        = scaler.inverse_transform(y_test.reshape(-1, 1))

    metrics = regression_metrics(y_true, y_pred)
    print("\nEvaluation Report")
    print("=" * 35)
    print(f"  Model      : {model_path}")
    print(f"  CSV        : {csv_path}")
    print(f"  Test rows  : {len(y_true)}")
    print(f"  Train split: {train_split:.0%} / {1 - train_split:.0%}")
    print("-" * 35)
    for k, v in metrics.items():
        print(f"  {k:<8}: {v:>10.4f}")
    print()

    if plot:
        test_years = df.index[train_len: train_len + len(y_true)]
        residuals  = y_true.flatten() - y_pred.flatten()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(test_years, y_true,  label="Actual",    linewidth=1.5)
        axes[0].plot(test_years, y_pred,  label="Predicted", linewidth=1.5, linestyle="--")
        axes[0].set_title("Actual vs Predicted AADT")
        axes[0].set_xlabel("Year")
        axes[0].set_ylabel("AADT")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].bar(test_years, residuals, color="steelblue", alpha=0.8)
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_title("Residuals (Actual − Predicted)")
        axes[1].set_xlabel("Year")
        axes[1].set_ylabel("Residual")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("output/figures/evaluation_plot.png", dpi=150)
        print("Plot saved to output/figures/evaluation_plot.png")
        plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a saved LSTM AADT model.")
    ap.add_argument("--csv",             required=True,  type=str)
    ap.add_argument("--model_path",      required=True,  type=str)
    ap.add_argument("--sequence_length", default=12,     type=int)
    ap.add_argument("--train_split",     default=0.7,    type=float)
    ap.add_argument("--plot",            action="store_true",
                    help="Save and display evaluation plots")
    args = ap.parse_args()

    evaluate(args.csv, args.model_path,
             args.sequence_length, args.train_split, args.plot)


if __name__ == "__main__":
    main()
