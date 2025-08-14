import argparse, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from utils.data_loader import load_adt_csv

def to_sequences(data_2d, sequence_length, pred_index=0):
    X = []
    for i in range(sequence_length, data_2d.shape[0]+1):
        X.append(data_2d[i-sequence_length:i, :])
    return np.array(X)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=str)
    ap.add_argument("--model_path", required=True, type=str)
    ap.add_argument("--sequence_length", type=int, default=12)
    args = ap.parse_args()

    df = load_adt_csv(args.csv)
    values = df.values
    scaler = MinMaxScaler((0,1))
    values_scaled = scaler.fit_transform(values)

    X = to_sequences(values_scaled, args.sequence_length, 0)[-1:]
    model = load_model(args.model_path)
    y_pred_scaled = model.predict(X)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()

    next_year = int(df.index.max()) + 1
    print(f"Forecast for {next_year}: {y_pred[0]:.2f}")

if __name__ == "__main__":
    main()
