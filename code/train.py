import argparse, math, os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from utils.data_loader import load_adt_csv
from utils.plot_utils import plot_series, plot_union, plot_test_vs_pred, plot_loss
from utils.metrics import regression_metrics
import pandas as pd

def to_sequences(data_2d, sequence_length, pred_index=0):
    X, y = [], []
    for i in range(sequence_length, data_2d.shape[0]):
        X.append(data_2d[i-sequence_length:i, :])
        y.append(data_2d[i, pred_index])
    return np.array(X), np.array(y)

def make_union_frames(series_df, train_split, y_true, y_pred):
    n_total = len(series_df)
    train_len = math.ceil(n_total * train_split)
    train = series_df.iloc[:train_len+1].rename(columns={series_df.columns[0]: "x_train"})
    valid = series_df.iloc[train_len:].rename(columns={series_df.columns[0]: "y_test"})
    valid = valid.copy()
    valid.insert(1, "y_pred", y_pred.flatten(), True)
    valid.insert(1, "residuals", (valid["y_pred"].values - y_true.flatten()), True)
    return pd.concat([train, valid])

def build_model(input_shape, units, dense_units):
    m = Sequential()
    m.add(LSTM(units, return_sequences=True, input_shape=input_shape))
    m.add(LSTM(units, return_sequences=True))
    m.add(LSTM(units, return_sequences=False))
    m.add(Dense(dense_units, activation="relu"))
    m.add(Dense(1))
    m.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"])
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=str)
    ap.add_argument("--sequence_length", type=int, default=12)
    ap.add_argument("--train_split", type=float, default=0.7)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--lstm_units", type=int, default=None)
    ap.add_argument("--dense_units", type=int, default=25)
    ap.add_argument("--save_model", action="store_true")
    args = ap.parse_args()

    df = load_adt_csv(args.csv)
    plot_series(df, df.columns[0])

    values = df.values
    scaler = MinMaxScaler((0,1))
    values_scaled = scaler.fit_transform(values)

    train_len = math.ceil(len(values_scaled) * args.train_split)
    train_data = values_scaled[:train_len, :]
    test_data = values_scaled[train_len-args.sequence_length:, :]

    X_train, y_train = to_sequences(train_data, args.sequence_length, 0)
    X_test, y_test = to_sequences(test_data, args.sequence_length, 0)

    units = args.lstm_units or args.sequence_length
    model = build_model((X_train.shape[1], X_train.shape[2]), units, args.dense_units)
    es = EarlyStopping(monitor="val_loss", mode="min", patience=args.patience, verbose=1)
    hist = model.fit(X_train, y_train, batch_size=args.batch_size, epochs=args.max_epochs,
                     validation_split=0.2, shuffle=True, verbose=0, callbacks=[es])

    print("Early-stopped at:", len(hist.history.get("val_loss", [])), "epochs")

    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test.reshape(-1,1))

    mets = regression_metrics(y_true, y_pred)
    print({k: round(v, 3) for k,v in mets.items()})

    union = make_union_frames(df, args.train_split, y_true, y_pred)
    plot_union(union, ylabel="ADT")
    valid = union.iloc[train_len:]
    plot_test_vs_pred(valid, ylabel="ADT")
    plot_loss(hist)

    if args.save_model:
        os.makedirs("output/models", exist_ok=True)
        model_path = "output/models/latest_lstm.keras"
        model.save(model_path)
        print("Saved model to:", model_path)

if __name__ == "__main__":
    main()
