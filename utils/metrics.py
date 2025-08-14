import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100.0
    mdape = np.median(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100.0
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"MAE": mae, "MAPE": mape, "MDAPE": mdape, "RMSE": rmse}
