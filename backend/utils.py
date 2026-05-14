import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_close_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and clean the 'Close' price column.
    """
    data = df[["Close"]].dropna().copy()
    return data


def scale_series(series: pd.Series):
    """
    Scale a 1D price series to [0, 1] using MinMaxScaler.
    Returns scaled numpy array and fitted scaler.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    return scaled, scaler


def create_sequences(dataset: np.ndarray, window_size: int = 60):
    """
    Turn a 1D scaled series into overlapping sequences for LSTM.
    Each sample is a window of length `window_size` and target is the next value.
    """
    x, y = [], []
    for i in range(window_size, len(dataset)):
        x.append(dataset[i - window_size : i, 0])
        y.append(dataset[i, 0])
    x, y = np.array(x), np.array(y)
    # reshape to (samples, timesteps, features)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x, y


def inverse_scale(scaler: MinMaxScaler, data: np.ndarray) -> np.ndarray:
    """
    Invert MinMax scaling for a 1D array.
    """
    return scaler.inverse_transform(data.reshape(-1, 1)).flatten()

