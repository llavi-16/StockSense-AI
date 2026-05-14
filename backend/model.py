import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

#from .utils import get_close_prices, scale_series, create_sequences, inverse_scale
from utils import get_close_prices, scale_series, create_sequences, inverse_scale
MODELS_DIR = Path(os.getenv("MODELS_DIR", Path(__file__).resolve().parent.parent / "models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _parse_period_to_days(period: str) -> int:
    
    s = period.strip().lower()
    try:
        if s.endswith("y"):
            return int(float(s[:-1]) * 365)
        if s.endswith("mo"):
            return int(float(s[:-2]) * 30)
        if s.endswith("d"):
            return int(float(s[:-1]))
    except ValueError:
        pass
    
    return 730


def _fetch_stock_history_stooq(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    
    sym = ticker.upper().strip()
    if "." not in sym:
        sym = f"{sym}.US"


    stooq_url = f"https://stooq.com/q/d/l/?s={sym.lower()}&i=d"

    df = pd.read_csv(stooq_url)
    if df.empty or "Close" not in df.columns:
        raise ValueError(f"Stooq returned no data for '{ticker}'.")
    
    df = df.sort_values("Date").reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"], utc=False)

    max_days = _parse_period_to_days(period)
    if len(df) > max_days:
        df = df.iloc[-max_days:].reset_index(drop=True)


    out = pd.DataFrame(
        {
            "Open": df.get("Open"),
            "High": df.get("High"),
            "Low": df.get("Low"),
            "Close": df.get("Close"),
            "Adj Close": df.get("Close"),
            "Volume": df.get("Volume"),
        }
    )
    out.index = pd.to_datetime(df["Date"]).values
    out.index.name = "Date"
    return out


def _fetch_stock_history_synthetic(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Last-resort fallback: generate a synthetic price series.

    This keeps the demo functional in environments where outbound network access to
    finance data providers is blocked.
    """
    days = _parse_period_to_days(period)
    days = max(days, 120)  
    


    seed = abs(hash(ticker.upper())) % (2**32)
    rng = np.random.default_rng(seed)


    start_price = 100.0 + (seed % 500) / 10.0
    mu = 0.0002
    sigma = 0.01

    returns = rng.normal(loc=mu, scale=sigma, size=days)
    prices = start_price * np.exp(np.cumsum(returns))


    close = prices
    open_ = close * (1 + rng.normal(0, 0.002, size=days))
    high = np.maximum(open_, close) * (1 + rng.random(size=days) * 0.003)
    low = np.minimum(open_, close) * (1 - rng.random(size=days) * 0.003)
    volume = rng.integers(low=1_000_000, high=20_000_000, size=days).astype(float)

    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")

    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": close,
            "Volume": volume,
        },
        index=dates,
    )
    df.index.name = "Date"
    df.attrs["source"] = "synthetic"
    df.attrs["warning"] = (
        "Yahoo Finance and Stooq were unreachable from this environment. "
        "Using synthetic data for demo purposes."
    )
    return df

def fetch_stock_history(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch historical data using Stooq (reliable).
    """
    df = _fetch_stock_history_stooq(ticker=ticker, period=period, interval=interval)
    
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")

    df.attrs["source"] = "stooq"
    return df
# def fetch_stock_history(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
#     """
#     Fetch historical data for a ticker from Yahoo Finance.
#     """
    
#     df = yf.download(
#         ticker,
#         period=period,
#         interval=interval,
#         progress=False,
#         threads=False,
#     )
#     if df.empty:
        
#         yahoo_error = (
#             f"Yahoo Finance returned no data for '{ticker}' "
#             f"(period={period}, interval={interval}). "
#             f"This usually means your network blocked Yahoo Finance requests "
#             f"(proxy/firewall/VPN) or Yahoo temporarily rate-limited you."
#         )

#         try:
            
#             stooq_df = _fetch_stock_history_stooq(ticker=ticker, period=period, interval=interval)
#             if stooq_df.empty:
#                 raise ValueError("Stooq returned an empty dataframe.")
#             stooq_df.attrs["source"] = "stooq"
#             return stooq_df
#         except Exception:
            
#             return _fetch_stock_history_synthetic(ticker=ticker, period=period, interval=interval)

#     df.attrs["source"] = "yfinance"

#     return df


def build_lstm_model(input_shape: Tuple[int, int]) -> Sequential:
  
  
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    return model


def train_lstm_for_ticker(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    window_size: int = 60,
    epochs: int = 10,
    batch_size: int = 32,
) -> dict:
    
    raw_df = fetch_stock_history(ticker, period=period, interval=interval)
    data_source = raw_df.attrs.get("source", "unknown")
    warning = raw_df.attrs.get("warning")
    close_df = get_close_prices(raw_df)

    scaled, scaler = scale_series(close_df["Close"])
    x_train, y_train = create_sequences(scaled, window_size=window_size)

    model = build_lstm_model((x_train.shape[1], 1))
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )


    train_pred = model.predict(x_train, verbose=0)
    train_pred_inv = inverse_scale(scaler, train_pred.flatten())
    y_train_inv = inverse_scale(scaler, y_train)
    rmse = float(np.sqrt(np.mean((train_pred_inv - y_train_inv) ** 2)))


    model_path = MODELS_DIR / f"{ticker.upper()}_lstm.h5"
    scaler_path = MODELS_DIR / f"{ticker.upper()}_scaler.npy"
    model.save(model_path)
    
    np.save(
        scaler_path,
        {
            "min_": scaler.min_,
            "scale_": scaler.scale_,
            "data_min_": scaler.data_min_,
            "data_max_": scaler.data_max_,
            "data_range_": scaler.data_range_,
            "feature_range": scaler.feature_range,
        },
        allow_pickle=True,
    )

    return {
        "ticker": ticker.upper(),
        "rmse": rmse,
        "data_source": data_source,
        "warning": warning,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "history_loss": [float(v) for v in history.history.get("loss", [])],
    }


def _load_scaler(scaler_path: Path):
    """
    Reconstruct MinMaxScaler from stored attributes.
    """
    from sklearn.preprocessing import MinMaxScaler

    data = np.load(scaler_path, allow_pickle=True).item()
    scaler = MinMaxScaler(feature_range=tuple(data["feature_range"]))
    scaler.min_ = data["min_"]
    scaler.scale_ = data["scale_"]
    scaler.data_min_ = data["data_min_"]
    scaler.data_max_ = data["data_max_"]
    scaler.data_range_ = data["data_range_"]
    return scaler


def predict_next_days(
    ticker: str,
    days_ahead: int = 7,
    window_size: int = 60,
    period: str = "2y",
    interval: str = "1d",
) -> dict:
    """
    Load saved model + scaler and forecast the next N days.
    Returns recent actual prices and predicted future prices.
    """
    ticker_upper = ticker.upper()
    model_path = MODELS_DIR / f"{ticker_upper}_lstm.h5"
    scaler_path = MODELS_DIR / f"{ticker_upper}_scaler.npy"

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            f"Model/scaler for {ticker_upper} not found. Train the model first."
        )

    model = load_model(model_path)
    scaler = _load_scaler(scaler_path)

    raw_df = fetch_stock_history(ticker_upper, period=period, interval=interval)
    data_source = raw_df.attrs.get("source", "unknown")
    warning = raw_df.attrs.get("warning")
    close_df = get_close_prices(raw_df)
    close_values = close_df["Close"].values

    scaled, _ = scale_series(close_df["Close"])

    #window size
    last_window = scaled[-window_size:].reshape(1, window_size, 1)
    future_scaled: List[float] = []

    for _ in range(days_ahead):
        next_scaled = model.predict(last_window, verbose=0)[0, 0]
        future_scaled.append(float(next_scaled))
        last_window = np.append(last_window[:, 1:, :], [[[next_scaled]]], axis=1)

    future_prices = inverse_scale(scaler, np.array(future_scaled))

    trend = "UP" if future_prices[-1] >= close_values[-1] else "DOWN"

    return {
        "ticker": ticker_upper,
        "actual": close_values.tolist(),
        "predicted": future_prices.tolist(),
        "last_actual": float(close_values[-1]),
        "last_predicted": float(future_prices[0]),
        "trend": trend,
        "data_source": data_source,
        "warning": warning,
    }

