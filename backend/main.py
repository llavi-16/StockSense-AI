from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.model import fetch_stock_history, train_lstm_for_ticker, predict_next_days
app = FastAPI(title="Stock Market Trend Prediction API", version="1.0.0")



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/fetch-data")
def fetch_data_endpoint(ticker: str, period: str = "6mo", interval: str = "1d"):

    try:
        df = fetch_stock_history(ticker, period=period, interval=interval)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


    df = df.reset_index()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return {
        "ticker": ticker.upper(),
        "data": df.to_dict(orient="records"),
    }


@app.post("/train-model")
def train_model_endpoint(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    epochs: int = 10,
):
    try:
        summary = train_lstm_for_ticker(
            ticker=ticker,
            period=period,
            interval=interval,
            epochs=epochs,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return summary


@app.get("/predict")
def predict_endpoint(
    ticker: str,
    days_ahead: int = 7,
    window_size: int = 60,
    period: str = "2y",
    interval: str = "1d",
):
    
    try:
        result = predict_next_days(
            ticker=ticker,
            days_ahead=days_ahead,
            window_size=window_size,
            period=period,
            interval=interval,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



    return {
        "ticker": result["ticker"],
        "actual": result["actual"],
        "predicted": result["predicted"],
        "last_actual": result["last_actual"],
        "last_predicted": result["last_predicted"],
        "trend": result["trend"],
        "data_source": result.get("data_source"),
        "warning": result.get("warning"),
    }

