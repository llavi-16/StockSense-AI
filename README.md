# Stock Market Trend Prediction Web App (LSTM + FastAPI)

This project is a minimal end-to-end demo that predicts future closing prices for a given stock ticker using an LSTM (Long Short-Term Memory) neural network trained on historical data from Yahoo Finance.  
The backend is built with FastAPI and TensorFlow/Keras, and the frontend is a small HTML/CSS/JS app that uses Chart.js for visualization.



## Project structure

- `backend/`
  - `main.py` &mdash; FastAPI app and HTTP endpoints.
  - `model.py` &mdash; LSTM model definition, training, and prediction logic.
  - `utils.py` &mdash; data preprocessing utilities (scaling, sequence generation).
- `frontend/`
  - `index.html` &mdash; UI layout.
  - `style.css` &mdash; styling.
  - `script.js` &mdash; calls the backend and renders Chart.js line charts.
- `models/` &mdash; saved LSTM models and scalers (created at runtime).
- `requirements.txt` &mdash; Python dependencies.

## Backend endpoints

- `GET /health` &mdash; simple health check.
- `GET /fetch-data?ticker=TSLA&period=6mo`  
  Fetch raw OHLCV data for the ticker from Yahoo Finance.
- `POST /train-model?ticker=TSLA&epochs=10`  
  Trains an LSTM model on the ticker's closing price, saves the model (`.h5`) and scaler, and returns a simple training summary including RMSE.
- `GET /predict?ticker=TSLA&days_ahead=7`  
  Loads the saved model and scaler, predicts the next N days of closing prices, and returns:
  - `actual`: full history of actual close prices used for context.
  - `predicted`: an array of length `days_ahead` with forecasted prices.
  - `trend`: `"UP"` or `"DOWN"` based on last actual vs future prices.
  - `last_actual`, `last_predicted`: scalar values for quick display.

## How to run (local development)

### 1. Create and activate a virtual environment (recommended)

```bash
cd c:\upd_lstm
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the FastAPI backend

From the project root:

```bash
uvicorn backend.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

You can quickly check that it is running by opening in a browser:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs` (interactive Swagger UI)

### 4. Serve the frontend

Use any simple static file server from the `frontend/` folder. For example:

```bash
cd frontend
python -m http.server 5500
```

Then open `http://127.0.0.1:5500` in your browser.

> The frontend assumes the backend is running at `http://127.0.0.1:8000`.  
> Adjust `API_BASE_URL` in `frontend/script.js` if you deploy elsewhere.

## Usage

1. Open the frontend in your browser.
2. Enter a ticker symbol (e.g. `AAPL`, `TSLA`).
3. Optionally adjust:
   - **Days Ahead** (how many future days to forecast).
   - **Training Epochs** (more epochs usually mean longer training but potentially better fit).
4. Click **Predict**.
5. The app will:
   - Fetch historical data from Yahoo Finance on the backend.
   - Train a 2-layer LSTM model on the closing prices.
   - Save the model and scaler to the `models/` directory.
   - Predict the next N days of prices.
   - Return both **actual** and **predicted** series as JSON.
6. The frontend plots a line chart (Chart.js):
   - Solid line: actual recent close prices.
   - Dashed line: future predicted prices.
7. The "Current Trend" card shows whether the trend is predicted to go **UP** or **DOWN**.

## Notes and extensions

- The example uses a univariate LSTM (only closing price as input).
- For more serious use, consider:
  - Training once and caching models instead of retraining on every request.
  - Adding more features (volume, technical indicators, etc.).
  - Adding RMSE or other metrics on a held-out validation set.
  - Implementing multi-stock comparison and candlestick charts in the frontend.

