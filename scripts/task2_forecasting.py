"""
Task 2 & Task 3: Time Series Forecasting
ARIMA and LSTM models for TSLA, BND, SPY

- CLI controlled model selection
- Logging-based execution
- Unit-testable functions
- Task 3 future forecasting extension
"""

# =============================
# 1. Imports
# =============================
import os
import argparse
import logging
import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


# =============================
# 2. Configuration
# =============================
TICKERS = ["TSLA", "BND", "SPY"]
WINDOW_SIZE = 30
EPOCHS = 8
FUTURE_HORIZON = 30  # Task 3 forecast days

DATA_PATH = "data/processed/prices.csv"
OUT_DATA = "data/processed"

os.makedirs(OUT_DATA, exist_ok=True)


# =============================
# 3. Logging Setup
# =============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================
# 4. Utility / Testable Functions
# =============================
def adf_test(series):
    stat, pval, *_ = adfuller(series.dropna())
    return stat, pval


def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape


# =============================
# 5. ARIMA Functions
# =============================
def fit_arima(train_series):
    return auto_arima(
        train_series,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
    )


def forecast_arima(model, steps):
    return model.predict(n_periods=steps)


# =============================
# 6. LSTM Functions
# =============================
def prepare_lstm_data(series, window):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window:i, 0])
        y.append(scaled[i, 0])

    X = np.array(X).reshape(-1, window, 1)
    y = np.array(y)
    return X, y, scaler


def build_lstm(input_shape):
    model = Sequential([
        LSTM(50, input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def forecast_lstm(model, last_window, scaler, steps):
    preds = []
    window = last_window.copy()

    for _ in range(steps):
        pred = model.predict(window.reshape(1, -1, 1), verbose=0)[0, 0]
        preds.append(pred)
        window = np.append(window[1:], pred)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds


# =============================
# 7. Task 2 + Task 3 Pipeline
# =============================
def run_pipeline(model_choice):
    prices = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)[TICKERS]

    train = prices.loc[: "2024-12-31"]
    test = prices.loc["2025-01-01":]

    results = []

    for ticker in TICKERS:
        logger.info(f"Processing {ticker}")

        # ---------- Diagnostics ----------
        stat, pval = adf_test(train[ticker])
        logger.info(f"ADF {ticker} | stat={stat:.3f}, p={pval:.3f}")

        # ---------- ARIMA ----------
        if model_choice in ("arima", "both"):
            arima = fit_arima(train[ticker])
            preds = forecast_arima(arima, len(test))
            mae, rmse, mape = evaluate_forecast(test[ticker].values, preds)

            results.append([ticker, "ARIMA", mae, rmse, mape])

            # Task 3 future forecast
            future_preds = forecast_arima(arima, FUTURE_HORIZON)

            pd.DataFrame({
                "date": pd.date_range(test.index[-1], periods=FUTURE_HORIZON + 1, freq="B")[1:],
                "forecast": future_preds
            }).to_csv(f"{OUT_DATA}/{ticker.lower()}_arima_future.csv", index=False)

        # ---------- LSTM ----------
        if model_choice in ("lstm", "both"):
            X_train, y_train, scaler = prepare_lstm_data(train[ticker], WINDOW_SIZE)
            model = build_lstm((WINDOW_SIZE, 1))

            model.fit(
                X_train,
                y_train,
                epochs=EPOCHS,
                batch_size=32,
                callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
                verbose=0
            )

            full_series = pd.concat([train[ticker], test[ticker]])
            X_all, _, _ = prepare_lstm_data(full_series, WINDOW_SIZE)
            preds = model.predict(X_all[-len(test):], verbose=0)
            preds = scaler.inverse_transform(preds).flatten()

            mae, rmse, mape = evaluate_forecast(test[ticker].values, preds)
            results.append([ticker, "LSTM", mae, rmse, mape])

            # Task 3 future forecast
            last_window = scaler.transform(
                full_series.values[-WINDOW_SIZE:].reshape(-1, 1)
            ).flatten()

            future_preds = forecast_lstm(
                model, last_window, scaler, FUTURE_HORIZON
            )

            pd.DataFrame({
                "date": pd.date_range(test.index[-1], periods=FUTURE_HORIZON + 1, freq="B")[1:],
                "forecast": future_preds
            }).to_csv(f"{OUT_DATA}/{ticker.lower()}_lstm_future.csv", index=False)

    # ---------- Save Metrics ----------
    metrics = pd.DataFrame(
        results, columns=["Ticker", "Model", "MAE", "RMSE", "MAPE (%)"]
    )
    metrics["Best_Model"] = metrics.groupby("Ticker")["MAE"].transform(
        lambda x: x == x.min()
    )
    metrics.to_csv(f"{OUT_DATA}/model_performance_comparison.csv", index=False)

    logger.info("Forecasting pipeline completed successfully.")
    return metrics


# =============================
# 8. CLI Entry Point
# =============================
def parse_args():
    parser = argparse.ArgumentParser(description="Task 2 & 3 Forecasting Pipeline")
    parser.add_argument(
        "--model",
        choices=["arima", "lstm", "both"],
        default="both",
        help="Select forecasting model"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.model)
