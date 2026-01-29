# task2_forecasting.py — Task 2: Time Series Forecasting Models for Tesla Stock

"""
Standalone script for Task 2 of Week 9 KAIM Challenge (GMF Investments).
Builds and evaluates ARIMA and LSTM models for TSLA (and optionally BND, SPY),
generates test forecasts, and computes performance metrics.
"""

# ==============================================================
# 0. Imports & Setup
# ==============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# Project directories
PROJECT_ROOT = r"C:\Users\JERUSALEM\Desktop\portfolio-optimization"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures", "task2_forecasting")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ==============================================================
# 1. Load price and return data
# ==============================================================
prices_path = os.path.join(PROCESSED_DIR, "prices_aligned.csv")
returns_path = os.path.join(PROCESSED_DIR, "returns_aligned.csv")

prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)

tickers = prices.columns.tolist()
print("Loaded tickers:", tickers)
print("Date range:", prices.index.min(), "to", prices.index.max())

# ==============================================================
# 2. Train/Test Split
# ==============================================================
train = prices.loc[:'2024-12-31']
test = prices.loc['2025-01-01':]
train_returns = train.pct_change().dropna()
test_returns = test.pct_change().dropna()

print("Train shape:", train.shape, "Test shape:", test.shape)

# ==============================================================
# 3. ADF Test for Stationarity
# ==============================================================
for ticker in tickers:
    result = adfuller(train[ticker])
    p_value = result[1]
    print(f"ADF {ticker}: Statistic={result[0]:.4f}, p-value={p_value:.4f}", end='')
    if p_value < 0.05:
        print(" -> Stationary")
    else:
        print(" -> Non-stationary, differencing needed")

# ==============================================================
# 4. ARIMA Forecasting
# ==============================================================
arima_forecasts = {}
for ticker in tickers:
    print(f"Fitting ARIMA for {ticker}...")
    model = auto_arima(train[ticker], seasonal=False, stepwise=True, suppress_warnings=True, trace=False)
    print(f"Selected ARIMA order for {ticker}: {model.order}")
    forecast = model.predict(n_periods=len(test))
    arima_forecasts[ticker] = forecast

    # Save forecast
    pd.DataFrame(forecast, index=test.index, columns=[f"{ticker}_ARIMA_Forecast"]).to_csv(
        os.path.join(PROCESSED_DIR, f"{ticker.lower()}_arima_forecast.csv")
    )

# ==============================================================
# 5. LSTM Forecasting (TSLA only for speed)
# ==============================================================
ticker = 'TSLA'
scaler = MinMaxScaler(feature_range=(0,1))
train_scaled = scaler.fit_transform(train[ticker].values.reshape(-1,1))

WINDOW_SIZE = 30
X, y = [], []
for i in range(WINDOW_SIZE, len(train_scaled)):
    X.append(train_scaled[i-WINDOW_SIZE:i, 0])
    y.append(train_scaled[i, 0])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

model = Sequential([Input(shape=(X.shape[1],1)), LSTM(32), Dense(1)])
model.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
model.fit(X, y, epochs=8, batch_size=64, verbose=1, callbacks=[early_stop])

# Forecast
last_seq = train_scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE,1)
pred_scaled = []
current_seq = last_seq.copy()
for _ in range(len(test)):
    next_pred = model.predict(current_seq, verbose=0)
    pred_scaled.append(next_pred[0,0])
    current_seq = np.roll(current_seq, -1, axis=1)
    current_seq[0,-1,0] = next_pred[0,0]

predictions = scaler.inverse_transform(np.array(pred_scaled).reshape(-1,1)).flatten()

pd.DataFrame(predictions, index=test.index, columns=["TSLA_LSTM_Forecast"]).to_csv(
    os.path.join(PROCESSED_DIR, "tsla_lstm_forecast.csv")
)
print("✅ LSTM forecast completed for TSLA")

# ==============================================================
# 6. Forecast Evaluation
# ==============================================================
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true[y_true != 0])) * 100

metrics = []
for ticker in tickers:
    actual = test[ticker].values
    for model_name, pred in [('ARIMA', arima_forecasts[ticker]), ('LSTM', predictions if ticker=='TSLA' else arima_forecasts[ticker])]:
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mape_val = mape(actual, pred)
        metrics.append({'Ticker': ticker, 'Model': model_name, 'MAE': round(mae,4), 'RMSE': round(rmse,4), 'MAPE (%)': round(mape_val,2)})

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(os.path.join(PROCESSED_DIR, "forecast_metrics.csv"), index=False)
print("✅ Forecast performance metrics saved")

# ==============================================================
# 7. Combine final forecasts per asset for Task 3
# ==============================================================
tsla_forecast_df = pd.read_csv(os.path.join(PROCESSED_DIR, "tsla_lstm_forecast.csv"), index_col=0, parse_dates=True)
bnd_forecast_df  = pd.read_csv(os.path.join(PROCESSED_DIR, "bnd_arima_forecast.csv"), index_col=0, parse_dates=True)
spy_forecast_df  = pd.read_csv(os.path.join(PROCESSED_DIR, "spy_arima_forecast.csv"), index_col=0, parse_dates=True)

forecast_df = pd.concat([tsla_forecast_df, bnd_forecast_df, spy_forecast_df], axis=1)
forecast_df.columns = ['TSLA','BND','SPY']
forecast_df.to_csv(os.path.join(PROCESSED_DIR, 'task2_final_forecasts.csv'))
print("✅ Task 2 final forecasts ready for Task 3")