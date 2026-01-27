# src/task4_optimization.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from src.portfolio_analysis import compute_forecast_metrics, portfolio_forecast

processed_folder = "../data/processed"
figures_folder = "../reports/figures"
os.makedirs(figures_folder, exist_ok=True)

tickers = ["tsla", "bnd", "spy"]
forecast_steps = 252  # ~1 year

# Load ARIMA future forecasts
forecasts = {}
for ticker in tickers:
    forecasts[ticker] = pd.read_csv(
        os.path.join(processed_folder, f"{ticker}_arima_future.csv"), index_col=0, parse_dates=True
    )

# Compute expected returns & daily returns
expected_returns = {}
returns_matrix = pd.DataFrame(index=forecasts[tickers[0]].index)

for ticker in tickers:
    series = forecasts[ticker]["Forecast"]
    exp_return, _ = compute_forecast_metrics(series)
    expected_returns[ticker] = exp_return
    returns_matrix[ticker] = series.pct_change().fillna(0)

# Covariance matrix
cov_matrix = returns_matrix.cov() * 252  # annualized
