# Task 3 — Forecast Future Market Trends (Python)

"""
This script generates 6–12 month forecasts for TSLA, BND, and SPY using the best-performing models from Task 2.
- TSLA → LSTM
- BND → ARIMA
- SPY → ARIMA
It also constructs confidence intervals and prepares the data for Task 4 portfolio optimization.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Project paths
project_root = '..'
processed_folder = os.path.join(project_root, 'data', 'processed')
figures_folder = os.path.join(project_root, 'reports', 'figures', 'task3_forecast')
os.makedirs(figures_folder, exist_ok=True)

# Load historical prices
prices = pd.read_csv(os.path.join(processed_folder, 'prices_aligned.csv'), index_col=0, parse_dates=True)

# Load Task 2 final forecasts
forecast_df = pd.read_csv(os.path.join(processed_folder, 'task2_final_forecasts.csv'), index_col=0, parse_dates=True)
forecast_df.columns = ['TSLA', 'BND', 'SPY']

# Construct confidence intervals
confidence_intervals = {}
for asset in forecast_df.columns:
    forecast = forecast_df[asset]
    if asset == 'TSLA':
        vol = prices[asset].pct_change().rolling(30).std().mean()
        ci_upper = forecast * (1 + 2 * vol)
        ci_lower = forecast * (1 - 2 * vol)
    else:
        vol = prices[asset].pct_change().std()
        ci_upper = forecast * (1 + 1.96 * vol)
        ci_lower = forecast * (1 - 1.96 * vol)
    confidence_intervals[asset] = (ci_lower, ci_upper)

# Plot forecasts with confidence intervals
for asset in forecast_df.columns:
    plt.figure(figsize=(10,5))
    plt.plot(prices[asset], label='Historical')
    plt.plot(forecast_df[asset], label='Forecast', color='orange' if asset=='TSLA' else ('green' if asset=='BND' else 'purple'))
    lower, upper = confidence_intervals[asset]
    plt.fill_between(forecast_df.index, lower, upper, color='orange' if asset=='TSLA' else ('green' if asset=='BND' else 'purple'), alpha=0.25)
    plt.title(f'{asset} Price Forecast')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_folder, f'{asset.lower()}_forecast.png'), dpi=300)
    plt.show()

# Generate summary of expected changes
df_summary = pd.DataFrame([{
    'Asset': asset,
    'Start Forecast Price': round(forecast_df[asset].iloc[0],2),
    'End Forecast Price': round(forecast_df[asset].iloc[-1],2),
    'Expected Change (%)': round((forecast_df[asset].iloc[-1]-forecast_df[asset].iloc[0])/forecast_df[asset].iloc[0]*100,2)
} for asset in forecast_df.columns])
df_summary.to_csv(os.path.join(processed_folder, 'task3_forecast_summary.csv'), index=False)

# Prepare Task 4 input
portfolio_input = forecast_df.copy()
portfolio_input.to_csv(os.path.join(processed_folder, 'task4_portfolio_input.csv'))

print('✅ Task 3 forecasts and summary prepared for Task 4 portfolio optimization')
portfolio_input.head()
