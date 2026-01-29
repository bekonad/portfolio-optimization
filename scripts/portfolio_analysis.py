# portfolio_analysis.py — Full Pipeline (Task 1 → Task 4)

"""
This script consolidates preprocessing, EDA, forecasting, and portfolio optimization
for the Week 9 KAIM challenge (GMF Investments).
Generates all figures and outputs to reports/figures/task*_ directories.
"""

# ==============================================================
# 0. Imports & Setup
# ==============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import expected_returns, risk_models, EfficientFrontier
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Project directories
PROJECT_ROOT = "C:\Users\JERUSALEM\Desktop\portfolio-optimization"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ==============================================================
# 1. Task 1 — Data Preprocessing & EDA
# ==============================================================
prices = pd.read_csv(os.path.join(PROCESSED_DIR, "prices_aligned.csv"), index_col=0, parse_dates=True)
returns = prices.pct_change().dropna()

# Example EDA: plot prices
plt.figure(figsize=(10,5))
for col in prices.columns:
    plt.plot(prices.index, prices[col], label=col)
plt.title("Adjusted Closing Prices (2015-2024)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "task1_eda", "prices_over_time.png"))
plt.close()

# ==============================================================
# 2. Task 2 — ARIMA & LSTM Forecasting (simplified)
# ==============================================================
# For demonstration: ARIMA on TSLA
tsla_train = prices['TSLA'].loc[:'2024-12-31']
arima_model = auto_arima(tsla_train, seasonal=False, suppress_warnings=True)
tsla_forecast_arima = arima_model.predict(n_periods=252)

# LSTM forecast prep (simplified)
scaler = MinMaxScaler()
tsla_scaled = scaler.fit_transform(tsla_train.values.reshape(-1,1))
X, y = [], []
window_size = 60
for i in range(window_size, len(tsla_scaled)):
    X.append(tsla_scaled[i-window_size:i, 0])
    y.append(tsla_scaled[i, 0])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

lstm_model = Sequential([LSTM(50, input_shape=(X.shape[1],1)), Dense(1)])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X, y, epochs=5, batch_size=32, verbose=0)
tsla_lstm_pred_scaled = lstm_model.predict(X[-252:])
tsla_lstm_forecast = scaler.inverse_transform(tsla_lstm_pred_scaled)

# Save forecast
pd.DataFrame(tsla_lstm_forecast, index=pd.date_range('2025-01-01', periods=252, freq='B'), columns=['Forecast']).to_csv(os.path.join(PROCESSED_DIR, 'tsla_lstm_forecast.csv'))

# ==============================================================
# 3. Task 3 — Forecast Evaluation (optional summary)
# ==============================================================
# Here you could compute RMSE/MAE/MAPE vs actuals if available

# ==============================================================
# 4. Task 4 — Portfolio Optimization
# ==============================================================
# Use historical returns 2015-2024
prices_hist = prices.loc['2015-01-01':'2024-12-31']
mu_hist = expected_returns.mean_historical_return(prices_hist)
cov_hist = risk_models.sample_cov(prices_hist)

# Load forecast for TSLA
tsla_forecast = pd.read_csv(os.path.join(PROCESSED_DIR, 'tsla_lstm_forecast.csv'), index_col=0, parse_dates=True)
tsla_last_price = prices_hist['TSLA'].iloc[-1]
tsla_forecast_price = tsla_forecast.iloc[-1,0]
forecast_days = len(tsla_forecast)
tsla_expected_return = (tsla_forecast_price/tsla_last_price)**(252/forecast_days) - 1

mu = mu_hist.copy()
mu['TSLA'] = tsla_expected_return

# Maximum Sharpe Portfolio
ef_sharpe = EfficientFrontier(mu, cov_hist)
ef_sharpe.max_sharpe(risk_free_rate=0.0)
weights_max_sharpe = ef_sharpe.clean_weights()
perf_max_sharpe = ef_sharpe.portfolio_performance(risk_free_rate=0.0)

# Minimum Volatility Portfolio
ef_minvol = EfficientFrontier(mu, cov_hist)
ef_minvol.min_volatility()
weights_min_vol = ef_minvol.clean_weights()
perf_min_vol = ef_minvol.portfolio_performance(risk_free_rate=0.0)

# Save Efficient Frontier plot
returns, vols = [], []
target_returns = np.linspace(mu.min(), 0.08, 100)
for r in target_returns:
    ef = EfficientFrontier(mu, cov_hist)
    try:
        ef.efficient_return(r)
        perf = ef.portfolio_performance()
        returns.append(perf[0])
        vols.append(perf[1])
    except ValueError:
        continue

plt.figure(figsize=(10,6))
plt.scatter(vols, returns, c=np.array(returns)/np.array(vols), cmap='viridis', alpha=0.7)
plt.scatter(perf_max_sharpe[1], perf_max_sharpe[0], marker='*', color='red', s=200, label='Max Sharpe')
plt.scatter(perf_min_vol[1], perf_min_vol[0], marker='*', color='blue', s=150, label='Min Volatility')
plt.xlabel('Annualized Volatility')
plt.ylabel('Expected Annual Return')
plt.title('Efficient Frontier — Full Portfolio Analysis')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,'task4_portfolio','efficient_frontier.png'), dpi=300)
plt.close()

# Save weights summary
weights_summary = pd.DataFrame({
    'Asset': ['TSLA','BND','SPY'],
    'Max_Sharpe': [weights_max_sharpe[a] for a in ['TSLA','BND','SPY']],
    'Min_Vol': [weights_min_vol[a] for a in ['TSLA','BND','SPY']]
})
weights_summary.to_csv(os.path.join(FIGURES_DIR,'task4_portfolio','portfolio_weights.csv'), index=False)

print('✔ Full portfolio analysis complete: Task 1 → Task 4')
