# task4_portfolio.py — Task 4: Portfolio Optimization (Standalone)

"""
Standalone script for Task 4 of Week 9 KAIM Challenge (GMF Investments).
Performs portfolio optimization using forecasted TSLA returns combined with historical BND and SPY returns.
Generates Efficient Frontier, Max Sharpe, and Min Volatility portfolios.
"""

# ==============================================================
# 0. Imports & Setup
# ==============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import expected_returns, risk_models, EfficientFrontier

# Project directories
PROJECT_ROOT = r"C:\Users\JERUSALEM\Desktop\portfolio-optimization"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures", "task4_portfolio")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ==============================================================
# 1. Load historical price data
# ==============================================================
prices = pd.read_csv(os.path.join(PROCESSED_DIR, "prices_aligned.csv"), index_col=0, parse_dates=True)
prices_hist = prices.loc['2015-01-01':'2024-12-31']

# ==============================================================
# 2. Compute historical expected returns and covariance
# ==============================================================
mu_hist = expected_returns.mean_historical_return(prices_hist)
cov_hist = risk_models.sample_cov(prices_hist)

# ==============================================================
# 3. Load TSLA LSTM forecast and compute forecast-based return
# ==============================================================
tsla_forecast = pd.read_csv(os.path.join(PROCESSED_DIR, 'tsla_lstm_forecast.csv'), index_col=0, parse_dates=True)
tsla_last_price = prices_hist['TSLA'].iloc[-1]
tsla_forecast_price = tsla_forecast.iloc[-1,0]
forecast_days = len(tsla_forecast)
tsla_expected_return = (tsla_forecast_price/tsla_last_price)**(252/forecast_days) - 1

# Update expected returns vector
mu = mu_hist.copy()
mu['TSLA'] = tsla_expected_return

# ==============================================================
# 4. Maximum Sharpe Ratio Portfolio
# ==============================================================
ef_sharpe = EfficientFrontier(mu, cov_hist)
ef_sharpe.max_sharpe(risk_free_rate=0.0)
weights_max_sharpe = ef_sharpe.clean_weights()
perf_max_sharpe = ef_sharpe.portfolio_performance(risk_free_rate=0.0)

# ==============================================================
# 5. Minimum Volatility Portfolio
# ==============================================================
ef_minvol = EfficientFrontier(mu, cov_hist)
ef_minvol.min_volatility()
weights_min_vol = ef_minvol.clean_weights()
perf_min_vol = ef_minvol.portfolio_performance(risk_free_rate=0.0)

# ==============================================================
# 6. Efficient Frontier simulation
# ==============================================================
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
plt.title('Efficient Frontier — Task 4 Portfolio Optimization')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR,'efficient_frontier.png'), dpi=300)
plt.close()

# ==============================================================
# 7. Save weights summary
# ==============================================================
weights_summary = pd.DataFrame({
    'Asset': ['TSLA','BND','SPY'],
    'Max_Sharpe': [weights_max_sharpe[a] for a in ['TSLA','BND','SPY']],
    'Min_Vol': [weights_min_vol[a] for a in ['TSLA','BND','SPY']]
})
weights_summary.to_csv(os.path.join(FIGURES_DIR,'portfolio_weights.csv'), index=False)

print('✔ Task 4 — Portfolio Optimization Complete')