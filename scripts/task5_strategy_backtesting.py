# =====================================================
# Task 5 – Strategy Backtesting (Script Version)
# =====================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# -------------------------
# 1. Paths & Directories
# -------------------------
PROJECT_ROOT = ".."
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures", "task5_backtesting")
os.makedirs(FIGURES_DIR, exist_ok=True)
print(f"✔ Figures will be saved to: {FIGURES_DIR}")

# -------------------------
# 2. Load Prices
# -------------------------
prices_csv_path = os.path.join(PROCESSED_DIR, "prices_aligned.csv")
prices = pd.read_csv(prices_csv_path, index_col=0, parse_dates=True)
prices = prices.ffill().dropna()
print(f"✔ Loaded prices from {prices_csv_path}")

# -------------------------
# 3. Backtesting Period
# -------------------------
BACKTEST_START = "2025-01-01"
BACKTEST_END   = "2026-01-15"

backtest_prices = prices.loc[BACKTEST_START:BACKTEST_END]
daily_returns = backtest_prices.pct_change().dropna()
print("✔ Daily returns for backtesting period loaded")

# -------------------------
# 4. Strategy & Benchmark Weights (Task 4 Max Sharpe)
# -------------------------
OPTIMAL_WEIGHTS = {
    "TSLA": 0.0,   # TSLA excluded due to negative forecast
    "SPY": 0.7273, # Replace with your Task 4 actual weight
    "BND": 0.2727
}

BENCHMARK_WEIGHTS = {
    "TSLA": 0.0,
    "SPY": 0.60,
    "BND": 0.40
}

print("✔ Strategy weights:", OPTIMAL_WEIGHTS)
print("✔ Benchmark weights:", BENCHMARK_WEIGHTS)

# -------------------------
# 5. Portfolio Returns Function
# -------------------------
def portfolio_returns(returns_df, weights_dict):
    weights = pd.Series(weights_dict)
    return (returns_df * weights).sum(axis=1)

# Static strategy
strategy_static_ret = portfolio_returns(daily_returns, OPTIMAL_WEIGHTS)
benchmark_ret       = portfolio_returns(daily_returns, BENCHMARK_WEIGHTS)

# -------------------------
# 6. Monthly Rebalanced Strategy
# -------------------------
strategy_rebal_ret = pd.Series(index=daily_returns.index, dtype=float)
weights = pd.Series(OPTIMAL_WEIGHTS)
last_month = daily_returns.index[0].month

for date in daily_returns.index:
    if date.month != last_month:
        last_month = date.month
        weights = pd.Series(OPTIMAL_WEIGHTS)  # Reset weights monthly
    strategy_rebal_ret.loc[date] = (daily_returns.loc[date] * weights).sum()

# -------------------------
# 7. Performance Metrics Function
# -------------------------
def calc_metrics(returns_series, rf_rate=0.0):
    total_ret = (1 + returns_series).prod() - 1
    n_days = len(returns_series)
    ann_ret = (1 + total_ret)**(252/n_days) - 1
    ann_vol = returns_series.std() * np.sqrt(252)
    sharpe  = (ann_ret - rf_rate)/ann_vol if ann_vol != 0 else np.nan
    cum_ret = (1 + returns_series).cumprod()
    max_dd  = (cum_ret / cum_ret.cummax() - 1).min()
    return {
        "Total Return": total_ret,
        "Annualized Return": ann_ret,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd
    }

metrics_static    = calc_metrics(strategy_static_ret)
metrics_rebalance = calc_metrics(strategy_rebal_ret)
metrics_benchmark = calc_metrics(benchmark_ret)

metrics_df = pd.DataFrame(
    [metrics_static, metrics_rebalance, metrics_benchmark],
    index=["Static Strategy", "Monthly Rebalance", "Benchmark 60/40"]
)

print("\n✔ Performance metrics comparison (2025–2026):")
print(metrics_df.round(4))

# -------------------------
# 8. Plot Cumulative Returns
# -------------------------
cum_static    = (1 + strategy_static_ret).cumprod()
cum_rebalance = (1 + strategy_rebal_ret).cumprod()
cum_benchmark = (1 + benchmark_ret).cumprod()

plt.figure(figsize=(13,7))
plt.plot(cum_static, label="Static Strategy (Task 4 Weights)", linewidth=2.5)
plt.plot(cum_rebalance, label="Monthly Rebalanced Strategy", linestyle="-.", linewidth=2.5)
plt.plot(cum_benchmark, label="Benchmark 60% SPY / 40% BND", linestyle="--", linewidth=2.5)

plt.title("Backtest: Cumulative Returns (2025–2026)", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Growth of $1", fontsize=12)
plt.legend(loc="upper left", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()

fig_path = os.path.join(FIGURES_DIR, "cumulative_returns_rebalance.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"✔ Figure saved to: {fig_path}")

# -------------------------
# 9. Optional: Save Metrics to CSV
# -------------------------
metrics_csv_path = os.path.join(FIGURES_DIR, "task5_performance_metrics.csv")
metrics_df.to_csv(metrics_csv_path)
print(f"✔ Performance metrics saved to CSV: {metrics_csv_path}")
