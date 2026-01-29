"""
Task 5: Strategy Backtesting (Full Script)
------------------------------------------
Features:
- Static portfolio using Task 4 optimal weights
- Optional monthly rebalancing
- Transaction costs
- Rolling window analysis
- Monte Carlo simulations
- Plots cumulative returns
- Saves performance metrics CSV
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

# ---------------------------
# 1. Paths & Config
# ---------------------------
PROJECT_ROOT = ".."
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures", "task5_backtesting")
os.makedirs(FIGURES_DIR, exist_ok=True)
print(f"✔ Figures and CSV will be saved to: {FIGURES_DIR}")

# ---------------------------
# 2. Load Prices (aligned CSV)
# ---------------------------
prices = pd.read_csv(os.path.join(PROCESSED_DIR, "prices_aligned.csv"), 
                     index_col=0, parse_dates=True)

# Backtest period: Jan 2025 - Jan 2026
prices_bt = prices.loc["2025-01-01":"2026-01-15"]
print(f"✔ Backtest price data loaded. Shape: {prices_bt.shape}")

# ---------------------------
# 3. Daily Returns
# ---------------------------
daily_returns = prices_bt.pct_change().dropna()
print("\nSample daily returns:\n", daily_returns.head(5))

# ---------------------------
# 4. Portfolio Weights
# ---------------------------
# Replace with your Task 4 Max Sharpe weights
OPTIMAL_WEIGHTS = {"TSLA": 0.0, "SPY": 0.60, "BND": 0.40}

# Benchmark 60/40 SPY/BND
BENCHMARK_WEIGHTS = {"TSLA": 0.0, "SPY": 0.60, "BND": 0.40}

transaction_cost = 0.001  # 0.1% per trade

# ---------------------------
# 5. Portfolio Return Functions
# ---------------------------
def portfolio_returns(returns_df, weights_dict):
    """Compute daily portfolio returns (static)."""
    weights = pd.Series(weights_dict)
    return (returns_df * weights).sum(axis=1)

def portfolio_returns_rebalance(returns_df, weights_dict, rebalance_freq='M', transaction_cost=0.0):
    """Compute daily portfolio returns with periodic rebalancing and transaction costs."""
    weights_prev = pd.Series(weights_dict)
    port_ret = []

    # Determine rebalance dates
    rebalance_dates = returns_df.resample(rebalance_freq).first().index

    for date, row in returns_df.iterrows():
        if date in rebalance_dates:
            weights_new = pd.Series(weights_dict)
            # Compute turnover
            turnover = (weights_new - weights_prev).abs().sum()
            weights_prev = weights_new.copy()
        else:
            turnover = 0.0
        daily_ret = (row * weights_prev).sum() - turnover * transaction_cost
        port_ret.append(daily_ret)
    return pd.Series(port_ret, index=returns_df.index)

# ---------------------------
# 6. Compute Returns
# ---------------------------
strategy_ret_static   = portfolio_returns(daily_returns, OPTIMAL_WEIGHTS)
strategy_ret_rebal    = portfolio_returns_rebalance(daily_returns, OPTIMAL_WEIGHTS, rebalance_freq='M', transaction_cost=transaction_cost)
benchmark_ret         = portfolio_returns(daily_returns, BENCHMARK_WEIGHTS)

# ---------------------------
# 7. Cumulative Returns Plot
# ---------------------------
cum_static    = (1 + strategy_ret_static).cumprod()
cum_rebalance = (1 + strategy_ret_rebal).cumprod()
cum_benchmark = (1 + benchmark_ret).cumprod()

plt.figure(figsize=(12,6))
plt.plot(cum_static, label="Static Strategy", linewidth=2.5)
plt.plot(cum_rebalance, label="Monthly Rebalance Strategy", linewidth=2.5, linestyle=":")
plt.plot(cum_benchmark, label="60/40 Benchmark", linewidth=2.5, linestyle="--")
plt.title("Cumulative Returns – Strategy vs Benchmark (2025–2026)", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Growth of $1", fontsize=12)
plt.legend(loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()

fig_path = os.path.join(FIGURES_DIR, "cumulative_returns.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"✔ Cumulative returns figure saved to: {fig_path}")

# ---------------------------
# 8. Performance Metrics
# ---------------------------
def calc_metrics(returns, rf=0.0):
    total_ret = (1 + returns).prod() - 1
    n_days = len(returns)
    ann_ret = (1 + total_ret) ** (252/n_days) - 1 if n_days > 0 else 0
    ann_vol = returns.std() * np.sqrt(252) if returns.std() != 0 else 0
    sharpe = (ann_ret - rf) / ann_vol if ann_vol != 0 else np.nan
    cum_ret = (1 + returns).cumprod()
    drawdown = cum_ret / cum_ret.cummax() - 1
    max_dd = drawdown.min()
    return {"Total Return": total_ret,
            "Annualized Return": ann_ret,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_dd}

metrics_static    = calc_metrics(strategy_ret_static)
metrics_rebalance = calc_metrics(strategy_ret_rebal)
metrics_benchmark = calc_metrics(benchmark_ret)

metrics_df = pd.DataFrame([metrics_static, metrics_rebalance, metrics_benchmark],
                          index=["Static Strategy", "Monthly Rebalance", "Benchmark 60/40"])
print("\n✔ Performance Metrics Comparison:\n")
print(metrics_df.round(4))

# ---------------------------
# 9. Save Metrics CSV
# ---------------------------
metrics_csv_path = os.path.join(FIGURES_DIR, "task5_performance_metrics.csv")
metrics_df.to_csv(metrics_csv_path)
print(f"✔ Performance metrics saved to CSV: {metrics_csv_path}")

# ---------------------------
# 10. Rolling Window Analysis (Optional)
# ---------------------------
window_days = 63  # ~3 months
rolling_static = (1 + strategy_ret_static).rolling(window_days).apply(lambda x: x.prod(), raw=True)
rolling_rebal  = (1 + strategy_ret_rebal).rolling(window_days).apply(lambda x: x.prod(), raw=True)
rolling_bench  = (1 + benchmark_ret).rolling(window_days).apply(lambda x: x.prod(), raw=True)

plt.figure(figsize=(12,6))
plt.plot(rolling_static, label="Static Strategy", linewidth=2)
plt.plot(rolling_rebal, label="Monthly Rebalance", linewidth=2, linestyle=":")
plt.plot(rolling_bench, label="Benchmark 60/40", linewidth=2, linestyle="--")
plt.title("3-Month Rolling Cumulative Return", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Rolling Growth of $1", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ---------------------------
# 11. Monte Carlo Simulation (Optional)
# ---------------------------
n_sim = 500
sim_cum = pd.DataFrame(index=daily_returns.index)
for i in range(n_sim):
    sim_daily = daily_returns.sample(frac=1, replace=True)
    sim_cum[i] = (1 + (sim_daily * pd.Series(OPTIMAL_WEIGHTS)).sum(axis=1)).cumprod()

plt.figure(figsize=(12,6))
plt.plot(sim_cum, color='blue', alpha=0.05)
plt.title("Monte Carlo Simulations of Strategy (500 paths)")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
