"""
Task 5 Backtesting Dashboard Script
----------------------------------
Interactive Streamlit backtesting for Task 5 portfolio optimization.
Features:
- Static portfolio, monthly/weekly rebalancing
- Rolling window (~3 months)
- Monte Carlo simulation
- Dynamic weights, transaction costs for strategy and benchmark
- Real-time metrics comparison table
- Scenario saving with metadata
- Reset to Task 4 Optimal Weights button
- Quick Comparison Plot with optional log scale
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

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
# 2. Load Prices
# ---------------------------
prices = pd.read_csv(os.path.join(PROCESSED_DIR, "prices_aligned.csv"), 
                     index_col=0, parse_dates=True)
prices_bt = prices.loc["2025-01-01":"2026-01-15"]
daily_returns = prices_bt.pct_change().dropna()
print(f"✔ Backtest data loaded. Shape: {prices_bt.shape}")

# ---------------------------
# 3. Portfolio Weights
# ---------------------------
OPTIMAL_WEIGHTS = {"TSLA": 0.0, "SPY": 0.60, "BND": 0.40}
BENCHMARK_WEIGHTS = {"TSLA": 0.0, "SPY": 0.60, "BND": 0.40}

# ---------------------------
# 4. Portfolio Functions
# ---------------------------
def portfolio_returns(returns_df, weights_dict):
    return (returns_df * pd.Series(weights_dict)).sum(axis=1)

def portfolio_returns_rebalance(returns_df, weights_dict, rebalance_freq='M', transaction_cost=0.0):
    weights_prev = pd.Series(weights_dict)
    port_ret = []
    rebalance_dates = returns_df.resample(rebalance_freq).first().index if rebalance_freq else []
    for date, row in returns_df.iterrows():
        if date in rebalance_dates:
            weights_new = pd.Series(weights_dict)
            turnover = (weights_new - weights_prev).abs().sum()
            weights_prev = weights_new.copy()
        else:
            turnover = 0.0
        daily_ret = (row * weights_prev).sum() - turnover * transaction_cost
        port_ret.append(daily_ret)
    return pd.Series(port_ret, index=returns_df.index)

# ---------------------------
# 5. Performance Metrics
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

# ---------------------------
# 6. Streamlit Dashboard
# ---------------------------
try:
    import streamlit as st
    st.set_page_config(page_title="Task 5 Backtesting Dashboard", layout="wide")

    st.title("📈 Task 5 Backtesting Interactive Dashboard")
    st.write("Adjust portfolio weights, rebalance frequency, and transaction costs to explore different scenarios.")

    # Sidebar: Portfolio weights
    st.sidebar.header("Strategy Portfolio Settings")
    tsla_w = st.sidebar.slider("TSLA Weight", 0.0, 1.0, float(OPTIMAL_WEIGHTS["TSLA"]), 0.01)
    spy_w  = st.sidebar.slider("SPY Weight", 0.0, 1.0, float(OPTIMAL_WEIGHTS["SPY"]), 0.01)
    bnd_w  = st.sidebar.slider("BND Weight", 0.0, 1.0, float(OPTIMAL_WEIGHTS["BND"]), 0.01)

    # Normalize weights
    total_w = tsla_w + spy_w + bnd_w
    if total_w > 0:
        tsla_w /= total_w
        spy_w  /= total_w
        bnd_w  /= total_w
    else:
        tsla_w, spy_w, bnd_w = OPTIMAL_WEIGHTS["TSLA"], OPTIMAL_WEIGHTS["SPY"], OPTIMAL_WEIGHTS["BND"]
    dynamic_weights = {"TSLA": tsla_w, "SPY": spy_w, "BND": bnd_w}

    # Sidebar: Reset to Task 4 Optimal
    st.sidebar.header("Quick Actions")
    if st.sidebar.button("🔄 Reset to Task 4 Optimal Weights"):
        tsla_w = OPTIMAL_WEIGHTS["TSLA"]
        spy_w  = OPTIMAL_WEIGHTS["SPY"]
        bnd_w  = OPTIMAL_WEIGHTS["BND"]
        st.experimental_rerun()

    # Rebalance frequency
    rebalance_option = st.sidebar.selectbox("Rebalance Frequency", ["None (Static)", "Weekly", "Monthly"])
    rebalance_map = {"None (Static)": None, "Weekly": 'W', "Monthly": 'M'}

    # Transaction costs
    st.sidebar.header("Transaction Costs")
    use_tc_strategy = st.sidebar.checkbox("Include Strategy Transaction Costs", value=True)
    tc_strategy = st.sidebar.number_input(
        "Strategy Cost per Trade (%)", min_value=0.0, max_value=5.0, value=0.1, step=0.05)/100 if use_tc_strategy else 0.0

    use_tc_benchmark = st.sidebar.checkbox("Include Benchmark Transaction Costs", value=True)
    tc_benchmark = st.sidebar.number_input(
        "Benchmark Cost per Trade (%)", min_value=0.0, max_value=5.0, value=0.05, step=0.05)/100 if use_tc_benchmark else 0.0

    # Scenario metadata
    st.sidebar.header("Scenario Metadata")
    scenario_name = st.sidebar.text_input("Scenario Name", value=f"Scenario_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    scenario_comment = st.sidebar.text_area("Optional Comments / Notes", value="")

    # ---------------------------
    # Compute returns
    # ---------------------------
    static_strategy_ret = portfolio_returns(daily_returns, dynamic_weights)
    rebalance_strategy_ret = portfolio_returns_rebalance(
        daily_returns, dynamic_weights,
        rebalance_freq=rebalance_map.get(rebalance_option, None),
        transaction_cost=tc_strategy
    )
    benchmark_ret = portfolio_returns_rebalance(
        daily_returns, BENCHMARK_WEIGHTS,
        rebalance_freq=rebalance_map.get(rebalance_option, None),
        transaction_cost=tc_benchmark
    )

    # Cumulative returns
    cum_static = (1 + static_strategy_ret).cumprod()
    cum_rebalance = (1 + rebalance_strategy_ret).cumprod()
    cum_benchmark = (1 + benchmark_ret).cumprod()

    # ---------------------------
    # Display plots
    # ---------------------------
    st.subheader("📈 Cumulative Returns")
    st.line_chart(pd.DataFrame({
        "Static": cum_static,
        "Rebalance": cum_rebalance,
        "Benchmark 60/40": cum_benchmark
    }))

    # Rolling window (~3 months)
    window_days = 63
    rolling_static = (1 + static_strategy_ret).rolling(window_days).apply(lambda x: x.prod(), raw=True)
    rolling_rebalance = (1 + rebalance_strategy_ret).rolling(window_days).apply(lambda x: x.prod(), raw=True)
    rolling_benchmark = (1 + benchmark_ret).rolling(window_days).apply(lambda x: x.prod(), raw=True)
    st.subheader(f"📊 {window_days}-Day Rolling Cumulative Returns")
    st.line_chart(pd.DataFrame({
        "Static": rolling_static,
        "Rebalance": rolling_rebalance,
        "Benchmark 60/40": rolling_benchmark
    }))

    # Monte Carlo simulation
    n_sim = 200
    sim_cum = pd.DataFrame(index=daily_returns.index)
    for i in range(n_sim):
        sim_daily = daily_returns.sample(frac=1, replace=True)
        sim_cum[i] = (1 + (sim_daily * pd.Series(dynamic_weights)).sum(axis=1)).cumprod()
    st.subheader(f"🎲 Monte Carlo Simulation ({n_sim} paths)")
    st.line_chart(sim_cum)

    # Metrics table
    metrics_static = calc_metrics(static_strategy_ret)
    metrics_rebalance = calc_metrics(rebalance_strategy_ret)
    metrics_benchmark = calc_metrics(benchmark_ret)
    comparison_df = pd.DataFrame(
        [metrics_static, metrics_rebalance, metrics_benchmark],
        index=["Static Strategy", f"Rebalance ({rebalance_option})", "Benchmark 60/40"]
    )
    st.subheader("📊 Real-Time Portfolio Comparison Metrics")
    st.dataframe(comparison_df.round(4))

    # ---------------------------
    # Quick Comparison Plot
    # ---------------------------
    st.sidebar.header("Quick Comparison Plot")
    show_log = st.sidebar.checkbox("Logarithmic Y-axis", value=False)
    if st.sidebar.button("📊 Generate Quick Comparison Plot"):
        plt.figure(figsize=(12,6))
        plt.plot(cum_static, label="Static", linewidth=2.5)
        plt.plot(cum_rebalance, label="Rebalance", linewidth=2.5, linestyle="--")
        plt.plot(cum_benchmark, label="Benchmark 60/40", linewidth=2.5, linestyle=":")
        plt.title("Quick Comparison: Static vs Rebalance vs Benchmark")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Growth ($)")
        plt.legend()
        plt.grid(alpha=0.3)
        if show_log:
            plt.yscale("log")
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()

    # ---------------------------
    # Save Scenario
    # ---------------------------
    import matplotlib
    matplotlib.use("Agg")
    if st.button("💾 Save Current Scenario"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = scenario_name.replace(" ", "_").replace("/", "_")
        scenario_dir = os.path.join(PROJECT_ROOT, "reports", "scenarios", f"{safe_name}_{timestamp}")
        os.makedirs(scenario_dir, exist_ok=True)

        # Cumulative returns figure
        plt.figure(figsize=(12,6))
        plt.plot(cum_static, label="Static", linewidth=2.5)
        plt.plot(cum_rebalance, label="Rebalance", linewidth=2.5, linestyle="--")
        plt.plot(cum_benchmark, label="Benchmark 60/40", linewidth=2.5, linestyle=":")
        plt.title("Cumulative Returns")
        plt.xlabel("Date")
        plt.ylabel("Growth of $1")
        plt.legend()
        plt.grid(alpha=0.3)
        cum_path = os.path.join(scenario_dir, "cumulative_returns.png")
        plt.savefig(cum_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Rolling returns figure
        plt.figure(figsize=(12,6))
        plt.plot(rolling_static, label="Static", linewidth=2)
        plt.plot(rolling_rebalance, label="Rebalance", linewidth=2, linestyle="--")
        plt.plot(rolling_benchmark, label="Benchmark 60/40", linewidth=2, linestyle=":")
        plt.title(f"{window_days}-Day Rolling Cumulative Returns")
        plt.xlabel("Date")
        plt.ylabel("Rolling Growth")
        plt.legend()
        plt.grid(alpha=0.3)
        rolling_path = os.path.join(scenario_dir, "rolling_returns.png")
        plt.savefig(rolling_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Save metrics CSV
        metrics_file = os.path.join(scenario_dir, "performance_metrics.csv")
        comparison_df.to_csv(metrics_file)

        # Save metadata
        meta_file = os.path.join(scenario_dir, "metadata.txt")
        with open(meta_file, "w") as f:
            f.write(f"Scenario Name: {scenario_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("Comments:\n")
            f.write(scenario_comment.strip() + "\n\n")
            f.write("Strategy Portfolio Weights:\n")
            for k, v in dynamic_weights.items():
                f.write(f"{k}: {v:.4f}\n")
            f.write("Benchmark Portfolio Weights:\n")
            for k, v in BENCHMARK_WEIGHTS.items():
                f.write(f"{k}: {v:.4f}\n")
            f.write(f"Rebalance Frequency: {rebalance_option}\n")
            f.write(f"Strategy Transaction Costs: {tc_strategy:.4f}\n")
            f.write(f"Benchmark Transaction Costs: {tc_benchmark:.4f}\n")

        st.success(f"✅ Scenario saved to `{scenario_dir}`")
        st.write(f"Files generated:\n- {cum_path}\n- {rolling_path}\n- {metrics_file}\n- {meta_file}")

except ModuleNotFoundError:
    print("Streamlit not installed. Install via `pip install streamlit` to run the interactive dashboard.")
