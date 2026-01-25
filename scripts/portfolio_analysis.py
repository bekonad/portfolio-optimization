# portfolio_analysis.py

import pandas as pd
import numpy as np
import os
import time
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.stats import norm

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)
plt.rcParams["axes.grid"] = True

# --------------------------------------------------
# Project setup
# --------------------------------------------------
project_root = r"C:\Users\JERUSALEM\Desktop\portfolio-optimization"
data_folder = os.path.join(project_root, "data")
figures_folder = os.path.join(project_root, "reports", "figures")
os.makedirs(data_folder, exist_ok=True)
os.makedirs(figures_folder, exist_ok=True)

# Tickers and date range
tickers = ["TSLA", "BND", "SPY"]
start_date = "2015-01-01"
end_date = "2026-01-16"

# --------------------------------------------------
# Download historical price data with retry
# --------------------------------------------------
data_dict = {}
for ticker in tickers:
    print(f"Downloading {ticker} ...")
    for attempt in range(1, 4):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, timeout=30)
            if df.empty:
                print(f"Empty data for {ticker} (attempt {attempt})")
                time.sleep(3)
                continue
            
            if 'Adj Close' in df.columns:
                series = df['Adj Close']
            elif 'Close' in df.columns:
                series = df['Close']
                print(f"⚠ Using 'Close' for {ticker} instead of 'Adj Close'")
            else:
                print(f"× No usable price column for {ticker}")
                time.sleep(3)
                continue
            
            series.name = ticker
            data_dict[ticker] = series
            print(f"✓ Success: {ticker} ({len(series)} rows)")
            break
        except Exception as e:
            print(f"Error on {ticker} (attempt {attempt}): {e}")
            time.sleep(3)
    else:
        print(f"× Failed to download {ticker} after 3 attempts")

# --------------------------------------------------
# Combine into DataFrame and forward-fill missing values
# --------------------------------------------------
data = pd.concat(data_dict.values(), axis=1)
data.columns = data_dict.keys()
data = data.ffill()

# Compute daily returns
returns = data.pct_change().dropna()

# --------------------------------------------------
# Plot daily returns
# --------------------------------------------------
returns.plot(title="Daily Returns")
plt.axhline(0, color="black", linewidth=0.5)
plt.ylabel("Daily Return")
plt.savefig(os.path.join(figures_folder, "daily_returns.png"), dpi=300)
plt.close()

# --------------------------------------------------
# Plot price series
# --------------------------------------------------
plt.figure(figsize=(12,6))
for ticker in tickers:
    plt.plot(data[ticker], label=ticker)
plt.title("Price Series")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.savefig(os.path.join(figures_folder, "price_series.png"), dpi=300)
plt.close()

# --------------------------------------------------
# Correlation heatmap
# --------------------------------------------------
plt.figure(figsize=(8,6))
sns.heatmap(returns.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation of Daily Returns")
plt.savefig(os.path.join(figures_folder, "correlation_heatmap.png"), dpi=300)
plt.close()

# --------------------------------------------------
# Risk metrics
# --------------------------------------------------
volatility = returns.std() * np.sqrt(252)
confidence_level = 0.05
VaR = returns.quantile(confidence_level)

def max_drawdown(series):
    cumulative = (1 + series).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

drawdowns = returns.apply(max_drawdown)

print("Annualized volatility:\n", volatility)
print("\n5% Value at Risk (VaR):\n", VaR)
print("\nMaximum drawdowns:\n", drawdowns)

# --------------------------------------------------
# Portfolio metrics
# --------------------------------------------------
weights = np.array([0.3, 0.5, 0.2])
portfolio_returns = returns.dot(weights)
annual_return = portfolio_returns.mean() * 252
annual_volatility = portfolio_returns.std() * np.sqrt(252)
risk_free_rate = 0.02
sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility

print("\nPortfolio Annual Return:", annual_return)
print("Portfolio Annual Volatility:", annual_volatility)
print("Portfolio Sharpe Ratio:", sharpe_ratio)

# Plot portfolio growth
plt.plot((1 + portfolio_returns).cumprod(), label="Risk-Aware Portfolio")
plt.title("Portfolio Cumulative Growth")
plt.ylabel("Cumulative Value")
plt.legend()
plt.savefig(os.path.join(figures_folder, "portfolio_growth.png"), dpi=300)
plt.close()

# --------------------------------------------------
# Save raw CSVs
# --------------------------------------------------
prices_file = os.path.join(data_folder, "prices.csv")
returns_file = os.path.join(data_folder, "returns.csv")
data.to_csv(prices_file)
returns.to_csv(returns_file)

# --------------------------------------------------
# Bonus: Align dates and save aligned CSVs
# --------------------------------------------------
full_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq='B')
aligned_data = data.reindex(full_index).ffill()
aligned_returns = aligned_data.pct_change().dropna()

aligned_prices_file = os.path.join(data_folder, "prices_aligned.csv")
aligned_returns_file = os.path.join(data_folder, "returns_aligned.csv")

aligned_data.to_csv(aligned_prices_file)
aligned_returns.to_csv(aligned_returns_file)

# --------------------------------------------------
# Bonus plots for aligned data
# --------------------------------------------------
plt.figure(figsize=(12,6))
for ticker in tickers:
    plt.plot(aligned_data[ticker], label=ticker)
plt.title("Aligned Price Series")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.savefig(os.path.join(figures_folder, "aligned_price_series.png"), dpi=300)
plt.close()

aligned_returns.plot(title="Aligned Daily Returns")
plt.axhline(0, color="black", linewidth=0.5)
plt.ylabel("Daily Return")
plt.savefig(os.path.join(figures_folder, "aligned_daily_returns.png"), dpi=300)
plt.close()

plt.figure(figsize=(8,6))
sns.heatmap(aligned_returns.corr(), annot=True, cmap="coolwarm")
plt.title("Aligned Asset Correlation Matrix")
plt.savefig(os.path.join(figures_folder, "aligned_correlation_heatmap.png"), dpi=300)
plt.close()

# --------------------------------------------------
# Final summary of all saved outputs
# --------------------------------------------------
print("\n✅ PROJECT OUTPUTS SUMMARY\n")

csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
print("Saved CSV files:")
for f in csv_files:
    print(" -", os.path.basename(f))

figure_files = glob.glob(os.path.join(figures_folder, "*.png"))
print("\nSaved Figures:")
for f in figure_files:
    print(" -", os.path.basename(f))
