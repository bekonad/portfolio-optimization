# =============================================================================
# task1_eda.py
# Modular & Reusable EDA Script for Week 9 KAIM Challenge
# Task 1: Data Preprocessing and Exploratory Data Analysis
# Author: Bereket Feleke (Dororo)
# =============================================================================

import os
import time
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from scipy.stats import norm


# Set plotting style (global)
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.grid"] = True


def create_directories(base_path: str = "..") -> Tuple[str, str, str]:
    """Create standardized project folders if they don't exist."""
    raw_folder      = os.path.join(base_path, "data", "raw")
    processed_folder = os.path.join(base_path, "data", "processed")
    figures_folder   = os.path.join(base_path, "reports", "figures")

    for folder in [raw_folder, processed_folder, figures_folder]:
        os.makedirs(folder, exist_ok=True)

    return processed_folder, figures_folder


def fetch_prices(
    tickers: List[str],
    start_date: str,
    end_date: str,
    retries: int = 3
) -> pd.DataFrame:
    """
    Fetch adjusted closing prices from yfinance with retry logic.
    Returns a DataFrame with tickers as columns and dates as index.
    """
    print("=== Fetching historical data ===")
    data_dict = {}

    for ticker in tickers:
        print(f"Downloading {ticker}...")
        for attempt in range(1, retries + 1):
            try:
                df = yf.download(ticker, start=start_date, end=end_date,
                                 progress=False, timeout=30)

                if df.empty:
                    time.sleep(3)
                    continue

                series = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
                series.name = ticker
                data_dict[ticker] = series
                break

            except Exception as e:
                print(f"Retry {attempt} failed for {ticker}: {e}")
                time.sleep(3)

    # Combine and forward-fill
    prices = pd.concat(data_dict.values(), axis=1)
    prices.columns = data_dict.keys()
    prices = prices.ffill()

    return prices


def clean_and_validate(prices: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: forward/backward fill remaining NaNs and print summary."""
    print("\nData shape:", prices.shape)
    print("\nMissing values before cleaning:\n", prices.isna().sum())

    prices = prices.ffill().bfill()

    print("\nMissing values after cleaning:\n", prices.isna().sum())
    return prices


def compute_returns_and_volatility(
    prices: pd.DataFrame,
    vol_window: int = 21
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute daily returns and rolling annualized volatility."""
    returns = prices.pct_change().dropna()
    rolling_vol = returns.rolling(window=vol_window).std() * np.sqrt(252)

    return returns, rolling_vol


def plot_price_series(prices: pd.DataFrame, save_path: str) -> None:
    """Plot historical price series and save figure."""
    plt.figure()
    for col in prices.columns:
        plt.plot(prices.index, prices[col], label=col)
    plt.title("Historical Adjusted Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Figure saved: {save_path}")


def plot_daily_returns(returns: pd.DataFrame, save_path: str) -> None:
    """Plot daily returns with zero line."""
    returns.plot(title="Daily Returns")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.ylabel("Daily Return")
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_rolling_volatility(rolling_vol: pd.DataFrame, save_path: str) -> None:
    """Plot rolling annualized volatility."""
    rolling_vol.plot(title="21-Day Rolling Annualized Volatility")
    plt.ylabel("Annualized Volatility")
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_correlation_heatmap(returns: pd.DataFrame, save_path: str) -> None:
    """Plot correlation heatmap of daily returns."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(returns.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
    plt.title("Correlation Matrix of Daily Returns")
    plt.savefig(save_path, dpi=300)
    plt.close()


def adf_test(series: pd.Series, name: str) -> None:
    """Perform Augmented Dickey-Fuller test for stationarity."""
    result = adfuller(series.dropna())
    print(f"\nADF Test for {name}:")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value:       {result[1]:.4f}")
    print("→ Stationary" if result[1] < 0.05 else "→ NOT Stationary (needs differencing)")


def compute_risk_metrics(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate basic annualized risk metrics (returns, volatility, Sharpe, VaR)."""
    ann_returns = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_returns / ann_vol  # assuming Rf = 0
    var_95 = returns.quantile(0.05)

    metrics = pd.DataFrame({
        "Annualized Return": ann_returns,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio (Rf=0)": sharpe,
        "Daily 95% VaR": var_95
    })

    return metrics


def run_task1_eda(
    tickers: List[str] = ["TSLA", "BND", "SPY"],
    start_date: str = "2015-01-01",
    end_date: str = "2026-01-16",
    base_path: str = "..",
    vol_window: int = 21
) -> None:
    """
    Main function to run complete Task 1 EDA pipeline.
    Saves all processed data and figures in standard folders.
    """
    processed_folder, figures_folder = create_directories(base_path)

    # Fetch & clean data
    prices = fetch_prices(tickers, start_date, end_date)
    prices = clean_and_validate(prices)

    # Save cleaned prices
    prices.to_csv(os.path.join(processed_folder, "cleaned_prices.csv"))

    # Compute returns & volatility
    returns, rolling_vol = compute_returns_and_volatility(prices, vol_window)
    returns.to_csv(os.path.join(processed_folder, "returns.csv"))
    rolling_vol.to_csv(os.path.join(processed_folder, "rolling_volatility.csv"))

    # Visualizations
    plot_price_series(prices, os.path.join(figures_folder, "price_series.png"))
    plot_daily_returns(returns, os.path.join(figures_folder, "daily_returns.png"))
    plot_rolling_volatility(rolling_vol, os.path.join(figures_folder, "rolling_volatility.png"))
    plot_correlation_heatmap(returns, os.path.join(figures_folder, "correlation_heatmap.png"))

    # Stationarity tests
    print("\n=== Stationarity Tests ===")
    adf_test(prices["TSLA"], "TSLA Prices")
    adf_test(returns["TSLA"], "TSLA Daily Returns")

    # Risk metrics
    print("\n=== Basic Risk Metrics (full period) ===")
    risk_metrics = compute_risk_metrics(returns)
    print(risk_metrics)
    risk_metrics.to_csv(os.path.join(processed_folder, "risk_metrics.csv"))

    # Summary
    print("\n" + "="*70)
    print("TASK 1 EDA COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nProcessed files saved in: {processed_folder}")
    print(f"Figures saved in: {figures_folder}")


# =============================================================================
# Run the pipeline (can be imported or run directly)
# =============================================================================
if __name__ == "__main__":
    run_task1_eda()