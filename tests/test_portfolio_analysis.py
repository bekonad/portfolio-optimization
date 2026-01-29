# tests/test_portfolio_analysis.py

"""
Unit tests for Task 1 → Task 4 pipeline: preprocessing, EDA, forecasting, and portfolio optimization.
"""

import os
import pytest
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier

PROJECT_ROOT = r"C:\Users\JERUSALEM\Desktop\portfolio-optimization"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")

# ------------------------------
# Test 1: Data files exist
# ------------------------------
def test_prices_file_exists():
    path = os.path.join(PROCESSED_DIR, "prices_aligned.csv")
    assert os.path.exists(path), f"Prices file not found: {path}"

def test_tsla_forecast_exists():
    path = os.path.join(PROCESSED_DIR, "tsla_lstm_forecast.csv")
    assert os.path.exists(path), f"TSLA forecast file not found: {path}"

# ------------------------------
# Test 2: Load data and basic sanity
# ------------------------------
def test_prices_dataframe():
    prices = pd.read_csv(os.path.join(PROCESSED_DIR, "prices_aligned.csv"), index_col=0, parse_dates=True)
    assert not prices.empty, "Prices DataFrame is empty"
    for col in ['TSLA','BND','SPY']:
        assert col in prices.columns, f"Column {col} missing in prices"

# ------------------------------
# Test 3: Forecast sanity
# ------------------------------
def test_tsla_forecast_values():
    forecast = pd.read_csv(os.path.join(PROCESSED_DIR, 'tsla_lstm_forecast.csv'), index_col=0, parse_dates=True)
    assert not forecast.empty, "TSLA forecast is empty"
    assert (forecast['Forecast'] > 0).all() or (forecast['Forecast'] < 1000).all(), "Forecast values look unrealistic"

# ------------------------------
# Test 4: Portfolio optimization
# ------------------------------
def test_efficient_frontier():
    prices = pd.read_csv(os.path.join(PROCESSED_DIR, "prices_aligned.csv"), index_col=0, parse_dates=True)
    prices_hist = prices.loc['2015-01-01':'2024-12-31']
    mu_hist = prices_hist.pct_change().mean() * 252
    cov_hist = prices_hist.pct_change().cov() * 252
    ef = EfficientFrontier(mu_hist, cov_hist)
    weights = ef.max_sharpe()
    assert isinstance(weights, dict), "Efficient Frontier weights should be a dict"
    for asset in ['TSLA','BND','SPY']:
        assert asset in weights, f"Asset {asset} missing in optimized weights"

# ------------------------------
# Test 5: Figures directory exists
# ------------------------------
def test_figures_dir():
    assert os.path.exists(FIGURES_DIR), f"Figures directory not found: {FIGURES_DIR}"
    subdirs = ['task1_eda','task4_portfolio']
    for d in subdirs:
        assert os.path.exists(os.path.join(FIGURES_DIR,d)), f"Subdirectory {d} missing in figures folder"