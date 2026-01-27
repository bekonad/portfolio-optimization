"""
Task 3: Portfolio Analysis & Optimization
Uses forecast outputs to compute expected returns and risk
"""

import numpy as np
import pandas as pd


def compute_expected_returns(forecast_df):
    """
    Compute expected returns from forecasted prices.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Columns: ['date', 'forecast']

    Returns
    -------
    float
        Expected return
    """
    prices = forecast_df["forecast"].values
    returns = np.diff(prices) / prices[:-1]
    return returns.mean()


def compute_volatility(forecast_df):
    """
    Compute forecasted volatility.

    Returns
    -------
    float
    """
    prices = forecast_df["forecast"].values
    returns = np.diff(prices) / prices[:-1]
    return returns.std()


def sharpe_ratio(expected_return, volatility, risk_free_rate=0.0):
    """
    Compute Sharpe Ratio.
    """
    if volatility == 0:
        return np.nan
    return (expected_return - risk_free_rate) / volatility

"""
Task 3 & Task 4: Portfolio Analysis and Optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Task 3 — Forecast Metrics
# =========================

def compute_forecast_metrics(forecast_series: pd.Series):
    """
    Compute expected return and annualized volatility from forecasted prices.
    """
    returns = forecast_series.pct_change().dropna()
    expected_return = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    return expected_return, annual_volatility


def build_portfolio_forecast(forecasts: dict, weights: dict):
    """
    Combine asset forecasts using portfolio weights.
    """
    portfolio = sum(
        weights[ticker] * forecasts[ticker]
        for ticker in weights
    )
    return portfolio


def plot_portfolio_forecast(portfolio_series, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_series, label="Portfolio Forecast", linewidth=2)
    plt.title("Combined Portfolio Forecast")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================
# Task 4 — Optimization
# =========================

def mean_variance_optimization(expected_returns, cov_matrix, risk_free_rate=0.0):
    """
    Compute maximum Sharpe ratio portfolio weights.
    """
    inv_cov = np.linalg.inv(cov_matrix)
    excess_returns = expected_returns - risk_free_rate
    weights = inv_cov @ excess_returns
    weights /= weights.sum()
    return weights


def portfolio_performance(weights, expected_returns, cov_matrix):
    """
    Compute portfolio return and volatility.
    """
    port_return = weights @ expected_returns
    port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    return port_return, port_vol
