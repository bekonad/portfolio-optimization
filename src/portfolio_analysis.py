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
