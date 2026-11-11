import numpy as np
import pandas as pd


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate the largest peak-to-trough decline from a return series."""
    clean_returns = returns.dropna()
    if clean_returns.empty:
        return 0.0

    cumulative_returns = (1 + clean_returns).cumprod()
    running_peaks = cumulative_returns.cummax().clip(lower=1)
    drawdowns = cumulative_returns / running_peaks - 1
    return float(drawdowns.min())


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    trading_days: int = 252,
) -> float:
    """Calculate the annualized Sharpe ratio using an annual risk-free rate."""
    if trading_days <= 0:
        raise ValueError("trading_days must be a positive integer.")

    clean_returns = returns.dropna()
    if clean_returns.empty:
        return 0.0

    daily_risk_free_rate = risk_free_rate / trading_days
    excess_returns = clean_returns - daily_risk_free_rate
    standard_deviation = excess_returns.std()
    if standard_deviation == 0 or np.isnan(standard_deviation):
        return 0.0

    return float(
        excess_returns.mean() / standard_deviation * np.sqrt(trading_days)
    )
