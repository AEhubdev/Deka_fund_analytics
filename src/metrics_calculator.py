import numpy as np
import pandas as pd


def compute_all_metrics(df, price_col='Price', return_col='Daily_Return', risk_free_rate=0.02, trading_days=252):
    """
    Computes performance and risk metrics for the dataset.
    """
    returns = df[return_col].dropna()
    prices = df[price_col]

    if len(returns) == 0:
        return {}

    # Basic stats
    start_price = prices.iloc[0]
    end_price = prices.iloc[-1]
    total_days = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days

    # Performance Metrics
    cumulative_return = (end_price - start_price) / start_price

    # Annualized Return (CAGR based on actual date span)
    annualized_return = (1 + cumulative_return) ** (365.0 / total_days) - 1 if total_days > 0 else 0.0

    # Risk Metrics
    daily_volatility = returns.std()
    annualized_volatility = daily_volatility * np.sqrt(trading_days)

    # Excess Returns
    excess_daily_rf = (1 + risk_free_rate) ** (1 / trading_days) - 1
    excess_returns = returns - excess_daily_rf

    # Sharpe Ratio
    sharpe_ratio = (returns.mean() - excess_daily_rf) / daily_volatility * np.sqrt(
        trading_days) if daily_volatility != 0 else np.nan

    # Downside Volatility & Sortino Ratio
    negative_returns = returns[returns < 0]
    downside_std = negative_returns.std() * np.sqrt(trading_days)
    sortino_ratio = (annualized_return - risk_free_rate) / downside_std if downside_std != 0 else np.nan

    # Drawdown Analysis
    cumulative_peaks = prices.cummax()
    drawdowns = (prices - cumulative_peaks) / cumulative_peaks
    max_drawdown = drawdowns.min()

    # Value at Risk (95% Historical) & CVaR
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

    # Statistical Higher Moments
    skewness = returns.skew()
    kurtosis = returns.kurtosis()

    metrics = {
        "Start Date": df['Date'].iloc[0].strftime('%d.%m.%Y'),
        "End Date": df['Date'].iloc[-1].strftime('%d.%m.%Y'),
        "Start Price": start_price,
        "End Price": end_price,
        "Cumulative Return": cumulative_return,
        "Annualized Return": annualized_return,
        "Daily Volatility": daily_volatility,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Max Drawdown": max_drawdown,
        "Historical VaR (95%)": var_95,
        "Historical CVaR / Expected Shortfall (95%)": cvar_95,
        "Skewness": skewness,
        "Kurtosis": kurtosis
    }

    return metrics, drawdowns