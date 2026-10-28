import numpy as np
import pandas as pd


def compute_all_metrics(df, price_col='Price', return_col='Daily_Return', risk_free_rate=0.02, trading_days=252):
    """
    Computes comprehensive performance and risk metrics for the Deka fund dataset.
    """
    returns = df[return_col].dropna()
    prices = df[price_col]

    if len(returns) == 0:
        return {}, pd.Series()

    start_price = prices.iloc[0]
    end_price = prices.iloc[-1]
    total_days = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days

    # Performance
    cumulative_return = (end_price - start_price) / start_price
    annualized_return = (1 + cumulative_return) ** (365.0 / total_days) - 1 if total_days > 0 else 0.0

    # Return distribution stats
    best_day = returns.max()
    worst_day = returns.min()
    positive_days_pct = (returns > 0).mean()

    # Risk Metrics
    daily_volatility = returns.std()
    annualized_volatility = daily_volatility * np.sqrt(trading_days)

    # Excess Returns & Sharpe
    excess_daily_rf = (1 + risk_free_rate) ** (1 / trading_days) - 1
    sharpe_ratio = (returns.mean() - excess_daily_rf) / daily_volatility * np.sqrt(
        trading_days) if daily_volatility != 0 else np.nan

    # Sortino Ratio
    negative_returns = returns[returns < 0]
    downside_std = negative_returns.std() * np.sqrt(trading_days)
    sortino_ratio = (annualized_return - risk_free_rate) / downside_std if downside_std != 0 else np.nan

    # Drawdowns
    cumulative_peaks = prices.cummax()
    drawdowns = (prices - cumulative_peaks) / cumulative_peaks
    max_drawdown = drawdowns.min()

    # Calmar Ratio (Annualized Return / |Max Drawdown|)
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

    # Value at Risk & Conditional VaR (95%)
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

    # Higher Moments
    skewness = returns.skew()
    kurtosis = returns.kurtosis()

    metrics = {
        "Analysis Period": f"{df['Date'].iloc[0].strftime('%d.%m.%Y')} to {df['Date'].iloc[-1].strftime('%d.%m.%Y')}",
        "Total Trading Days": len(returns),
        "Start NAV / Price": start_price,
        "End NAV / Price": end_price,
        "Cumulative Return": cumulative_return,
        "Annualized Return (CAGR)": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio (rf=2%)": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Calmar Ratio": calmar_ratio,
        "Maximum Drawdown": max_drawdown,
        "Historical VaR (95% Daily)": var_95,
        "Historical Expected Shortfall / CVaR (95%)": cvar_95,
        "Best Daily Return": best_day,
        "Worst Daily Return": worst_day,
        "Positive Days Ratio": positive_days_pct,
        "Return Skewness": skewness,
        "Return Kurtosis": kurtosis
    }

    return metrics, drawdowns