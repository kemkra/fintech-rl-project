import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def calculate_performance_metrics(equity_curve, date_column="Date", value_column="Portfolio_Value"):
    df = equity_curve.copy()
    if df.empty:
        raise ValueError("equity_curve cannot be empty.")

    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column).reset_index(drop=True)
    values = pd.to_numeric(df[value_column], errors="coerce")

    if values.isna().any():
        raise ValueError(f"{value_column} contains missing or non-numeric values.")

    daily_returns = values.pct_change().dropna()
    total_return = values.iloc[-1] / values.iloc[0] - 1
    periods = max(len(values) - 1, 1)
    annualized_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / periods) - 1
    annualized_volatility = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    if annualized_volatility == 0 or np.isnan(annualized_volatility):
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = annualized_return / annualized_volatility

    running_max = values.cummax()
    drawdown = values / running_max - 1
    max_drawdown = drawdown.min()
    win_rate = (daily_returns > 0).mean() if not daily_returns.empty else np.nan

    return {
        "start_date": df[date_column].iloc[0].date().isoformat(),
        "end_date": df[date_column].iloc[-1].date().isoformat(),
        "start_value": float(values.iloc[0]),
        "end_value": float(values.iloc[-1]),
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe_ratio) if not np.isnan(sharpe_ratio) else None,
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate) if not np.isnan(win_rate) else None,
    }
