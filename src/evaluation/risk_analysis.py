from pathlib import Path

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def _safe_float(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def _max_drawdown_duration(drawdown):
    longest = 0
    current = 0
    for value in drawdown.fillna(0):
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _ticker_return_frame(df, ticker):
    ticker_df = df[df["Ticker"] == ticker].copy()
    ticker_df["Date"] = pd.to_datetime(ticker_df["Date"], errors="coerce")
    ticker_df = ticker_df.dropna(subset=["Date"]).sort_values("Date")
    close = pd.to_numeric(ticker_df["Close"], errors="coerce")
    if "Daily_Return" in ticker_df.columns:
        returns = pd.to_numeric(ticker_df["Daily_Return"], errors="coerce")
    else:
        returns = close.pct_change()
    ticker_df["Close"] = close
    ticker_df["Risk_Return"] = returns
    return ticker_df.dropna(subset=["Close"]).reset_index(drop=True)


def _beta_and_correlation(asset_returns, benchmark_returns):
    aligned = pd.concat([asset_returns, benchmark_returns], axis=1, join="inner").dropna()
    if aligned.shape[0] < 3:
        return None, None

    asset = aligned.iloc[:, 0]
    benchmark = aligned.iloc[:, 1]
    benchmark_variance = benchmark.var()
    if benchmark_variance == 0 or pd.isna(benchmark_variance):
        beta = None
    else:
        beta = asset.cov(benchmark) / benchmark_variance
    correlation = asset.corr(benchmark)
    return _safe_float(beta), _safe_float(correlation)


def calculate_risk_analysis(
    df,
    benchmark_ticker="SPY",
    rolling_window=20,
    var_confidence=0.95,
):
    required_columns = {"Date", "Ticker", "Close"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Risk analysis requires columns: {sorted(required_columns)}. Missing: {sorted(missing)}")

    if df.empty:
        raise ValueError("Risk analysis requires a non-empty processed dataset.")

    frame = df.copy()
    frame["Ticker"] = frame["Ticker"].astype(str).str.upper()
    tickers = sorted(frame["Ticker"].dropna().unique().tolist())
    benchmark_ticker = str(benchmark_ticker or "").upper().strip()

    benchmark_returns = None
    if benchmark_ticker in tickers:
        benchmark_df = _ticker_return_frame(frame, benchmark_ticker)
        benchmark_returns = benchmark_df.set_index("Date")["Risk_Return"].rename("benchmark_return")

    summary_rows = []
    rolling_frames = []

    for ticker in tickers:
        ticker_df = _ticker_return_frame(frame, ticker)
        if ticker_df.empty:
            continue

        close = ticker_df["Close"]
        returns = ticker_df["Risk_Return"].dropna()
        periods = max(len(close) - 1, 1)
        total_return = close.iloc[-1] / close.iloc[0] - 1 if close.iloc[0] else np.nan
        annualized_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / periods) - 1 if pd.notna(total_return) else np.nan
        annualized_volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if not returns.empty else np.nan
        downside = returns[returns < 0]
        downside_volatility = downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if len(downside) > 1 else np.nan
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility and not pd.isna(annualized_volatility) else np.nan
        sortino_ratio = annualized_return / downside_volatility if downside_volatility and not pd.isna(downside_volatility) else np.nan

        var_level = 1 - float(var_confidence)
        value_at_risk = returns.quantile(var_level) if not returns.empty else np.nan
        tail_losses = returns[returns <= value_at_risk] if not returns.empty and pd.notna(value_at_risk) else pd.Series(dtype=float)
        conditional_var = tail_losses.mean() if not tail_losses.empty else np.nan

        running_max = close.cummax()
        drawdown = close / running_max - 1
        max_drawdown = drawdown.min()
        drawdown_duration = _max_drawdown_duration(drawdown)

        beta = None
        correlation = None
        if benchmark_returns is not None and ticker != benchmark_ticker:
            asset_returns = ticker_df.set_index("Date")["Risk_Return"].rename("asset_return")
            beta, correlation = _beta_and_correlation(asset_returns, benchmark_returns)

        summary_rows.append(
            {
                "Ticker": ticker,
                "Start_Date": ticker_df["Date"].iloc[0].date().isoformat(),
                "End_Date": ticker_df["Date"].iloc[-1].date().isoformat(),
                "Observations": int(len(ticker_df)),
                "Total_Return": _safe_float(total_return),
                "Annualized_Return": _safe_float(annualized_return),
                "Annualized_Volatility": _safe_float(annualized_volatility),
                "Downside_Volatility": _safe_float(downside_volatility),
                "Sharpe_Ratio": _safe_float(sharpe_ratio),
                "Sortino_Ratio": _safe_float(sortino_ratio),
                "VaR_95": _safe_float(value_at_risk),
                "CVaR_95": _safe_float(conditional_var),
                "Max_Drawdown": _safe_float(max_drawdown),
                "Max_Drawdown_Duration_Days": drawdown_duration,
                "Beta_vs_Benchmark": beta,
                "Correlation_vs_Benchmark": correlation,
                "Benchmark_Ticker": benchmark_ticker if benchmark_returns is not None else None,
            }
        )

        rolling = ticker_df[["Date", "Ticker"]].copy()
        rolling_returns = ticker_df["Risk_Return"]
        rolling["Rolling_Volatility"] = rolling_returns.rolling(rolling_window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        rolling["Rolling_Sharpe"] = (
            rolling_returns.rolling(rolling_window).mean() * TRADING_DAYS_PER_YEAR
        ) / rolling["Rolling_Volatility"]
        rolling["Drawdown"] = drawdown.values
        rolling_frames.append(rolling)

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(["Annualized_Volatility", "Max_Drawdown"], ascending=[False, True]).reset_index(drop=True)

    rolling = pd.concat(rolling_frames, axis=0, ignore_index=True) if rolling_frames else pd.DataFrame()
    return summary, rolling


def run_risk_analysis(
    data_file,
    summary_file,
    rolling_file,
    benchmark_ticker="SPY",
    rolling_window=20,
    var_confidence=0.95,
):
    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(f"Processed dataset not found: {data_file}")

    df = pd.read_csv(data_file)
    summary, rolling = calculate_risk_analysis(
        df,
        benchmark_ticker=benchmark_ticker,
        rolling_window=rolling_window,
        var_confidence=var_confidence,
    )

    summary_file = Path(summary_file)
    rolling_file = Path(rolling_file)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    rolling_file.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_file, index=False)
    rolling.to_csv(rolling_file, index=False)
    return summary, rolling
