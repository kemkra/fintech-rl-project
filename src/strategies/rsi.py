from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import calculate_performance_metrics
from src.strategies.buy_hold import load_feature_data


RESULTS_DIR = Path("reports/results")
RSI_EQUITY_FILE = RESULTS_DIR / "rsi_equity_curves.csv"
RSI_METRICS_FILE = RESULTS_DIR / "rsi_metrics.csv"


def create_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_rsi_for_ticker(
    df,
    ticker,
    initial_cash=100_000.0,
    buy_threshold=30,
    sell_threshold=70,
):
    ticker = ticker.upper()
    ticker_df = df[df["Ticker"] == ticker].copy()
    if ticker_df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")
    if "RSI" not in ticker_df.columns:
        raise ValueError(f"{ticker} missing RSI column.")

    ticker_df = ticker_df.sort_values("Date").reset_index(drop=True)
    raw_signal = pd.Series(index=ticker_df.index, dtype="float64")
    raw_signal[ticker_df["RSI"] < buy_threshold] = 1
    raw_signal[ticker_df["RSI"] > sell_threshold] = 0
    ticker_df["Raw_Signal"] = raw_signal.ffill().fillna(0).astype(int)
    ticker_df["Position"] = ticker_df["Raw_Signal"].shift(1).fillna(0).astype(int)

    cash = float(initial_cash)
    shares = 0
    rows = []

    for row in ticker_df.itertuples(index=False):
        price = float(row.Close)
        target_position = int(row.Position)
        trade_shares = 0

        if target_position == 1 and shares == 0:
            shares = int(cash // price)
            trade_shares = shares
            cash -= shares * price
        elif target_position == 0 and shares > 0:
            trade_shares = -shares
            cash += shares * price
            shares = 0

        portfolio_value = cash + shares * price
        rows.append(
            {
                "Date": row.Date,
                "Ticker": ticker,
                "Close": price,
                "RSI": float(row.RSI),
                "Strategy": "RSI",
                "Buy_Threshold": float(buy_threshold),
                "Sell_Threshold": float(sell_threshold),
                "Raw_Signal": int(row.Raw_Signal),
                "Position": target_position,
                "Trade_Shares": int(trade_shares),
                "Shares": int(shares),
                "Cash": float(cash),
                "Portfolio_Value": float(portfolio_value),
            }
        )

    equity_curve = pd.DataFrame(rows)
    equity_curve["Daily_Return"] = equity_curve["Portfolio_Value"].pct_change().fillna(0)
    equity_curve["Cumulative_Return"] = equity_curve["Portfolio_Value"] / initial_cash - 1

    metrics = calculate_performance_metrics(equity_curve)
    metrics["Ticker"] = ticker
    metrics["Strategy"] = "RSI"
    metrics["Initial_Cash"] = float(initial_cash)
    metrics["Buy_Threshold"] = float(buy_threshold)
    metrics["Sell_Threshold"] = float(sell_threshold)
    metrics["Trades"] = int((equity_curve["Trade_Shares"] != 0).sum())
    metrics["Final_Shares"] = int(equity_curve["Shares"].iloc[-1])
    metrics["Final_Cash"] = float(equity_curve["Cash"].iloc[-1])

    return equity_curve, metrics


def run_rsi_strategy(
    tickers=None,
    initial_cash=100_000.0,
    buy_threshold=30,
    sell_threshold=70,
):
    create_dirs()
    df = load_feature_data()

    if tickers is None:
        tickers = sorted(df["Ticker"].unique())
    tickers = [ticker.upper() for ticker in tickers]

    equity_curves = []
    metrics_rows = []

    for ticker in tickers:
        equity_curve, metrics = run_rsi_for_ticker(
            df=df,
            ticker=ticker,
            initial_cash=initial_cash,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )
        equity_curves.append(equity_curve)
        metrics_rows.append(metrics)

    equity_df = pd.concat(equity_curves, axis=0, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df[
        [
            "Ticker",
            "Strategy",
            "start_date",
            "end_date",
            "start_value",
            "end_value",
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "Initial_Cash",
            "Buy_Threshold",
            "Sell_Threshold",
            "Trades",
            "Final_Shares",
            "Final_Cash",
        ]
    ].sort_values("total_return", ascending=False)

    equity_df.to_csv(RSI_EQUITY_FILE, index=False)
    metrics_df.to_csv(RSI_METRICS_FILE, index=False)

    print(f"RSI equity curves saved to {RSI_EQUITY_FILE}")
    print(f"RSI metrics saved to {RSI_METRICS_FILE}")
    print(f"Tickers analyzed: {len(tickers)}")

    return equity_df, metrics_df


if __name__ == "__main__":
    run_rsi_strategy()
