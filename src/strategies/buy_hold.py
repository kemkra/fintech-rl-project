from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import calculate_performance_metrics


PROCESSED_DATA_FILE = Path("data/processed/stock_features.csv")
RESULTS_DIR = Path("reports/results")
BUY_HOLD_EQUITY_FILE = RESULTS_DIR / "buy_hold_equity_curves.csv"
BUY_HOLD_METRICS_FILE = RESULTS_DIR / "buy_hold_metrics.csv"


def create_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_feature_data(file_path=PROCESSED_DATA_FILE):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path} not found. Run src/features/feature_engineering.py first."
        )

    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def run_buy_hold_for_ticker(df, ticker, initial_cash=100_000.0):
    ticker = ticker.upper()
    ticker_df = df[df["Ticker"] == ticker].copy()
    if ticker_df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    ticker_df = ticker_df.sort_values("Date").reset_index(drop=True)
    first_price = float(ticker_df.loc[0, "Close"])
    shares = int(initial_cash // first_price)
    remaining_cash = initial_cash - shares * first_price

    equity_curve = ticker_df[["Date", "Ticker", "Close"]].copy()
    equity_curve["Strategy"] = "BuyHold"
    equity_curve["Shares"] = shares
    equity_curve["Cash"] = remaining_cash
    equity_curve["Portfolio_Value"] = remaining_cash + shares * equity_curve["Close"]
    equity_curve["Daily_Return"] = equity_curve["Portfolio_Value"].pct_change().fillna(0)
    equity_curve["Cumulative_Return"] = equity_curve["Portfolio_Value"] / initial_cash - 1

    metrics = calculate_performance_metrics(equity_curve)
    metrics["Ticker"] = ticker
    metrics["Strategy"] = "BuyHold"
    metrics["Initial_Cash"] = float(initial_cash)
    metrics["Shares"] = int(shares)
    metrics["Remaining_Cash"] = float(remaining_cash)

    return equity_curve, metrics


def run_buy_hold_strategy(tickers=None, initial_cash=100_000.0):
    create_dirs()
    df = load_feature_data()

    if tickers is None:
        tickers = sorted(df["Ticker"].unique())
    tickers = [ticker.upper() for ticker in tickers]

    equity_curves = []
    metrics_rows = []

    for ticker in tickers:
        equity_curve, metrics = run_buy_hold_for_ticker(
            df=df,
            ticker=ticker,
            initial_cash=initial_cash,
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
            "Shares",
            "Remaining_Cash",
        ]
    ].sort_values("total_return", ascending=False)

    equity_df.to_csv(BUY_HOLD_EQUITY_FILE, index=False)
    metrics_df.to_csv(BUY_HOLD_METRICS_FILE, index=False)

    print(f"Buy & Hold equity curves saved to {BUY_HOLD_EQUITY_FILE}")
    print(f"Buy & Hold metrics saved to {BUY_HOLD_METRICS_FILE}")
    print(f"Tickers analyzed: {len(tickers)}")

    return equity_df, metrics_df


if __name__ == "__main__":
    run_buy_hold_strategy()
