import os
import time
from pathlib import Path

import pandas as pd
import yfinance as yf


RAW_DATA_DIR = Path("data/raw")

PROXY = os.getenv("CLASH_PROXY", "http://127.0.0.1:7897")
USE_PROXY = os.getenv("USE_PROXY", "true").lower() in {"1", "true", "yes"}


def create_dirs():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def setup_proxy():
    if not USE_PROXY or not PROXY:
        return

    os.environ["HTTP_PROXY"] = PROXY
    os.environ["HTTPS_PROXY"] = PROXY
    os.environ["http_proxy"] = PROXY
    os.environ["https_proxy"] = PROXY
    print(f"Using proxy: {PROXY}")


def normalize_price_data(df, ticker):
    df = df.copy()

    if "Date" not in df.columns:
        df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(column).strip().title() for column in df.columns]

    date_aliases = ["Date", "Datetime", "Timestamp", "Index"]
    for column in date_aliases:
        if column in df.columns:
            df = df.rename(columns={column: "Date"})
            break

    if "Date" not in df.columns:
        for column in df.columns:
            parsed_dates = pd.to_datetime(df[column], errors="coerce")
            if parsed_dates.notna().mean() > 0.8:
                df = df.rename(columns={column: "Date"})
                break

    expected_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing_columns = [column for column in expected_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{ticker} missing columns: {missing_columns}")

    df = df[expected_columns]
    df["Date"] = pd.to_datetime(df["Date"])
    df["Ticker"] = ticker

    return df.sort_values("Date").reset_index(drop=True)


def download_ticker_data(ticker, start_date, end_date, interval):
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
        timeout=30,
        multi_level_index=False,
    )

    if df.empty:
        return pd.DataFrame()

    return normalize_price_data(df, ticker)


def download_price_data(
    tickers,
    start_date="2015-01-01",
    end_date="2025-12-31",
    interval="1d"
):
    all_data = {}

    for ticker in tickers:
        print(f"Downloading {ticker}...")

        try:
            df = download_ticker_data(ticker, start_date, end_date, interval)
        except Exception as exc:
            print(f"Warning: Failed to download {ticker}: {exc}")
            continue

        if df.empty:
            print(f"Warning: No data found for {ticker}")
            continue

        file_path = RAW_DATA_DIR / f"{ticker}.csv"
        df.to_csv(file_path, index=False)

        print(f"Saved {ticker} data to {file_path}")
        all_data[ticker] = df
        time.sleep(1)

    return all_data


if __name__ == "__main__":
    create_dirs()
    setup_proxy()

    tickers = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "JPM",
        "BAC",
        "GS",
        "SPY",
        "QQQ"
    ]

    download_price_data(
        tickers=tickers,
        start_date="2015-01-01",
        end_date="2025-12-31",
        interval="1d"
    )
