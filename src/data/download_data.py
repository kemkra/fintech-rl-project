import os
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf


RAW_DATA_DIR = Path("data/raw")

PROXY = os.getenv("CLASH_PROXY", "")
USE_PROXY = os.getenv("USE_PROXY", "false").lower() in {"1", "true", "yes"}


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


def clear_proxy():
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
        os.environ.pop(key, None)


def to_unix_timestamp(date_value):
    return int(pd.Timestamp(date_value).tz_localize("UTC").timestamp())


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
        return download_ticker_data_yahoo_chart(ticker, start_date, end_date, interval)

    normalized = normalize_price_data(df, ticker)
    normalized.attrs["source"] = "yfinance"
    return normalized


def download_ticker_data_yahoo_chart(ticker, start_date, end_date, interval):
    if interval != "1d":
        return pd.DataFrame()

    params = urlencode(
        {
            "period1": to_unix_timestamp(start_date),
            "period2": to_unix_timestamp(end_date),
            "interval": interval,
            "events": "history",
            "includeAdjustedClose": "true",
        }
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?{params}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        payload = response.read().decode("utf-8")

    data = json.loads(payload)
    chart = data.get("chart", {})
    if chart.get("error"):
        raise ValueError(f"Yahoo chart API error for {ticker}: {chart['error']}")

    results = chart.get("result") or []
    if not results:
        return pd.DataFrame()

    result = results[0]
    timestamps = result.get("timestamp") or []
    quotes = (result.get("indicators", {}).get("quote") or [{}])[0]
    if not timestamps or not quotes:
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).date,
            "Open": quotes.get("open"),
            "High": quotes.get("high"),
            "Low": quotes.get("low"),
            "Close": quotes.get("close"),
            "Volume": quotes.get("volume"),
        }
    )
    normalized = normalize_price_data(df, ticker)
    normalized.attrs["source"] = "yahoo_chart"
    return normalized


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
