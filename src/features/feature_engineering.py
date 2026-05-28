from pathlib import Path

import numpy as np
import pandas as pd


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_FILE = PROCESSED_DATA_DIR / "stock_features.csv"


def create_dirs():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data(raw_data_dir=RAW_DATA_DIR):
    raw_data_dir = Path(raw_data_dir)
    csv_files = sorted(
        path
        for path in raw_data_dir.glob("*.csv")
        if path.stem.isupper()
    )

    if not csv_files:
        raise FileNotFoundError(
            "No raw CSV files found. Run src/data/download_data.py before feature engineering."
        )

    frames = []
    for path in csv_files:
        df = pd.read_csv(path)
        if "Ticker" not in df.columns:
            df["Ticker"] = path.stem
        frames.append(df)

    return pd.concat(frames, axis=0, ignore_index=True)


def normalize_price_columns(df):
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]

    unnamed_columns = [column for column in df.columns if column.startswith("Unnamed")]
    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)

    required_columns = {"Date", "Close", "Ticker"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df["Date"] = pd.to_datetime(df["Date"])

    numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def calculate_rsi(close_prices, window=14):
    delta = close_prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.rolling(window=window, min_periods=window).mean()
    average_loss = loss.rolling(window=window, min_periods=window).mean()

    relative_strength = average_gain / average_loss
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.replace([np.inf, -np.inf], 100)


def add_technical_indicators(df):
    df = normalize_price_columns(df)
    grouped = df.groupby("Ticker", group_keys=False)

    df["Daily_Return"] = grouped["Close"].pct_change()
    df["MA5"] = grouped["Close"].transform(
        lambda prices: prices.rolling(window=5, min_periods=5).mean()
    )
    df["MA20"] = grouped["Close"].transform(
        lambda prices: prices.rolling(window=20, min_periods=20).mean()
    )
    df["RSI"] = grouped["Close"].transform(calculate_rsi)

    ema12 = grouped["Close"].transform(lambda prices: prices.ewm(span=12, adjust=False).mean())
    ema26 = grouped["Close"].transform(lambda prices: prices.ewm(span=26, adjust=False).mean())
    df["MACD"] = ema12 - ema26
    df["Volatility"] = grouped["Daily_Return"].transform(
        lambda returns: returns.rolling(window=20, min_periods=20).std()
    )

    feature_columns = ["Daily_Return", "MA5", "MA20", "RSI", "MACD", "Volatility"]
    df = df.dropna(subset=feature_columns).reset_index(drop=True)

    return df


def save_processed_data(df, output_path=PROCESSED_FILE):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def run_feature_engineering(raw_df=None, output_path=PROCESSED_FILE):
    create_dirs()
    if raw_df is None:
        raw_df = load_raw_data()
    feature_df = add_technical_indicators(raw_df)
    output_path = save_processed_data(feature_df, output_path=output_path)
    print(f"Processed feature data saved to {output_path}")
    print(f"Rows: {len(feature_df)}")
    print(f"Tickers: {feature_df['Ticker'].nunique()}")
    return feature_df


if __name__ == "__main__":
    run_feature_engineering()
