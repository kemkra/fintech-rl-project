from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROCESSED_DATA_FILE = Path("data/processed/stock_features.csv")
FIGURES_DIR = Path("reports/figures")
RESULTS_DIR = Path("reports/results")
SUMMARY_FILE = RESULTS_DIR / "eda_summary.csv"
DATA_QUALITY_FILE = RESULTS_DIR / "data_quality_summary.csv"


def create_dirs():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_feature_data(file_path=None):
    file_path = Path(file_path or PROCESSED_DATA_FILE)
    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path} not found. Run src/features/feature_engineering.py first."
        )

    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def save_data_quality_summary(df, output_path=None):
    output_path = Path(output_path or DATA_QUALITY_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = (
        df.groupby("Ticker")
        .agg(
            rows=("Date", "size"),
            start_date=("Date", "min"),
            end_date=("Date", "max"),
            missing_values=("Close", lambda series: int(series.isna().sum())),
        )
        .reset_index()
    )
    summary["start_date"] = summary["start_date"].dt.date
    summary["end_date"] = summary["end_date"].dt.date
    summary.to_csv(output_path, index=False)
    return summary


def calculate_eda_summary(df, output_path=None):
    output_path = Path(output_path or SUMMARY_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = (
        df.groupby("Ticker")
        .agg(
            rows=("Date", "size"),
            first_close=("Close", "first"),
            last_close=("Close", "last"),
            mean_daily_return=("Daily_Return", "mean"),
            daily_volatility=("Daily_Return", "std"),
            average_volume=("Volume", "mean"),
            average_rsi=("RSI", "mean"),
        )
        .reset_index()
    )

    summary["total_return"] = summary["last_close"] / summary["first_close"] - 1
    summary["annualized_return"] = (1 + summary["total_return"]) ** (252 / summary["rows"]) - 1
    summary["annualized_volatility"] = summary["daily_volatility"] * (252 ** 0.5)
    summary = summary.sort_values("total_return", ascending=False)
    summary.to_csv(output_path, index=False)
    return summary


def pivot_by_ticker(df, value_column):
    return df.pivot(index="Date", columns="Ticker", values=value_column).sort_index()


def plot_close_prices(df):
    close_prices = pivot_by_ticker(df, "Close")
    ax = close_prices.plot(figsize=(14, 7), linewidth=1.4)
    ax.set_title("Close Price Trends")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Close Price")
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "close_price_trends.png", dpi=160)
    plt.close()


def plot_cumulative_returns(df):
    returns = pivot_by_ticker(df, "Daily_Return")
    cumulative_returns = (1 + returns).cumprod() - 1

    ax = cumulative_returns.plot(figsize=(14, 7), linewidth=1.4)
    ax.set_title("Cumulative Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cumulative_returns.png", dpi=160)
    plt.close()


def plot_daily_return_distribution(df):
    plt.figure(figsize=(14, 7))
    sns.histplot(
        data=df,
        x="Daily_Return",
        hue="Ticker",
        bins=80,
        element="step",
        stat="density",
        common_norm=False,
        alpha=0.25,
    )
    plt.title("Daily Return Distribution")
    plt.xlabel("Daily Return")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "daily_return_distribution.png", dpi=160)
    plt.close()


def plot_volatility_comparison(summary):
    plot_data = summary.sort_values("annualized_volatility", ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_data, x="Ticker", y="annualized_volatility")
    plt.title("Annualized Volatility by Asset")
    plt.xlabel("Ticker")
    plt.ylabel("Annualized Volatility")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "volatility_comparison.png", dpi=160)
    plt.close()


def plot_correlation_heatmap(df):
    returns = pivot_by_ticker(df, "Daily_Return")
    corr = returns.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", square=True)
    plt.title("Daily Return Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()


def plot_technical_indicators(df, ticker="AAPL"):
    ticker_df = df[df["Ticker"] == ticker].copy()
    if ticker_df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(ticker_df["Date"], ticker_df["Close"], label="Close", linewidth=1.3)
    axes[0].plot(ticker_df["Date"], ticker_df["MA5"], label="MA5", linewidth=1.0)
    axes[0].plot(ticker_df["Date"], ticker_df["MA20"], label="MA20", linewidth=1.0)
    axes[0].set_title(f"{ticker} Price and Moving Averages")
    axes[0].set_ylabel("Price")
    axes[0].legend()

    axes[1].plot(ticker_df["Date"], ticker_df["RSI"], color="tab:orange", linewidth=1.0)
    axes[1].axhline(70, color="red", linestyle="--", linewidth=0.8)
    axes[1].axhline(30, color="green", linestyle="--", linewidth=0.8)
    axes[1].set_title("RSI")
    axes[1].set_ylabel("RSI")

    axes[2].plot(ticker_df["Date"], ticker_df["MACD"], color="tab:purple", linewidth=1.0)
    axes[2].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[2].set_title("MACD")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("MACD")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"technical_indicators_{ticker}.png", dpi=160)
    plt.close()


def run_eda(example_ticker="AAPL"):
    create_dirs()
    df = load_feature_data()
    quality_summary = save_data_quality_summary(df)
    eda_summary = calculate_eda_summary(df)

    sns.set_theme(style="whitegrid")
    plot_close_prices(df)
    plot_cumulative_returns(df)
    plot_daily_return_distribution(df)
    plot_volatility_comparison(eda_summary)
    plot_correlation_heatmap(df)
    plot_technical_indicators(df, ticker=example_ticker)

    print(f"Data quality summary saved to {DATA_QUALITY_FILE}")
    print(f"EDA summary saved to {SUMMARY_FILE}")
    print(f"Figures saved to {FIGURES_DIR}")
    print(f"Tickers analyzed: {df['Ticker'].nunique()}")
    return quality_summary, eda_summary


if __name__ == "__main__":
    run_eda()
