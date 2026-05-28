from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RESULTS_DIR = Path("reports/results")
STRATEGY_COMPARISON_FILE = RESULTS_DIR / "strategy_comparison.csv"

DEFAULT_METRICS_FILES = {
    "BuyHold": RESULTS_DIR / "buy_hold_metrics.csv",
    "MovingAverage": RESULTS_DIR / "ma_metrics.csv",
    "RSI": RESULTS_DIR / "rsi_metrics.csv",
}

COMPARISON_COLUMNS = [
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
    "rank_total_return",
    "rank_sharpe",
    "rank_max_drawdown",
]

NUMERIC_COLUMNS = [
    "start_value",
    "end_value",
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "win_rate",
]


def load_strategy_metric_files(metrics_files=None):
    metrics_files = metrics_files or DEFAULT_METRICS_FILES
    frames = []

    for strategy_name, file_path in metrics_files.items():
        file_path = Path(file_path)
        if not file_path.exists():
            continue

        df = pd.read_csv(file_path)
        if df.empty:
            continue

        if "Strategy" not in df.columns:
            df["Strategy"] = strategy_name
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)

    return pd.concat(frames, axis=0, ignore_index=True)


def build_strategy_comparison(metrics_files=None, output_file=STRATEGY_COMPARISON_FILE):
    comparison = load_strategy_metric_files(metrics_files)
    if comparison.empty:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)

    for column in COMPARISON_COLUMNS:
        if column not in comparison.columns:
            comparison[column] = pd.NA

    for column in NUMERIC_COLUMNS:
        comparison[column] = pd.to_numeric(comparison[column], errors="coerce")

    comparison["rank_total_return"] = (
        comparison.groupby("Ticker")["total_return"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    comparison["rank_sharpe"] = (
        comparison.groupby("Ticker")["sharpe_ratio"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    comparison["rank_max_drawdown"] = (
        comparison.groupby("Ticker")["max_drawdown"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )

    comparison = comparison[COMPARISON_COLUMNS].sort_values(
        ["Ticker", "rank_total_return", "Strategy"],
        ascending=[True, True, True],
    )

    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(output_file, index=False)

    return comparison


if __name__ == "__main__":
    result = build_strategy_comparison()
    print(f"Strategy comparison saved to {STRATEGY_COMPARISON_FILE}")
    print(f"Rows compared: {len(result)}")
