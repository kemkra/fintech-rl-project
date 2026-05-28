from datetime import datetime
import json
from pathlib import Path
import shutil
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from src.storage import runtime_store


CONFIG_DIR = Path("config")
ACTIVE_ANALYSIS_DATASET_FILE = CONFIG_DIR / "active_analysis_dataset.json"
REFERENCE_DATA_DIR = Path("data/reference")
US_SYMBOLS_FILE = REFERENCE_DATA_DIR / "us_stock_symbols.csv"
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
EXCHANGE_CODE_NAMES = {
    "A": "NYSE American",
    "N": "NYSE",
    "P": "NYSE Arca",
    "Z": "Cboe BZX",
    "V": "Investors Exchange",
}
PROCESSED_DATA_FILE = Path("data/processed/stock_features.csv")
RAW_DATA_DIR = Path("data/raw")
CHAT_WORKSPACES_DIR = Path("data/workspaces/chats")
DEFAULT_CHAT_WORKSPACE_ID = "default"
LEGACY_LLM_WORKSPACE_DIR = Path("data/llm_workspace")
LLM_WORKSPACE_DIR = CHAT_WORKSPACES_DIR / DEFAULT_CHAT_WORKSPACE_ID
EDA_SUMMARY_FILE = Path("reports/results/eda_summary.csv")
DATA_QUALITY_FILE = Path("reports/results/data_quality_summary.csv")
BUY_HOLD_EQUITY_FILE = Path("reports/results/buy_hold_equity_curves.csv")
BUY_HOLD_METRICS_FILE = Path("reports/results/buy_hold_metrics.csv")
MA_EQUITY_FILE = Path("reports/results/ma_equity_curves.csv")
MA_METRICS_FILE = Path("reports/results/ma_metrics.csv")
RSI_EQUITY_FILE = Path("reports/results/rsi_equity_curves.csv")
RSI_METRICS_FILE = Path("reports/results/rsi_metrics.csv")
STRATEGY_COMPARISON_FILE = Path("reports/results/strategy_comparison.csv")
FIGURES_DIR = Path("reports/figures")
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "BAC", "GS", "SPY", "QQQ"]
SCREENER_SEED_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "LLY", "JPM",
    "V", "UNH", "XOM", "MA", "COST", "HD", "NFLX", "WMT", "PG", "JNJ",
    "BAC", "AMD", "CRM", "ORCL", "ADBE", "KO", "PEP", "CSCO", "MCD", "INTC",
    "SPY", "QQQ", "DIA", "IWM",
]
SUPPORTED_INTERVAL = "1d"
RUNTIME_SESSION_ID = None
RUNTIME_DB_FILE = None


def configure_runtime_storage(session_id=None, root=None):
    """Point generated Web artifacts at a per-session runtime directory."""
    global CONFIG_DIR, ACTIVE_ANALYSIS_DATASET_FILE, REFERENCE_DATA_DIR, US_SYMBOLS_FILE
    global PROCESSED_DATA_FILE, RAW_DATA_DIR, CHAT_WORKSPACES_DIR, LLM_WORKSPACE_DIR
    global EDA_SUMMARY_FILE, DATA_QUALITY_FILE, BUY_HOLD_EQUITY_FILE, BUY_HOLD_METRICS_FILE
    global MA_EQUITY_FILE, MA_METRICS_FILE, RSI_EQUITY_FILE, RSI_METRICS_FILE
    global STRATEGY_COMPARISON_FILE, FIGURES_DIR, RUNTIME_SESSION_ID, RUNTIME_DB_FILE

    paths = runtime_store.ensure_runtime(session_id=session_id, root=root)
    CONFIG_DIR = paths["config_dir"]
    ACTIVE_ANALYSIS_DATASET_FILE = CONFIG_DIR / "active_analysis_dataset.json"
    REFERENCE_DATA_DIR = paths["reference_dir"]
    US_SYMBOLS_FILE = REFERENCE_DATA_DIR / "us_stock_symbols.csv"
    PROCESSED_DATA_FILE = paths["processed_dir"] / "stock_features.csv"
    RAW_DATA_DIR = paths["raw_dir"]
    CHAT_WORKSPACES_DIR = paths["chat_workspaces_dir"]
    LLM_WORKSPACE_DIR = CHAT_WORKSPACES_DIR / DEFAULT_CHAT_WORKSPACE_ID
    EDA_SUMMARY_FILE = paths["results_dir"] / "eda_summary.csv"
    DATA_QUALITY_FILE = paths["results_dir"] / "data_quality_summary.csv"
    BUY_HOLD_EQUITY_FILE = paths["results_dir"] / "buy_hold_equity_curves.csv"
    BUY_HOLD_METRICS_FILE = paths["results_dir"] / "buy_hold_metrics.csv"
    MA_EQUITY_FILE = paths["results_dir"] / "ma_equity_curves.csv"
    MA_METRICS_FILE = paths["results_dir"] / "ma_metrics.csv"
    RSI_EQUITY_FILE = paths["results_dir"] / "rsi_equity_curves.csv"
    RSI_METRICS_FILE = paths["results_dir"] / "rsi_metrics.csv"
    STRATEGY_COMPARISON_FILE = paths["results_dir"] / "strategy_comparison.csv"
    FIGURES_DIR = paths["figures_dir"]
    RUNTIME_SESSION_ID = paths["session_id"]
    RUNTIME_DB_FILE = paths["db_file"]
    return {key: str(value) if isinstance(value, Path) else value for key, value in paths.items()}


def get_runtime_storage_status():
    if not RUNTIME_DB_FILE:
        return {
            "enabled": False,
            "message": "Runtime storage has not been configured.",
        }

    status = runtime_store.get_runtime_status(RUNTIME_DB_FILE, RUNTIME_SESSION_ID)
    status["enabled"] = True
    status["raw_dir"] = str(RAW_DATA_DIR)
    status["processed_file"] = str(PROCESSED_DATA_FILE)
    status["chat_workspaces_dir"] = str(CHAT_WORKSPACES_DIR)
    return status


def _save_runtime_dataset(dataset_id, raw_df, feature_df=None, scope="project"):
    if not RUNTIME_DB_FILE:
        return None
    runtime_store.save_runtime_dataset(
        db_file=RUNTIME_DB_FILE,
        session_id=RUNTIME_SESSION_ID,
        dataset_id=dataset_id,
        raw_df=raw_df,
        feature_df=feature_df,
        scope=scope,
        interval=SUPPORTED_INTERVAL,
    )
    return str(RUNTIME_DB_FILE)


def _raise_no_usable_data(source_rows):
    details = []
    for row in source_rows:
        ticker = row.get("ticker", "UNKNOWN")
        source = row.get("source", "unknown")
        error = row.get("error")
        status = row.get("status") or {}
        reason = error or status.get("reason") or "No data returned."
        details.append(f"{ticker} via {source}: {reason}")

    message = "No usable raw data found or downloaded for the requested tickers/date range."
    if details:
        message = f"{message} Details: {'; '.join(details)}"
    raise ValueError(message)


def normalize_chat_id(chat_id=None):
    text = str(chat_id or DEFAULT_CHAT_WORKSPACE_ID).strip()
    safe = "".join(char for char in text if char.isalnum() or char in {"-", "_"})
    return safe or DEFAULT_CHAT_WORKSPACE_ID


def get_chat_workspace_paths(chat_id=None):
    chat_id = normalize_chat_id(chat_id)
    workspace_dir = CHAT_WORKSPACES_DIR / chat_id
    processed_dir = workspace_dir / "processed"
    figures_dir = workspace_dir / "figures"
    results_dir = workspace_dir / "results"
    return {
        "chat_id": chat_id,
        "workspace_dir": workspace_dir,
        "processed_dir": processed_dir,
        "processed_file": processed_dir / "stock_features.csv",
        "figures_dir": figures_dir,
        "results_dir": results_dir,
        "buy_hold_equity_file": results_dir / "buy_hold_equity_curves.csv",
        "buy_hold_metrics_file": results_dir / "buy_hold_metrics.csv",
        "ma_equity_file": results_dir / "ma_equity_curves.csv",
        "ma_metrics_file": results_dir / "ma_metrics.csv",
        "rsi_equity_file": results_dir / "rsi_equity_curves.csv",
        "rsi_metrics_file": results_dir / "rsi_metrics.csv",
        "strategy_comparison_file": results_dir / "strategy_comparison.csv",
    }


def migrate_legacy_llm_workspace_if_needed(chat_id=None):
    paths = get_chat_workspace_paths(chat_id)
    if paths["chat_id"] != DEFAULT_CHAT_WORKSPACE_ID:
        return False
    if paths["processed_file"].exists() or not LEGACY_LLM_WORKSPACE_DIR.exists():
        return False

    for name in ["processed", "figures", "results"]:
        source = LEGACY_LLM_WORKSPACE_DIR / name
        target = paths["workspace_dir"] / name
        if source.exists() and not target.exists():
            shutil.copytree(source, target)
    return True


def normalize_ticker(ticker):
    return str(ticker).strip().upper()


def parse_ticker_input(ticker_text):
    if not ticker_text:
        return []

    separators = [",", "\n", "\t", ";"]
    normalized_text = str(ticker_text)
    for separator in separators:
        normalized_text = normalized_text.replace(separator, " ")

    tickers = []
    for token in normalized_text.split(" "):
        ticker = normalize_ticker(token)
        if ticker and ticker not in tickers:
            tickers.append(ticker)

    return tickers


def merge_ticker_lists(*ticker_lists):
    merged = []
    for ticker_list in ticker_lists:
        for ticker in ticker_list or []:
            normalized = normalize_ticker(ticker)
            if normalized and normalized not in merged:
                merged.append(normalized)
    return merged


def _read_symbol_directory_url(url):
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=20) as response:
        text = response.read().decode("utf-8", errors="replace")

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("File Creation Time"):
            continue
        rows.append(line)
    if not rows:
        return pd.DataFrame()

    header = rows[0].split("|")
    records = []
    for row in rows[1:]:
        values = row.split("|")
        if len(values) != len(header):
            continue
        records.append(dict(zip(header, values)))
    return pd.DataFrame(records)


def _to_yfinance_symbol(symbol):
    return normalize_ticker(symbol).replace(".", "-").replace("/", "-")


def _normalize_symbol_directory(nasdaq_df, other_df):
    frames = []

    if not nasdaq_df.empty:
        nasdaq = pd.DataFrame(
            {
                "symbol": nasdaq_df.get("Symbol", pd.Series(dtype=str)),
                "name": nasdaq_df.get("Security Name", pd.Series(dtype=str)),
                "exchange": "NASDAQ",
                "etf": nasdaq_df.get("ETF", pd.Series(dtype=str)),
                "test_issue": nasdaq_df.get("Test Issue", pd.Series(dtype=str)),
            }
        )
        frames.append(nasdaq)

    if not other_df.empty:
        other = pd.DataFrame(
            {
                "symbol": other_df.get("ACT Symbol", pd.Series(dtype=str)),
                "name": other_df.get("Security Name", pd.Series(dtype=str)),
                "exchange": other_df.get("Exchange", pd.Series(dtype=str)),
                "etf": other_df.get("ETF", pd.Series(dtype=str)),
                "test_issue": other_df.get("Test Issue", pd.Series(dtype=str)),
            }
        )
        frames.append(other)

    if not frames:
        return pd.DataFrame(columns=["symbol", "yfinance_symbol", "name", "exchange", "etf"])

    symbols = pd.concat(frames, axis=0, ignore_index=True)
    symbols["symbol"] = symbols["symbol"].astype(str).str.strip().str.upper()
    symbols["name"] = symbols["name"].astype(str).str.strip()
    symbols["exchange"] = symbols["exchange"].astype(str).str.strip()
    symbols["exchange"] = symbols["exchange"].replace(EXCHANGE_CODE_NAMES)
    symbols["etf"] = symbols["etf"].astype(str).str.strip()
    symbols["test_issue"] = symbols["test_issue"].astype(str).str.strip()
    symbols = symbols[(symbols["symbol"] != "") & (symbols["test_issue"] != "Y")]
    symbols = symbols[~symbols["symbol"].str.contains(r"[$]", regex=True, na=False)]
    symbols["yfinance_symbol"] = symbols["symbol"].map(_to_yfinance_symbol)
    symbols = symbols.drop_duplicates(subset=["yfinance_symbol"]).sort_values("yfinance_symbol")
    return symbols[["symbol", "yfinance_symbol", "name", "exchange", "etf"]].reset_index(drop=True)


def refresh_us_symbol_universe():
    REFERENCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    nasdaq_df = _read_symbol_directory_url(NASDAQ_LISTED_URL)
    other_df = _read_symbol_directory_url(OTHER_LISTED_URL)
    symbols = _normalize_symbol_directory(nasdaq_df, other_df)
    if symbols.empty:
        raise ValueError("No symbols were loaded from Nasdaq Trader symbol directories.")
    symbols.to_csv(US_SYMBOLS_FILE, index=False)
    return {
        "available": True,
        "source": "Nasdaq Trader Symbol Directory",
        "path": str(US_SYMBOLS_FILE),
        "rows": int(len(symbols)),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def load_us_symbol_universe(force_refresh=False):
    if force_refresh or not US_SYMBOLS_FILE.exists():
        try:
            refresh_us_symbol_universe()
        except Exception as exc:
            fallback = pd.DataFrame(
                {
                    "symbol": DEFAULT_TICKERS,
                    "yfinance_symbol": DEFAULT_TICKERS,
                    "name": DEFAULT_TICKERS,
                    "exchange": "fallback",
                    "etf": "",
                }
            )
            return fallback, {
                "available": False,
                "source": "fallback",
                "path": str(US_SYMBOLS_FILE),
                "rows": int(len(fallback)),
                "error": str(exc),
            }

    symbols = pd.read_csv(US_SYMBOLS_FILE)
    return symbols, {
        "available": True,
        "source": "Nasdaq Trader Symbol Directory cache",
        "path": str(US_SYMBOLS_FILE),
        "rows": int(len(symbols)),
    }


def search_us_symbols(query="", limit=500, force_refresh=False):
    symbols, status = load_us_symbol_universe(force_refresh=force_refresh)
    if symbols.empty:
        return {**status, "records": []}

    query = str(query or "").strip().upper()
    results = symbols.copy()
    if query:
        mask = (
            results["yfinance_symbol"].astype(str).str.upper().str.contains(query, na=False)
            | results["symbol"].astype(str).str.upper().str.contains(query, na=False)
            | results["name"].astype(str).str.upper().str.contains(query, na=False)
        )
        results = results[mask]

    results = results.sort_values("yfinance_symbol").head(int(limit))
    return {
        **status,
        "query": query,
        "returned": int(len(results)),
        "records": _json_records(results),
    }


def _select_screener_universe(symbols, max_candidates, query="", include_etfs=False, exchanges=None):
    universe = symbols.copy()
    if not include_etfs and "etf" in universe.columns:
        universe = universe[universe["etf"].astype(str).str.upper() != "Y"]

    if exchanges:
        wanted = {str(exchange).strip().upper() for exchange in exchanges}
        universe = universe[universe["exchange"].astype(str).str.upper().isin(wanted)]

    query = str(query or "").strip().upper()
    if query:
        mask = (
            universe["yfinance_symbol"].astype(str).str.upper().str.contains(query, na=False)
            | universe["symbol"].astype(str).str.upper().str.contains(query, na=False)
            | universe["name"].astype(str).str.upper().str.contains(query, na=False)
        )
        universe = universe[mask]

    universe = universe.drop_duplicates(subset=["yfinance_symbol"]).reset_index(drop=True)
    if universe.empty:
        return universe

    max_candidates = max(1, int(max_candidates))
    seed_rows = universe[universe["yfinance_symbol"].isin(SCREENER_SEED_TICKERS)]
    remaining = universe[~universe["yfinance_symbol"].isin(seed_rows["yfinance_symbol"])]
    slots = max(max_candidates - len(seed_rows), 0)

    if slots and not remaining.empty:
        if len(remaining) <= slots:
            sampled = remaining
        else:
            indices = np.linspace(0, len(remaining) - 1, num=slots, dtype=int)
            sampled = remaining.iloc[indices]
        selected = pd.concat([seed_rows, sampled], axis=0, ignore_index=True)
    else:
        selected = seed_rows

    if selected.empty:
        selected = universe.head(max_candidates)
    return selected.drop_duplicates(subset=["yfinance_symbol"]).head(max_candidates).reset_index(drop=True)


def _download_screening_history(tickers, period, chunk_size=80):
    frames = []
    for index in range(0, len(tickers), chunk_size):
        chunk = tickers[index:index + chunk_size]
        try:
            history = yf.download(
                tickers=chunk,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="ticker",
                timeout=30,
            )
        except Exception:
            continue

        if history.empty:
            continue

        for ticker in chunk:
            try:
                if isinstance(history.columns, pd.MultiIndex):
                    ticker_df = history[ticker].copy()
                else:
                    ticker_df = history.copy()
                if ticker_df.empty or "Close" not in ticker_df.columns:
                    continue
                ticker_df = ticker_df.reset_index()
                ticker_df["Ticker"] = ticker
                frames.append(ticker_df)
            except Exception:
                continue

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


def _period_to_start_timestamp(period):
    period = str(period or "1y").strip().lower()
    number = "".join(char for char in period if char.isdigit())
    unit = "".join(char for char in period if char.isalpha())
    value = int(number or 1)
    today = pd.Timestamp.today().normalize()
    if unit in {"d", "day", "days"}:
        return today - pd.DateOffset(days=value)
    if unit in {"wk", "w", "week", "weeks"}:
        return today - pd.DateOffset(weeks=value)
    if unit in {"mo", "m", "month", "months"}:
        return today - pd.DateOffset(months=value)
    return today - pd.DateOffset(years=value)


def _period_to_trading_rows(period):
    period = str(period or "1y").strip().lower()
    number = "".join(char for char in period if char.isdigit())
    unit = "".join(char for char in period if char.isalpha())
    value = int(number or 1)
    if unit in {"d", "day", "days"}:
        return max(value, 40)
    if unit in {"wk", "w", "week", "weeks"}:
        return max(value * 5, 40)
    if unit in {"mo", "m", "month", "months"}:
        return max(value * 21, 40)
    return max(value * 252, 40)


def _load_local_screening_history(tickers, period):
    start_ts = _period_to_start_timestamp(period)
    fallback_rows = _period_to_trading_rows(period)
    frames = []
    used_tickers = []
    for ticker in tickers:
        status = inspect_local_raw_ticker(ticker)
        if not status["available"] and not Path(status["file_path"]).exists():
            continue
        try:
            raw_df = load_local_raw_ticker(ticker)
        except Exception:
            continue
        raw_df = raw_df[raw_df["Date"] >= start_ts].copy()
        if len(raw_df) < 40:
            raw_df = load_local_raw_ticker(ticker).tail(fallback_rows).copy()
        if raw_df.empty:
            continue
        frames.append(raw_df)
        used_tickers.append(ticker)

    if not frames:
        return pd.DataFrame(), []
    return pd.concat(frames, axis=0, ignore_index=True), used_tickers


def _calculate_screening_metrics(history_df, symbol_lookup):
    records = []
    for ticker, ticker_df in history_df.groupby("Ticker"):
        ticker_df = ticker_df.sort_values("Date").copy()
        close = pd.to_numeric(ticker_df["Close"], errors="coerce").dropna()
        if len(close) < 40:
            continue

        volume = pd.to_numeric(ticker_df["Volume"], errors="coerce") if "Volume" in ticker_df.columns else pd.Series(dtype=float)
        daily_returns = close.pct_change().dropna()
        if daily_returns.empty:
            continue

        total_return = close.iloc[-1] / close.iloc[0] - 1
        periods = max(len(close) - 1, 1)
        annualized_return = (1 + total_return) ** (252 / periods) - 1
        annualized_volatility = daily_returns.std() * np.sqrt(252)
        risk_adjusted_return = None
        if annualized_volatility and not pd.isna(annualized_volatility):
            risk_adjusted_return = annualized_return / annualized_volatility

        running_max = close.cummax()
        max_drawdown = (close / running_max - 1).min()
        avg_volume = volume.tail(60).mean() if not volume.empty else np.nan
        symbol_row = symbol_lookup.get(ticker, {})
        records.append(
            {
                "Ticker": ticker,
                "Name": symbol_row.get("name"),
                "Exchange": symbol_row.get("exchange"),
                "Rows": int(len(close)),
                "Start_Date": pd.to_datetime(ticker_df["Date"].iloc[0]).date().isoformat(),
                "End_Date": pd.to_datetime(ticker_df["Date"].iloc[-1]).date().isoformat(),
                "Last_Close": float(close.iloc[-1]),
                "Total_Return": float(total_return),
                "Annualized_Return": float(annualized_return),
                "Annualized_Volatility": float(annualized_volatility) if not pd.isna(annualized_volatility) else None,
                "Risk_Adjusted_Return": None if risk_adjusted_return is None or pd.isna(risk_adjusted_return) else float(risk_adjusted_return),
                "Max_Drawdown": float(max_drawdown),
                "Average_Volume_60D": None if pd.isna(avg_volume) else float(avg_volume),
            }
        )

    return pd.DataFrame(records)


def screen_stock_candidates(
    query="",
    max_candidates=300,
    shortlist_size=10,
    lookback_period="1y",
    min_avg_volume=500_000,
    min_price=5,
    include_etfs=False,
    exchanges=None,
    sort_by="risk_adjusted_return",
    chat_id=None,
):
    paths = get_chat_workspace_paths(chat_id)
    symbols, symbol_status = load_us_symbol_universe(force_refresh=False)
    selected = _select_screener_universe(
        symbols=symbols,
        max_candidates=max_candidates,
        query=query,
        include_etfs=include_etfs,
        exchanges=exchanges,
    )
    tickers = selected["yfinance_symbol"].tolist()
    if not tickers:
        return {
            "available": False,
            "chat_id": paths["chat_id"],
            "message": "No candidate symbols matched the requested universe.",
            "records": [],
            "shortlist": [],
        }

    local_history, local_tickers = _load_local_screening_history(tickers=tickers, period=lookback_period)
    missing_tickers = [ticker for ticker in tickers if ticker not in set(local_tickers)]
    downloaded_history = _download_screening_history(tickers=missing_tickers, period=lookback_period) if missing_tickers else pd.DataFrame()
    history_frames = [frame for frame in [local_history, downloaded_history] if not frame.empty]
    history = pd.concat(history_frames, axis=0, ignore_index=True) if history_frames else pd.DataFrame()
    if history.empty:
        return {
            "available": False,
            "chat_id": paths["chat_id"],
            "message": "No yfinance price history was returned for the candidate universe.",
            "candidate_count": len(tickers),
            "local_history_count": int(len(local_tickers)),
            "records": [],
            "shortlist": [],
        }

    symbol_lookup = selected.set_index("yfinance_symbol").to_dict(orient="index")
    metrics = _calculate_screening_metrics(history, symbol_lookup)
    if metrics.empty:
        return {
            "available": False,
            "chat_id": paths["chat_id"],
            "message": "Candidate price history was insufficient for screening metrics.",
            "candidate_count": len(tickers),
            "local_history_count": int(len(local_tickers)),
            "records": [],
            "shortlist": [],
        }

    if min_avg_volume is not None:
        metrics = metrics[metrics["Average_Volume_60D"].fillna(0) >= float(min_avg_volume)]
    if min_price is not None:
        metrics = metrics[metrics["Last_Close"] >= float(min_price)]

    sort_columns = {
        "risk_adjusted_return": "Risk_Adjusted_Return",
        "total_return": "Total_Return",
        "annualized_return": "Annualized_Return",
        "low_drawdown": "Max_Drawdown",
        "volume": "Average_Volume_60D",
    }
    sort_column = sort_columns.get(str(sort_by).lower(), "Risk_Adjusted_Return")
    ascending = sort_column == "Max_Drawdown"
    metrics = metrics.sort_values(sort_column, ascending=ascending, na_position="last").reset_index(drop=True)
    metrics["Rank"] = range(1, len(metrics) + 1)
    output_columns = ["Rank"] + [column for column in metrics.columns if column != "Rank"]
    metrics = metrics[output_columns]

    paths["results_dir"].mkdir(parents=True, exist_ok=True)
    output_file = paths["results_dir"] / "stock_screen_candidates.csv"
    metrics.to_csv(output_file, index=False)

    shortlist = metrics.head(int(shortlist_size))
    return {
        "available": not shortlist.empty,
        "chat_id": paths["chat_id"],
        "universe_source": symbol_status.get("source"),
        "candidate_count": int(len(tickers)),
        "local_history_count": int(len(local_tickers)),
        "download_attempt_count": int(len(missing_tickers)),
        "screened_count": int(len(metrics)),
        "shortlist_size": int(len(shortlist)),
        "lookback_period": lookback_period,
        "sort_by": sort_by,
        "results_file": str(output_file),
        "shortlist_tickers": shortlist["Ticker"].tolist(),
        "records": _json_records(shortlist),
        "message": "Historical screening only. Treat results as candidates for further research, not investment advice.",
    }


def load_shortlist_for_analysis(
    tickers,
    start_date,
    end_date,
    use_proxy=True,
    data_source="auto",
    run_baseline_after=True,
    chat_id=None,
):
    tickers = merge_ticker_lists(tickers)
    if not tickers:
        raise ValueError("tickers cannot be empty.")
    result = refresh_llm_workspace_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        use_proxy=use_proxy,
        run_baseline_after=run_baseline_after,
        data_source=data_source,
        chat_id=chat_id,
    )
    result["loaded_shortlist"] = tickers
    return result


def validate_ticker_candidates(query, candidates, validation_mode="local_first"):
    status = get_dataset_status()
    available_tickers = status.get("tickers", [])
    results = []

    for candidate in merge_ticker_lists(candidates):
        in_dataset = candidate in available_tickers
        valid_yfinance = False
        long_name = None
        quote_type = None
        currency = None
        error = None

        if validation_mode == "local_only":
            valid_yfinance = False
        elif validation_mode == "local_first" and in_dataset:
            valid_yfinance = True
        else:
            try:
                ticker_obj = yf.Ticker(candidate)
                fast_info = ticker_obj.fast_info
                history = ticker_obj.history(period="5d", interval="1d", auto_adjust=True)
                valid_yfinance = not history.empty or bool(fast_info)
                info = {}
                if valid_yfinance:
                    try:
                        info = ticker_obj.get_info()
                    except Exception:
                        info = {}
                long_name = info.get("longName") or info.get("shortName")
                quote_type = info.get("quoteType")
                currency = info.get("currency")
            except Exception as exc:
                error = str(exc)

        results.append(
            {
                "candidate": candidate,
                "valid_yfinance": bool(valid_yfinance),
                "in_current_dataset": bool(in_dataset),
                "long_name": long_name,
                "quote_type": quote_type,
                "currency": currency,
                "error": error,
            }
        )

    selected = None
    for result in results:
        if result["in_current_dataset"]:
            selected = result["candidate"]
            break
    if selected is None:
        for result in results:
            if result["valid_yfinance"]:
                selected = result["candidate"]
                break

    return {
        "query": query,
        "selected_ticker": selected,
        "validation_mode": validation_mode,
        "validated_candidates": results,
        "available_tickers": available_tickers,
    }


def _normalize_raw_columns(df, ticker):
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]
    if "Date" not in df.columns:
        df = df.reset_index()
        df.columns = [str(column).strip() for column in df.columns]

    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{ticker} raw file missing columns: {sorted(missing)}")

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["Date", "Close"])
    df["Ticker"] = ticker
    return df.sort_values("Date").reset_index(drop=True)


def inspect_raw_ticker_in_dir(raw_data_dir, ticker, start_date=None, end_date=None, interval=SUPPORTED_INTERVAL):
    ticker = normalize_ticker(ticker)
    raw_data_dir = Path(raw_data_dir)
    file_path = raw_data_dir / f"{ticker}.csv"
    if interval != SUPPORTED_INTERVAL:
        return {
            "available": False,
            "ticker": ticker,
            "file_path": str(file_path),
            "reason": f"Only {SUPPORTED_INTERVAL} raw interval is supported locally.",
        }
    if not file_path.exists():
        return {
            "available": False,
            "ticker": ticker,
            "file_path": str(file_path),
            "reason": "Local raw file does not exist.",
        }

    try:
        raw_df = _normalize_raw_columns(pd.read_csv(file_path), ticker)
    except Exception as exc:
        return {
            "available": False,
            "ticker": ticker,
            "file_path": str(file_path),
            "reason": str(exc),
        }

    if raw_df.empty:
        return {
            "available": False,
            "ticker": ticker,
            "file_path": str(file_path),
            "reason": "Local raw file has no usable rows.",
        }

    requested_start = pd.to_datetime(start_date) if start_date else raw_df["Date"].min()
    requested_end = pd.to_datetime(end_date) if end_date else raw_df["Date"].max()
    tolerance = pd.Timedelta(days=7)
    covers_range = (
        raw_df["Date"].min() <= requested_start + tolerance
        and raw_df["Date"].max() >= requested_end - tolerance
    )

    max_gap_days = raw_df["Date"].sort_values().diff().dt.days.max()
    interval_ok = pd.isna(max_gap_days) or max_gap_days <= 7
    usable = bool(covers_range and interval_ok)

    return {
        "available": usable,
        "ticker": ticker,
        "file_path": str(file_path),
        "rows": int(len(raw_df)),
        "start_date": raw_df["Date"].min().date().isoformat(),
        "end_date": raw_df["Date"].max().date().isoformat(),
        "requested_start": requested_start.date().isoformat(),
        "requested_end": requested_end.date().isoformat(),
        "covers_range": bool(covers_range),
        "interval_ok": bool(interval_ok),
        "max_gap_days": None if pd.isna(max_gap_days) else int(max_gap_days),
        "reason": None if usable else "Local raw file does not fully match requested date range or interval.",
    }


def inspect_local_raw_ticker(ticker, start_date=None, end_date=None, interval=SUPPORTED_INTERVAL):
    return inspect_raw_ticker_in_dir(
        raw_data_dir=RAW_DATA_DIR,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
    )


def inspect_llm_workspace_raw_ticker(ticker, start_date=None, end_date=None, interval=SUPPORTED_INTERVAL):
    return inspect_raw_ticker_in_dir(
        raw_data_dir=RAW_DATA_DIR,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
    )


def load_local_raw_ticker(ticker, start_date=None, end_date=None):
    return load_raw_ticker_from_dir(RAW_DATA_DIR, ticker, start_date=start_date, end_date=end_date)


def load_raw_ticker_from_dir(raw_data_dir, ticker, start_date=None, end_date=None):
    ticker = normalize_ticker(ticker)
    file_path = Path(raw_data_dir) / f"{ticker}.csv"
    raw_df = _normalize_raw_columns(pd.read_csv(file_path), ticker)
    if start_date:
        raw_df = raw_df[raw_df["Date"] >= pd.to_datetime(start_date)]
    if end_date:
        raw_df = raw_df[raw_df["Date"] <= pd.to_datetime(end_date)]
    return raw_df.reset_index(drop=True)


def load_llm_workspace_raw_ticker(ticker, start_date=None, end_date=None):
    return load_raw_ticker_from_dir(RAW_DATA_DIR, ticker, start_date=start_date, end_date=end_date)


def _read_processed_data(data_file=None):
    data_file = Path(data_file or PROCESSED_DATA_FILE)
    if not data_file.exists():
        return pd.DataFrame()

    df = pd.read_csv(data_file)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def _read_workspace_processed_data(chat_id=None):
    migrate_legacy_llm_workspace_if_needed(chat_id)
    return _read_processed_data(get_chat_workspace_paths(chat_id)["processed_file"])


def _describe_processed_file(data_file):
    data_file = Path(data_file)
    df = _read_processed_data(data_file)
    if df.empty:
        return {
            "available": False,
            "message": f"{data_file} not found or empty",
            "rows": 0,
            "tickers": [],
            "processed_file": str(data_file),
        }

    return {
        "available": True,
        "rows": int(len(df)),
        "ticker_count": int(df["Ticker"].nunique()),
        "tickers": sorted(df["Ticker"].unique().tolist()),
        "start_date": df["Date"].min().date().isoformat(),
        "end_date": df["Date"].max().date().isoformat(),
        "columns": df.columns.tolist(),
        "processed_file": str(data_file),
    }


def _read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _json_records(df, limit=None):
    if limit is not None:
        df = df.head(limit)

    output = df.copy()
    for column in output.columns:
        if pd.api.types.is_datetime64_any_dtype(output[column]):
            output[column] = output[column].dt.strftime("%Y-%m-%d")
    return output.to_dict(orient="records")


def get_dataset_status():
    return _describe_processed_file(PROCESSED_DATA_FILE)


def _load_active_analysis_dataset_config():
    if not ACTIVE_ANALYSIS_DATASET_FILE.exists():
        return {
            "source": "project",
            "note": "Default main project dataset.",
            "updated_at": None,
            "pushed_by_chat_id": None,
        }

    try:
        with ACTIVE_ANALYSIS_DATASET_FILE.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except Exception:
        config = {}

    source = config.get("source", "project")
    if source not in {"project", "llm_workspace", "chat_workspace"}:
        source = "project"

    return {
        "source": source,
        "note": config.get("note"),
        "updated_at": config.get("updated_at"),
        "pushed_by_chat_id": config.get("pushed_by_chat_id"),
        "chat_id": config.get("chat_id") or config.get("pushed_by_chat_id"),
    }


def _save_active_analysis_dataset_config(source, note=None, chat_id=None):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "source": source,
        "note": note,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "pushed_by_chat_id": chat_id,
        "chat_id": normalize_chat_id(chat_id) if chat_id else None,
    }
    with ACTIVE_ANALYSIS_DATASET_FILE.open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
    return config


def get_active_analysis_dataset_status():
    config = _load_active_analysis_dataset_config()
    source = config["source"]
    chat_id = config.get("chat_id")
    if source in {"llm_workspace", "chat_workspace"}:
        processed_file = get_chat_workspace_paths(chat_id)["processed_file"]
        label = f"AI chat workspace dataset ({normalize_chat_id(chat_id)})"
    else:
        processed_file = PROCESSED_DATA_FILE
        label = "Main project dataset"
    dataset_status = _describe_processed_file(processed_file)
    return {
        "source": source,
        "label": label,
        "processed_file": str(processed_file),
        "note": config.get("note"),
        "updated_at": config.get("updated_at"),
        "pushed_by_chat_id": config.get("pushed_by_chat_id"),
        "chat_id": normalize_chat_id(chat_id) if chat_id else None,
        "dataset": dataset_status,
    }


def push_llm_workspace_to_app_pages(note=None, chat_id=None):
    workspace_status = _describe_processed_file(get_chat_workspace_paths(chat_id)["processed_file"])
    if not workspace_status["available"]:
        return {
            "pushed": False,
            "message": "No AI chat workspace processed dataset is available to push. Refresh workspace data first.",
            "workspace_status": workspace_status,
        }

    config = _save_active_analysis_dataset_config(
        source="chat_workspace",
        note=note or "Pushed from AI Assistant.",
        chat_id=chat_id,
    )
    return {
        "pushed": True,
        "message": "AI workspace dataset is now the active dataset for app pages. Main project data was not overwritten.",
        "active_dataset": get_active_analysis_dataset_status(),
        "config": config,
    }


def reset_app_pages_to_project_dataset(note=None):
    config = _save_active_analysis_dataset_config(
        source="project",
        note=note or "Reset to main project dataset.",
        chat_id=None,
    )
    return {
        "reset": True,
        "message": "App pages now use the main project dataset.",
        "active_dataset": get_active_analysis_dataset_status(),
        "config": config,
    }


def list_raw_data_in_dir(raw_data_dir):
    raw_data_dir = Path(raw_data_dir)
    records = []
    if not raw_data_dir.exists():
        return {
            "available": False,
            "directory": str(raw_data_dir),
            "ticker_count": 0,
            "tickers": [],
            "records": records,
            "message": "Raw data directory does not exist.",
        }

    for path in sorted(raw_data_dir.glob("*.csv")):
        if not path.stem.isupper():
            continue

        ticker = normalize_ticker(path.stem)
        try:
            raw_df = _normalize_raw_columns(pd.read_csv(path), ticker)
        except Exception as exc:
            records.append(
                {
                    "ticker": ticker,
                    "file_path": str(path),
                    "valid": False,
                    "rows": 0,
                    "start_date": None,
                    "end_date": None,
                    "error": str(exc),
                }
            )
            continue

        records.append(
            {
                "ticker": ticker,
                "file_path": str(path),
                "valid": not raw_df.empty,
                "rows": int(len(raw_df)),
                "start_date": None if raw_df.empty else raw_df["Date"].min().date().isoformat(),
                "end_date": None if raw_df.empty else raw_df["Date"].max().date().isoformat(),
                "error": None,
            }
        )

    valid_tickers = [record["ticker"] for record in records if record["valid"]]
    return {
        "available": bool(records),
        "directory": str(raw_data_dir),
        "ticker_count": len(valid_tickers),
        "tickers": valid_tickers,
        "records": records,
    }


def list_local_raw_data():
    return list_raw_data_in_dir(RAW_DATA_DIR)


def list_llm_workspace_raw_data():
    return list_local_raw_data()


def get_llm_workspace_status(chat_id=None):
    paths = get_chat_workspace_paths(chat_id)
    raw_status = list_local_raw_data()
    processed_df = _read_workspace_processed_data(chat_id)
    if processed_df.empty:
        processed_status = {
            "available": False,
            "message": f"{paths['processed_file']} not found",
            "rows": 0,
            "tickers": [],
        }
    else:
        processed_status = {
            "available": True,
            "rows": int(len(processed_df)),
            "ticker_count": int(processed_df["Ticker"].nunique()),
            "tickers": sorted(processed_df["Ticker"].unique().tolist()),
            "start_date": processed_df["Date"].min().date().isoformat(),
            "end_date": processed_df["Date"].max().date().isoformat(),
            "columns": processed_df.columns.tolist(),
        }

    return {
        "chat_id": paths["chat_id"],
        "workspace_dir": str(paths["workspace_dir"]),
        "raw_files": raw_status,
        "raw_storage": str(RAW_DATA_DIR),
        "processed_dataset": processed_status,
    }


def get_local_data_inventory(chat_id=None):
    processed_status = get_dataset_status()
    raw_status = list_local_raw_data()
    workspace_status = get_llm_workspace_status(chat_id=chat_id)
    active_status = get_active_analysis_dataset_status()
    processed_tickers = set(processed_status.get("tickers", []))
    raw_tickers = set(raw_status.get("tickers", []))

    return {
        "active_analysis_dataset": active_status,
        "processed_dataset": processed_status,
        "raw_files": raw_status,
        "llm_workspace": workspace_status,
        "raw_only_tickers": sorted(raw_tickers.difference(processed_tickers)),
        "processed_only_tickers": sorted(processed_tickers.difference(raw_tickers)),
        "all_local_tickers": sorted(processed_tickers.union(raw_tickers)),
    }


def get_eda_summary():
    df = _read_csv(EDA_SUMMARY_FILE)
    return {
        "available": not df.empty,
        "path": str(EDA_SUMMARY_FILE),
        "records": _json_records(df),
    }


def get_data_quality_summary():
    df = _read_csv(DATA_QUALITY_FILE)
    return {
        "available": not df.empty,
        "path": str(DATA_QUALITY_FILE),
        "records": _json_records(df),
    }


def list_available_figures():
    return {
        "figures": [path.name for path in sorted(FIGURES_DIR.glob("*.png"))],
        "directory": str(FIGURES_DIR),
    }


def _configure_eda_module(eda_module):
    eda_module.PROCESSED_DATA_FILE = PROCESSED_DATA_FILE
    eda_module.FIGURES_DIR = FIGURES_DIR
    eda_module.RESULTS_DIR = EDA_SUMMARY_FILE.parent
    eda_module.SUMMARY_FILE = EDA_SUMMARY_FILE
    eda_module.DATA_QUALITY_FILE = DATA_QUALITY_FILE


def get_ticker_history(ticker, start_date=None, end_date=None, columns=None, max_rows=500, chat_id=None):
    ticker = normalize_ticker(ticker)
    df = _read_processed_data()
    data_source = "project"
    filtered = df[df["Ticker"] == ticker].copy()
    if filtered.empty:
        workspace_df = _read_workspace_processed_data(chat_id)
        filtered = workspace_df[workspace_df["Ticker"] == ticker].copy() if not workspace_df.empty else pd.DataFrame()
        data_source = "chat_workspace"
    if filtered.empty:
        return {"available": False, "message": f"No data found for ticker {ticker}.", "records": []}

    if start_date:
        filtered = filtered[filtered["Date"] >= pd.to_datetime(start_date)]
    if end_date:
        filtered = filtered[filtered["Date"] <= pd.to_datetime(end_date)]

    if columns is None:
        columns = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume", "MA5", "MA20", "RSI", "MACD"]
    columns = [column for column in columns if column in filtered.columns]

    filtered = filtered[columns].tail(max_rows)
    return {
        "available": True,
        "ticker": ticker,
        "data_source": data_source,
        "rows": int(len(filtered)),
        "records": _json_records(filtered),
    }


def create_ticker_price_chart(ticker, start_date=None, end_date=None, months=12, include_ma=True, chat_id=None):
    ticker = normalize_ticker(ticker)
    df = _read_processed_data()
    data_source = "project"
    figures_dir = FIGURES_DIR
    ticker_df = df[df["Ticker"] == ticker].copy()
    if ticker_df.empty:
        workspace_df = _read_workspace_processed_data(chat_id)
        ticker_df = workspace_df[workspace_df["Ticker"] == ticker].copy() if not workspace_df.empty else pd.DataFrame()
        data_source = "chat_workspace"
        figures_dir = get_chat_workspace_paths(chat_id)["figures_dir"]
    if ticker_df.empty:
        return {
            "available": False,
            "message": f"No data found for ticker {ticker}.",
            "hint": "If the user provided a company name, validate ticker candidates first.",
        }

    ticker_df = ticker_df.sort_values("Date")
    if end_date:
        end_ts = pd.to_datetime(end_date)
    else:
        end_ts = ticker_df["Date"].max()

    if start_date:
        start_ts = pd.to_datetime(start_date)
    else:
        start_ts = end_ts - pd.DateOffset(months=int(months))

    chart_df = ticker_df[(ticker_df["Date"] >= start_ts) & (ticker_df["Date"] <= end_ts)].copy()
    if chart_df.empty:
        return {
            "available": False,
            "message": f"No rows found for {ticker} between {start_ts.date()} and {end_ts.date()}.",
            "dataset_start": ticker_df["Date"].min().date().isoformat(),
            "dataset_end": ticker_df["Date"].max().date().isoformat(),
        }

    figures_dir.mkdir(parents=True, exist_ok=True)
    safe_start = chart_df["Date"].min().date().isoformat()
    safe_end = chart_df["Date"].max().date().isoformat()
    figure_path = figures_dir / f"price_chart_{ticker}_{safe_start}_{safe_end}.png"

    plt.figure(figsize=(12, 6))
    plt.plot(chart_df["Date"], chart_df["Close"], label="Close", linewidth=1.4)
    if include_ma:
        if "MA5" in chart_df.columns:
            plt.plot(chart_df["Date"], chart_df["MA5"], label="MA5", linewidth=1.0)
        if "MA20" in chart_df.columns:
            plt.plot(chart_df["Date"], chart_df["MA20"], label="MA20", linewidth=1.0)

    plt.title(f"{ticker} Price Chart ({safe_start} to {safe_end})")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()

    first_close = float(chart_df["Close"].iloc[0])
    last_close = float(chart_df["Close"].iloc[-1])
    total_return = last_close / first_close - 1

    return {
        "available": True,
        "ticker": ticker,
        "data_source": data_source,
        "figure_path": str(figure_path),
        "rows": int(len(chart_df)),
        "start_date": safe_start,
        "end_date": safe_end,
        "first_close": first_close,
        "last_close": last_close,
        "total_return": float(total_return),
        "message": f"Generated price chart for {ticker} from {safe_start} to {safe_end}.",
    }


def get_ticker_metrics(ticker):
    summary = _read_csv(EDA_SUMMARY_FILE)
    if summary.empty:
        return {"available": False, "message": "EDA summary is not available."}

    ticker = normalize_ticker(ticker)
    row = summary[summary["Ticker"] == ticker]
    if row.empty:
        return {"available": False, "message": f"No EDA metrics found for ticker {ticker}."}

    record = row.iloc[0].to_dict()
    for key, value in list(record.items()):
        if hasattr(value, "item"):
            record[key] = value.item()
    return {"available": True, "ticker": ticker, "metrics": record}


def _strategy_metric_columns(extra_columns):
    return [
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
        *extra_columns,
    ]


def _build_strategy_data(tickers=None, data_scope="auto", chat_id=None):
    data_scope = str(data_scope).lower()
    if data_scope not in {"auto", "project", "workspace"}:
        raise ValueError("data_scope must be one of: auto, project, workspace.")

    project_df = _read_processed_data()
    workspace_df = _read_workspace_processed_data(chat_id)

    if tickers is None:
        if data_scope == "workspace":
            if workspace_df.empty:
                raise ValueError("No AI chat workspace processed data available.")
            return workspace_df, sorted(workspace_df["Ticker"].unique()), "chat_workspace"
        if project_df.empty:
            raise ValueError("No project processed data available.")
        return project_df, sorted(project_df["Ticker"].unique()), "project"

    selected_tickers = merge_ticker_lists(tickers)
    if not selected_tickers:
        raise ValueError("tickers cannot be empty.")

    frames = []
    source_rows = []
    for ticker in selected_tickers:
        project_rows = project_df[project_df["Ticker"] == ticker].copy() if not project_df.empty else pd.DataFrame()
        workspace_rows = workspace_df[workspace_df["Ticker"] == ticker].copy() if not workspace_df.empty else pd.DataFrame()

        if data_scope in {"auto", "project"} and not project_rows.empty:
            frames.append(project_rows)
            source_rows.append("project")
        elif data_scope in {"auto", "workspace"} and not workspace_rows.empty:
            frames.append(workspace_rows)
            source_rows.append("chat_workspace")
        else:
            raise ValueError(f"No data found for ticker: {ticker}")

    output_scope = "chat_workspace" if "chat_workspace" in source_rows else "project"
    return pd.concat(frames, axis=0, ignore_index=True), selected_tickers, output_scope


def _save_strategy_outputs(equity_curves, metrics_rows, metric_columns, equity_file, metrics_file):
    equity_file = Path(equity_file)
    metrics_file = Path(metrics_file)
    equity_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    equity_df = pd.concat(equity_curves, axis=0, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df[metric_columns].sort_values("total_return", ascending=False)
    equity_df.to_csv(equity_file, index=False)
    metrics_df.to_csv(metrics_file, index=False)
    return equity_df, metrics_df


def run_buy_hold_baseline(tickers=None, initial_cash=100_000.0, data_scope="auto", chat_id=None):
    from src.strategies.buy_hold import run_buy_hold_for_ticker

    df, selected_tickers, output_scope = _build_strategy_data(tickers=tickers, data_scope=data_scope, chat_id=chat_id)
    equity_curves = []
    metrics_rows = []
    for ticker in selected_tickers:
        equity_curve, metrics = run_buy_hold_for_ticker(df=df, ticker=ticker, initial_cash=initial_cash)
        equity_curves.append(equity_curve)
        metrics_rows.append(metrics)

    if output_scope == "chat_workspace":
        paths = get_chat_workspace_paths(chat_id)
        equity_file = paths["buy_hold_equity_file"]
        metrics_file = paths["buy_hold_metrics_file"]
    else:
        equity_file = BUY_HOLD_EQUITY_FILE
        metrics_file = BUY_HOLD_METRICS_FILE

    _, metrics = _save_strategy_outputs(
        equity_curves=equity_curves,
        metrics_rows=metrics_rows,
        metric_columns=_strategy_metric_columns(["Shares", "Remaining_Cash"]),
        equity_file=equity_file,
        metrics_file=metrics_file,
    )
    return {
        "strategy": "BuyHold",
        "data_scope": output_scope,
        "metrics_file": str(metrics_file),
        "equity_file": str(equity_file),
        "records": _json_records(metrics),
    }


def run_ma_baseline(tickers=None, initial_cash=100_000.0, short_window=5, long_window=20, data_scope="auto", chat_id=None):
    from src.strategies.moving_average import run_ma_for_ticker

    df, selected_tickers, output_scope = _build_strategy_data(tickers=tickers, data_scope=data_scope, chat_id=chat_id)
    equity_curves = []
    metrics_rows = []
    for ticker in selected_tickers:
        equity_curve, metrics = run_ma_for_ticker(
            df=df,
            ticker=ticker,
            initial_cash=initial_cash,
            short_window=short_window,
            long_window=long_window,
        )
        equity_curves.append(equity_curve)
        metrics_rows.append(metrics)

    if output_scope == "chat_workspace":
        paths = get_chat_workspace_paths(chat_id)
        equity_file = paths["ma_equity_file"]
        metrics_file = paths["ma_metrics_file"]
    else:
        equity_file = MA_EQUITY_FILE
        metrics_file = MA_METRICS_FILE

    _, metrics = _save_strategy_outputs(
        equity_curves=equity_curves,
        metrics_rows=metrics_rows,
        metric_columns=_strategy_metric_columns(["Short_Window", "Long_Window", "Trades", "Final_Shares", "Final_Cash"]),
        equity_file=equity_file,
        metrics_file=metrics_file,
    )
    return {
        "strategy": "MovingAverage",
        "data_scope": output_scope,
        "metrics_file": str(metrics_file),
        "equity_file": str(equity_file),
        "records": _json_records(metrics),
    }


def run_rsi_baseline(
    tickers=None,
    initial_cash=100_000.0,
    buy_threshold=30,
    sell_threshold=70,
    data_scope="auto",
    chat_id=None,
):
    from src.strategies.rsi import run_rsi_for_ticker

    df, selected_tickers, output_scope = _build_strategy_data(tickers=tickers, data_scope=data_scope, chat_id=chat_id)
    equity_curves = []
    metrics_rows = []
    for ticker in selected_tickers:
        equity_curve, metrics = run_rsi_for_ticker(
            df=df,
            ticker=ticker,
            initial_cash=initial_cash,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )
        equity_curves.append(equity_curve)
        metrics_rows.append(metrics)

    if output_scope == "chat_workspace":
        paths = get_chat_workspace_paths(chat_id)
        equity_file = paths["rsi_equity_file"]
        metrics_file = paths["rsi_metrics_file"]
    else:
        equity_file = RSI_EQUITY_FILE
        metrics_file = RSI_METRICS_FILE

    _, metrics = _save_strategy_outputs(
        equity_curves=equity_curves,
        metrics_rows=metrics_rows,
        metric_columns=_strategy_metric_columns(["Buy_Threshold", "Sell_Threshold", "Trades", "Final_Shares", "Final_Cash"]),
        equity_file=equity_file,
        metrics_file=metrics_file,
    )
    return {
        "strategy": "RSI",
        "data_scope": output_scope,
        "metrics_file": str(metrics_file),
        "equity_file": str(equity_file),
        "records": _json_records(metrics),
    }


def get_buy_hold_metrics(ticker=None, chat_id=None):
    metrics = _read_csv(BUY_HOLD_METRICS_FILE)
    data_scope = "project"
    if ticker:
        ticker = normalize_ticker(ticker)
        filtered = metrics[metrics["Ticker"] == ticker] if not metrics.empty else pd.DataFrame()
        if filtered.empty:
            workspace_metrics = _read_csv(get_chat_workspace_paths(chat_id)["buy_hold_metrics_file"])
            metrics = workspace_metrics[workspace_metrics["Ticker"] == ticker] if not workspace_metrics.empty else pd.DataFrame()
            data_scope = "chat_workspace"
        else:
            metrics = filtered
    if metrics.empty:
        return {
            "available": False,
            "message": "Buy & Hold metrics are not available. Run run_buy_hold_baseline first.",
            "records": [],
        }

    return {
        "available": not metrics.empty,
        "strategy": "BuyHold",
        "data_scope": data_scope,
        "path": str(get_chat_workspace_paths(chat_id)["buy_hold_metrics_file"] if data_scope == "chat_workspace" else BUY_HOLD_METRICS_FILE),
        "records": _json_records(metrics),
    }


def get_ma_metrics(ticker=None, chat_id=None):
    metrics = _read_csv(MA_METRICS_FILE)
    data_scope = "project"
    if ticker:
        ticker = normalize_ticker(ticker)
        filtered = metrics[metrics["Ticker"] == ticker] if not metrics.empty else pd.DataFrame()
        if filtered.empty:
            workspace_metrics = _read_csv(get_chat_workspace_paths(chat_id)["ma_metrics_file"])
            metrics = workspace_metrics[workspace_metrics["Ticker"] == ticker] if not workspace_metrics.empty else pd.DataFrame()
            data_scope = "chat_workspace"
        else:
            metrics = filtered
    if metrics.empty:
        return {
            "available": False,
            "message": "Moving Average metrics are not available. Run run_ma_baseline first.",
            "records": [],
        }

    return {
        "available": not metrics.empty,
        "strategy": "MovingAverage",
        "data_scope": data_scope,
        "path": str(get_chat_workspace_paths(chat_id)["ma_metrics_file"] if data_scope == "chat_workspace" else MA_METRICS_FILE),
        "records": _json_records(metrics),
    }


def get_rsi_metrics(ticker=None, chat_id=None):
    metrics = _read_csv(RSI_METRICS_FILE)
    data_scope = "project"
    if ticker:
        ticker = normalize_ticker(ticker)
        filtered = metrics[metrics["Ticker"] == ticker] if not metrics.empty else pd.DataFrame()
        if filtered.empty:
            workspace_metrics = _read_csv(get_chat_workspace_paths(chat_id)["rsi_metrics_file"])
            metrics = workspace_metrics[workspace_metrics["Ticker"] == ticker] if not workspace_metrics.empty else pd.DataFrame()
            data_scope = "chat_workspace"
        else:
            metrics = filtered
    if metrics.empty:
        return {
            "available": False,
            "message": "RSI metrics are not available. Run run_rsi_baseline first.",
            "records": [],
        }

    return {
        "available": not metrics.empty,
        "strategy": "RSI",
        "data_scope": data_scope,
        "path": str(get_chat_workspace_paths(chat_id)["rsi_metrics_file"] if data_scope == "chat_workspace" else RSI_METRICS_FILE),
        "records": _json_records(metrics),
    }


def get_buy_hold_equity_curve(ticker, max_rows=500, data_scope="auto", chat_id=None):
    data_scope = str(data_scope).lower()
    if data_scope not in {"auto", "project", "workspace"}:
        raise ValueError("data_scope must be one of: auto, project, workspace.")
    paths = get_chat_workspace_paths(chat_id)
    equity = _read_csv(paths["buy_hold_equity_file"] if data_scope == "workspace" else BUY_HOLD_EQUITY_FILE)
    output_scope = "chat_workspace" if data_scope == "workspace" else "project"
    ticker = normalize_ticker(ticker)
    filtered = equity[equity["Ticker"] == ticker].copy() if not equity.empty else pd.DataFrame()
    if filtered.empty and data_scope == "auto":
        workspace_equity = _read_csv(paths["buy_hold_equity_file"])
        equity = workspace_equity[workspace_equity["Ticker"] == ticker].copy() if not workspace_equity.empty else pd.DataFrame()
        output_scope = "chat_workspace"
    else:
        equity = filtered
    if equity.empty:
        return {
            "available": False,
            "message": "Buy & Hold equity curve is not available. Run run_buy_hold_baseline first.",
            "records": [],
        }

    if "Date" in equity.columns:
        equity["Date"] = pd.to_datetime(equity["Date"])

    return {
        "available": True,
        "strategy": "BuyHold",
        "data_scope": output_scope,
        "ticker": ticker,
        "rows": int(len(equity)),
        "records": _json_records(equity.tail(max_rows)),
    }


def get_ma_equity_curve(ticker, max_rows=500, data_scope="auto", chat_id=None):
    data_scope = str(data_scope).lower()
    if data_scope not in {"auto", "project", "workspace"}:
        raise ValueError("data_scope must be one of: auto, project, workspace.")
    paths = get_chat_workspace_paths(chat_id)
    equity = _read_csv(paths["ma_equity_file"] if data_scope == "workspace" else MA_EQUITY_FILE)
    output_scope = "chat_workspace" if data_scope == "workspace" else "project"
    ticker = normalize_ticker(ticker)
    filtered = equity[equity["Ticker"] == ticker].copy() if not equity.empty else pd.DataFrame()
    if filtered.empty and data_scope == "auto":
        workspace_equity = _read_csv(paths["ma_equity_file"])
        equity = workspace_equity[workspace_equity["Ticker"] == ticker].copy() if not workspace_equity.empty else pd.DataFrame()
        output_scope = "chat_workspace"
    else:
        equity = filtered
    if equity.empty:
        return {
            "available": False,
            "message": "Moving Average equity curve is not available. Run run_ma_baseline first.",
            "records": [],
        }

    if "Date" in equity.columns:
        equity["Date"] = pd.to_datetime(equity["Date"])

    return {
        "available": True,
        "strategy": "MovingAverage",
        "data_scope": output_scope,
        "ticker": ticker,
        "rows": int(len(equity)),
        "records": _json_records(equity.tail(max_rows)),
    }


def get_rsi_equity_curve(ticker, max_rows=500, data_scope="auto", chat_id=None):
    data_scope = str(data_scope).lower()
    if data_scope not in {"auto", "project", "workspace"}:
        raise ValueError("data_scope must be one of: auto, project, workspace.")
    paths = get_chat_workspace_paths(chat_id)
    equity = _read_csv(paths["rsi_equity_file"] if data_scope == "workspace" else RSI_EQUITY_FILE)
    output_scope = "chat_workspace" if data_scope == "workspace" else "project"
    ticker = normalize_ticker(ticker)
    filtered = equity[equity["Ticker"] == ticker].copy() if not equity.empty else pd.DataFrame()
    if filtered.empty and data_scope == "auto":
        workspace_equity = _read_csv(paths["rsi_equity_file"])
        equity = workspace_equity[workspace_equity["Ticker"] == ticker].copy() if not workspace_equity.empty else pd.DataFrame()
        output_scope = "chat_workspace"
    else:
        equity = filtered
    if equity.empty:
        return {
            "available": False,
            "message": "RSI equity curve is not available. Run run_rsi_baseline first.",
            "records": [],
        }

    if "Date" in equity.columns:
        equity["Date"] = pd.to_datetime(equity["Date"])

    return {
        "available": True,
        "strategy": "RSI",
        "data_scope": output_scope,
        "ticker": ticker,
        "rows": int(len(equity)),
        "records": _json_records(equity.tail(max_rows)),
    }


def run_strategy_comparison(data_scope="project", chat_id=None):
    from src.evaluation.strategy_comparison import build_strategy_comparison

    data_scope = str(data_scope).lower()
    if data_scope not in {"project", "workspace"}:
        raise ValueError("data_scope must be one of: project, workspace.")

    if data_scope == "workspace":
        paths = get_chat_workspace_paths(chat_id)
        metrics_files = {
            "BuyHold": paths["buy_hold_metrics_file"],
            "MovingAverage": paths["ma_metrics_file"],
            "RSI": paths["rsi_metrics_file"],
        }
        output_file = paths["strategy_comparison_file"]
        output_scope = "chat_workspace"
    else:
        metrics_files = {
            "BuyHold": BUY_HOLD_METRICS_FILE,
            "MovingAverage": MA_METRICS_FILE,
            "RSI": RSI_METRICS_FILE,
        }
        output_file = STRATEGY_COMPARISON_FILE
        output_scope = "project"

    comparison = build_strategy_comparison(metrics_files=metrics_files, output_file=output_file)
    return {
        "available": not comparison.empty,
        "data_scope": output_scope,
        "comparison_file": str(output_file),
        "rows": int(len(comparison)),
        "records": _json_records(comparison),
    }


def get_strategy_comparison(ticker=None, data_scope="auto", chat_id=None):
    data_scope = str(data_scope).lower()
    if data_scope not in {"auto", "project", "workspace"}:
        raise ValueError("data_scope must be one of: auto, project, workspace.")

    ticker = normalize_ticker(ticker) if ticker else None
    candidate_files = []
    if data_scope in {"auto", "project"}:
        candidate_files.append(("project", STRATEGY_COMPARISON_FILE))
    if data_scope in {"auto", "workspace"}:
        candidate_files.append(("chat_workspace", get_chat_workspace_paths(chat_id)["strategy_comparison_file"]))

    for scope, file_path in candidate_files:
        comparison = _read_csv(file_path)
        if ticker and not comparison.empty:
            comparison = comparison[comparison["Ticker"] == ticker]
        if not comparison.empty:
            return {
                "available": True,
                "data_scope": scope,
                "path": str(file_path),
                "records": _json_records(comparison),
            }

    return {
        "available": False,
        "message": "Strategy comparison is not available. Run run_strategy_comparison first.",
        "records": [],
    }


def refresh_market_data(
    tickers,
    start_date,
    end_date,
    use_proxy=True,
    run_eda_after=True,
    run_baseline_after=True,
    data_source="auto",
):
    from src.analysis import eda
    from src.data import download_data
    from src.features.feature_engineering import add_technical_indicators, save_processed_data

    if not tickers:
        raise ValueError("tickers cannot be empty.")

    tickers = merge_ticker_lists(tickers)
    data_source = str(data_source).lower()
    if data_source not in {"auto", "download"}:
        raise ValueError("data_source must be one of: auto, download.")

    download_data.USE_PROXY = bool(use_proxy)
    download_data.RAW_DATA_DIR = RAW_DATA_DIR
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if use_proxy:
        download_data.setup_proxy()

    raw_frames = []
    source_rows = []

    for ticker in tickers:
        local_status = inspect_local_raw_ticker(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval=SUPPORTED_INTERVAL,
        )
        use_local = data_source == "auto" and local_status["available"]

        if use_local:
            raw_df = load_local_raw_ticker(ticker, start_date=start_date, end_date=end_date)
            raw_frames.append(raw_df)
            source_rows.append({"ticker": ticker, "source": "local_raw", "status": local_status})
            continue

        try:
            downloaded_df = download_data.download_ticker_data(
                ticker=ticker,
                start_date=str(start_date),
                end_date=str(end_date),
                interval=SUPPORTED_INTERVAL,
            )
            download_error = None
        except Exception as exc:
            downloaded_df = pd.DataFrame()
            download_error = str(exc)
        if not downloaded_df.empty:
            file_path = RAW_DATA_DIR / f"{ticker}.csv"
            downloaded_df.to_csv(file_path, index=False)
            raw_frames.append(downloaded_df)
            source_rows.append({"ticker": ticker, "source": "yfinance", "rows": int(len(downloaded_df))})
        else:
            source_rows.append({"ticker": ticker, "source": "yfinance", "rows": 0, "error": download_error or "No data returned. yfinance may be rate-limited or the ticker/date range may be unavailable."})

    if not raw_frames:
        _raise_no_usable_data(source_rows)

    raw_df = pd.concat(raw_frames, axis=0, ignore_index=True)
    feature_df = add_technical_indicators(raw_df)
    save_processed_data(feature_df, output_path=PROCESSED_DATA_FILE)
    runtime_db = _save_runtime_dataset(
        dataset_id="project_active",
        raw_df=raw_df,
        feature_df=feature_df,
        scope="project",
    )

    result = {
        "requested_tickers": tickers,
        "data_source_mode": data_source,
        "source_details": source_rows,
        "raw_rows_used": int(len(raw_df)),
        "processed_rows": int(len(feature_df)),
        "processed_ticker_count": int(feature_df["Ticker"].nunique()),
        "processed_file": str(PROCESSED_DATA_FILE),
        "runtime_database": runtime_db,
    }

    if run_eda_after:
        _configure_eda_module(eda)
        eda.run_eda(example_ticker=tickers[0])
        result["eda_summary_file"] = str(EDA_SUMMARY_FILE)
        result["figures_dir"] = str(FIGURES_DIR)

    if run_baseline_after:
        buy_hold_result = run_buy_hold_baseline()
        ma_result = run_ma_baseline()
        rsi_result = run_rsi_baseline()
        comparison_result = run_strategy_comparison()
        result["buy_hold_metrics_file"] = buy_hold_result["metrics_file"]
        result["buy_hold_equity_file"] = buy_hold_result["equity_file"]
        result["ma_metrics_file"] = ma_result["metrics_file"]
        result["ma_equity_file"] = ma_result["equity_file"]
        result["rsi_metrics_file"] = rsi_result["metrics_file"]
        result["rsi_equity_file"] = rsi_result["equity_file"]
        result["strategy_comparison_file"] = comparison_result["comparison_file"]

    return result


def refresh_llm_workspace_data(
    tickers,
    start_date,
    end_date,
    use_proxy=True,
    run_baseline_after=True,
    data_source="auto",
    chat_id=None,
):
    from src.data import download_data
    from src.features.feature_engineering import add_technical_indicators

    if not tickers:
        raise ValueError("tickers cannot be empty.")

    tickers = merge_ticker_lists(tickers)
    data_source = str(data_source).lower()
    if data_source not in {"auto", "download"}:
        raise ValueError("data_source must be one of: auto, download.")

    paths = get_chat_workspace_paths(chat_id)
    download_data.USE_PROXY = bool(use_proxy)
    download_data.RAW_DATA_DIR = RAW_DATA_DIR
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths["processed_dir"].mkdir(parents=True, exist_ok=True)
    if use_proxy:
        download_data.setup_proxy()

    raw_frames = []
    source_rows = []

    for ticker in tickers:
        local_status = inspect_local_raw_ticker(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            interval=SUPPORTED_INTERVAL,
        )

        if data_source == "auto" and local_status["available"]:
            raw_df = load_local_raw_ticker(ticker, start_date=start_date, end_date=end_date)
            raw_frames.append(raw_df)
            source_rows.append({"ticker": ticker, "source": "shared_raw", "status": local_status})
            continue

        try:
            downloaded_df = download_data.download_ticker_data(
                ticker=ticker,
                start_date=str(start_date),
                end_date=str(end_date),
                interval=SUPPORTED_INTERVAL,
            )
            download_error = None
        except Exception as exc:
            downloaded_df = pd.DataFrame()
            download_error = str(exc)
        if not downloaded_df.empty:
            downloaded_df.to_csv(RAW_DATA_DIR / f"{ticker}.csv", index=False)
            raw_frames.append(downloaded_df)
            source_rows.append({"ticker": ticker, "source": "yfinance_to_shared_raw", "rows": int(len(downloaded_df))})
        else:
            source_rows.append({"ticker": ticker, "source": "yfinance_to_shared_raw", "rows": 0, "error": download_error or "No data returned. yfinance may be rate-limited or the ticker/date range may be unavailable."})

    if not raw_frames:
        _raise_no_usable_data(source_rows)

    raw_df = pd.concat(raw_frames, axis=0, ignore_index=True)
    feature_df = add_technical_indicators(raw_df)
    feature_df.to_csv(paths["processed_file"], index=False)
    runtime_db = _save_runtime_dataset(
        dataset_id=f"chat_{paths['chat_id']}_active",
        raw_df=raw_df,
        feature_df=feature_df,
        scope=f"chat:{paths['chat_id']}",
    )

    result = {
        "requested_tickers": tickers,
        "chat_id": paths["chat_id"],
        "workspace_dir": str(paths["workspace_dir"]),
        "raw_storage": str(RAW_DATA_DIR),
        "data_source_mode": data_source,
        "source_details": source_rows,
        "raw_rows_used": int(len(raw_df)),
        "processed_rows": int(len(feature_df)),
        "processed_ticker_count": int(feature_df["Ticker"].nunique()),
        "processed_file": str(paths["processed_file"]),
        "runtime_database": runtime_db,
        "message": "Chat workspace data refreshed without modifying the main project dataset.",
    }

    if run_baseline_after:
        buy_hold_result = run_buy_hold_baseline(tickers=tickers, data_scope="workspace", chat_id=chat_id)
        ma_result = run_ma_baseline(tickers=tickers, data_scope="workspace", chat_id=chat_id)
        rsi_result = run_rsi_baseline(tickers=tickers, data_scope="workspace", chat_id=chat_id)
        comparison_result = run_strategy_comparison(data_scope="workspace", chat_id=chat_id)
        result["buy_hold_metrics_file"] = buy_hold_result["metrics_file"]
        result["buy_hold_equity_file"] = buy_hold_result["equity_file"]
        result["ma_metrics_file"] = ma_result["metrics_file"]
        result["ma_equity_file"] = ma_result["equity_file"]
        result["rsi_metrics_file"] = rsi_result["metrics_file"]
        result["rsi_equity_file"] = rsi_result["equity_file"]
        result["strategy_comparison_file"] = comparison_result["comparison_file"]

    return result


def refresh_llm_workspace_ticker(ticker, start_date, end_date, use_proxy=True, run_baseline_after=True, data_source="auto", chat_id=None):
    ticker = normalize_ticker(ticker)
    result = refresh_llm_workspace_data(
        tickers=[ticker],
        start_date=start_date,
        end_date=end_date,
        use_proxy=use_proxy,
        run_baseline_after=run_baseline_after,
        data_source=data_source,
        chat_id=chat_id,
    )
    result["requested_ticker"] = ticker
    return result


def clear_llm_workspace(chat_id=None):
    paths = get_chat_workspace_paths(chat_id)
    if paths["workspace_dir"].exists():
        shutil.rmtree(paths["workspace_dir"])
    if chat_id is None and LEGACY_LLM_WORKSPACE_DIR.exists():
        shutil.rmtree(LEGACY_LLM_WORKSPACE_DIR)
    active_config = _load_active_analysis_dataset_config()
    if active_config["source"] in {"llm_workspace", "chat_workspace"} and normalize_chat_id(active_config.get("chat_id")) == paths["chat_id"]:
        _save_active_analysis_dataset_config(
            source="project",
            note="Reset automatically because the AI chat workspace was cleared.",
            chat_id=None,
        )
    return {
        "cleared": True,
        "chat_id": paths["chat_id"],
        "workspace_dir": str(paths["workspace_dir"]),
        "message": "AI chat workspace cleared.",
    }


def merge_llm_workspace_to_project(tickers=None, run_eda_after=True, run_baseline_after=True, chat_id=None):
    from src.analysis.eda import run_eda

    workspace_df = _read_workspace_processed_data(chat_id)
    available_tickers = sorted(workspace_df["Ticker"].unique().tolist()) if not workspace_df.empty else []
    selected_tickers = merge_ticker_lists(tickers or available_tickers)
    if not selected_tickers:
        raise ValueError("No AI chat workspace tickers are available to merge.")

    result = refresh_market_data(
        tickers=selected_tickers,
        start_date=workspace_df["Date"].min().date().isoformat(),
        end_date=workspace_df["Date"].max().date().isoformat(),
        use_proxy=False,
        run_eda_after=run_eda_after,
        run_baseline_after=run_baseline_after,
        data_source="auto",
    )

    result["merged_tickers"] = selected_tickers
    result["source_chat_id"] = get_chat_workspace_paths(chat_id)["chat_id"]
    return result


def refresh_ticker_data(
    ticker,
    start_date,
    end_date,
    use_proxy=True,
    run_eda_after=True,
    run_baseline_after=True,
    data_source="auto",
):
    ticker = normalize_ticker(ticker)
    result = refresh_market_data(
        tickers=[ticker],
        start_date=start_date,
        end_date=end_date,
        use_proxy=use_proxy,
        run_eda_after=run_eda_after,
        run_baseline_after=run_baseline_after,
        data_source=data_source,
    )
    result["requested_ticker"] = ticker
    return result
