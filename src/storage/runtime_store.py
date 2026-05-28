from datetime import datetime
from pathlib import Path
import sqlite3

import pandas as pd


RUNTIME_ROOT = Path(".streamlit_runtime")
DATABASE_NAME = "runtime.db"


def normalize_session_id(session_id):
    text = str(session_id or "default").strip()
    safe = "".join(char for char in text if char.isalnum() or char in {"-", "_"})
    return safe or "default"


def get_runtime_paths(session_id=None, root=None):
    session_id = normalize_session_id(session_id)
    root = Path(root or RUNTIME_ROOT)
    session_dir = root / "sessions" / session_id
    return {
        "session_id": session_id,
        "root": root,
        "db_file": root / DATABASE_NAME,
        "session_dir": session_dir,
        "config_dir": session_dir / "config",
        "raw_dir": session_dir / "raw",
        "processed_dir": session_dir / "processed",
        "reference_dir": root / "reference",
        "reports_dir": session_dir / "reports",
        "results_dir": session_dir / "reports" / "results",
        "figures_dir": session_dir / "reports" / "figures",
        "logs_dir": session_dir / "reports" / "logs",
        "tool_proposals_dir": session_dir / "reports" / "tool_proposals",
        "temp_tools_file": session_dir / "config" / "temp_composite_tools.json",
        "chat_workspaces_dir": session_dir / "workspaces" / "chats",
    }


def ensure_runtime(session_id=None, root=None):
    paths = get_runtime_paths(session_id=session_id, root=root)
    for key in [
        "root",
        "session_dir",
        "config_dir",
        "raw_dir",
        "processed_dir",
        "reference_dir",
        "results_dir",
        "figures_dir",
        "logs_dir",
        "tool_proposals_dir",
        "chat_workspaces_dir",
    ]:
        paths[key].mkdir(parents=True, exist_ok=True)
    init_database(paths["db_file"])
    return paths


def get_connection(db_file):
    db_file = Path(db_file)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_file)


def init_database(db_file):
    with get_connection(db_file) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                scope TEXT NOT NULL,
                tickers TEXT NOT NULL,
                start_date TEXT,
                end_date TEXT,
                interval TEXT,
                row_count INTEGER NOT NULL,
                feature_row_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_prices (
                dataset_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS features (
                dataset_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                daily_return REAL,
                ma5 REAL,
                ma20 REAL,
                rsi REAL,
                macd REAL,
                volatility REAL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_raw_prices_dataset ON raw_prices(dataset_id, ticker, date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_features_dataset ON features(dataset_id, ticker, date)")


def _prepare_dataframe(df):
    output = df.copy()
    for column in output.columns:
        if pd.api.types.is_datetime64_any_dtype(output[column]):
            output[column] = output[column].dt.strftime("%Y-%m-%d")
    return output


def _normalize_market_columns(df):
    rename_map = {
        "Date": "date",
        "Ticker": "ticker",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Daily_Return": "daily_return",
        "MA5": "ma5",
        "MA20": "ma20",
        "RSI": "rsi",
        "MACD": "macd",
        "Volatility": "volatility",
    }
    return _prepare_dataframe(df).rename(columns=rename_map)


def save_runtime_dataset(
    db_file,
    session_id,
    dataset_id,
    raw_df,
    feature_df=None,
    scope="project",
    interval="1d",
):
    session_id = normalize_session_id(session_id)
    raw_df = raw_df.copy()
    feature_df = pd.DataFrame() if feature_df is None else feature_df.copy()
    tickers = sorted(raw_df["Ticker"].astype(str).unique().tolist()) if not raw_df.empty else []
    start_date = pd.to_datetime(raw_df["Date"]).min().date().isoformat() if not raw_df.empty else None
    end_date = pd.to_datetime(raw_df["Date"]).max().date().isoformat() if not raw_df.empty else None

    raw_rows = _normalize_market_columns(raw_df)
    raw_rows["dataset_id"] = dataset_id
    raw_rows["session_id"] = session_id
    raw_columns = ["dataset_id", "session_id", "date", "ticker", "open", "high", "low", "close", "volume"]
    raw_rows = raw_rows[[column for column in raw_columns if column in raw_rows.columns]]

    feature_rows = _normalize_market_columns(feature_df)
    feature_rows["dataset_id"] = dataset_id
    feature_rows["session_id"] = session_id
    feature_columns = [
        "dataset_id",
        "session_id",
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "daily_return",
        "ma5",
        "ma20",
        "rsi",
        "macd",
        "volatility",
    ]
    feature_rows = feature_rows[[column for column in feature_columns if column in feature_rows.columns]]

    with get_connection(db_file) as conn:
        conn.execute("DELETE FROM raw_prices WHERE dataset_id = ?", (dataset_id,))
        conn.execute("DELETE FROM features WHERE dataset_id = ?", (dataset_id,))
        conn.execute("DELETE FROM datasets WHERE dataset_id = ?", (dataset_id,))
        conn.execute(
            """
            INSERT INTO datasets (
                dataset_id, session_id, scope, tickers, start_date, end_date,
                interval, row_count, feature_row_count, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_id,
                session_id,
                scope,
                ",".join(tickers),
                start_date,
                end_date,
                interval,
                int(len(raw_df)),
                int(len(feature_df)),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        raw_rows.to_sql("raw_prices", conn, if_exists="append", index=False)
        if not feature_rows.empty:
            feature_rows.to_sql("features", conn, if_exists="append", index=False)


def get_runtime_status(db_file, session_id=None):
    session_id = normalize_session_id(session_id)
    db_file = Path(db_file)
    if not db_file.exists():
        return {"available": False, "session_id": session_id, "datasets": []}

    with get_connection(db_file) as conn:
        datasets = pd.read_sql_query(
            """
            SELECT dataset_id, scope, tickers, start_date, end_date, interval,
                   row_count, feature_row_count, created_at
            FROM datasets
            WHERE session_id = ?
            ORDER BY created_at DESC
            """,
            conn,
            params=(session_id,),
        )

    return {
        "available": not datasets.empty,
        "session_id": session_id,
        "database": str(db_file),
        "datasets": datasets.to_dict(orient="records"),
    }
