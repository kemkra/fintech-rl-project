from datetime import datetime
import json
import os
from pathlib import Path
import sys
import threading
import traceback
from uuid import uuid4

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools import market_tools
from src.llm.assistant import (
    PROVIDER_PRESETS,
    list_ollama_models,
    list_openai_compatible_models,
    run_llm_tool_chat,
)


st.set_page_config(page_title="Financial Market Analysis", layout="wide")

CONFIG_DIR = PROJECT_ROOT / "config"
LLM_PREFERENCES_FILE = CONFIG_DIR / "llm_preferences.json"
LLM_CHATS_FILE = CONFIG_DIR / "llm_chats.json"
LLM_JOBS_DIR = CONFIG_DIR / "llm_jobs"
WEB_RUNTIME_ROOT = PROJECT_ROOT / ".streamlit_runtime"
STREAMLIT_CLOUD_ENV_KEYS = [
    "STREAMLIT_CLOUD",
    "STREAMLIT_COMMUNITY_CLOUD",
    "STREAMLIT_SHARING",
    "STREAMLIT_CLOUD_APP_NAME",
    "STREAMLIT_CLOUD_APP_URL",
]
POPULAR_TICKERS = [
    "AAPL",
    "ABBV",
    "ABT",
    "ACN",
    "ADBE",
    "AIG",
    "AMGN",
    "AMD",
    "AMZN",
    "AXP",
    "BA",
    "AVGO",
    "BAC",
    "BRK-B",
    "C",
    "CAT",
    "CHTR",
    "CL",
    "CMCSA",
    "COF",
    "COP",
    "CRM",
    "CSCO",
    "CVS",
    "CVX",
    "COST",
    "DIA",
    "DIS",
    "DUK",
    "F",
    "FDX",
    "GE",
    "GM",
    "GOOGL",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "IWM",
    "JNJ",
    "JPM",
    "KO",
    "LIN",
    "LLY",
    "LOW",
    "MCD",
    "MA",
    "MDT",
    "META",
    "MMM",
    "MRK",
    "MSFT",
    "NEE",
    "NKE",
    "NFLX",
    "NOW",
    "NVDA",
    "ORCL",
    "PFE",
    "PG",
    "PEP",
    "QQQ",
    "RTX",
    "SBUX",
    "SCHW",
    "SLB",
    "SPY",
    "T",
    "TGT",
    "TMO",
    "TSLA",
    "TXN",
    "UBER",
    "UNH",
    "UPS",
    "USB",
    "V",
    "VZ",
    "WFC",
    "WMT",
    "XOM",
]


def should_use_session_runtime_storage():
    mode = os.getenv("FINTECH_STORAGE_MODE", "auto").strip().lower()
    if mode in {"local", "project", "persistent"}:
        return False
    if mode in {"session", "runtime", "web", "cloud"}:
        return True
    try:
        secret_mode = str(st.secrets.get("FINTECH_STORAGE_MODE", "")).strip().lower()
    except Exception:
        secret_mode = ""
    if secret_mode in {"local", "project", "persistent"}:
        return False
    if secret_mode in {"session", "runtime", "web", "cloud"}:
        return True
    project_parts = PROJECT_ROOT.resolve().parts
    if len(project_parts) >= 3 and project_parts[1:3] == ("mount", "src"):
        return True
    if os.getenv("HOME") == "/home/adminuser":
        return True
    return any(os.getenv(key) for key in STREAMLIT_CLOUD_ENV_KEYS)


def init_app_storage():
    global CONFIG_DIR, LLM_PREFERENCES_FILE, LLM_CHATS_FILE, LLM_JOBS_DIR

    if not should_use_session_runtime_storage():
        st.session_state["storage_mode"] = "local"
        st.session_state.setdefault("web_session_id", None)
        st.session_state["runtime_paths"] = None
        CONFIG_DIR = PROJECT_ROOT / "config"
        LLM_PREFERENCES_FILE = CONFIG_DIR / "llm_preferences.json"
        LLM_CHATS_FILE = CONFIG_DIR / "llm_chats.json"
        LLM_JOBS_DIR = CONFIG_DIR / "llm_jobs"
        return {
            "storage_mode": "local",
            "message": "Using persistent project folders: data/, reports/, models/, and config/.",
        }

    if "web_session_id" not in st.session_state or not st.session_state["web_session_id"]:
        st.session_state["web_session_id"] = uuid4().hex[:12]

    st.session_state["storage_mode"] = "session"
    runtime_root = Path(os.getenv("FINTECH_RUNTIME_ROOT", WEB_RUNTIME_ROOT))
    paths = market_tools.configure_runtime_storage(
        session_id=st.session_state["web_session_id"],
        root=runtime_root,
    )
    CONFIG_DIR = Path(paths["config_dir"])
    LLM_PREFERENCES_FILE = CONFIG_DIR / "llm_preferences.json"
    LLM_CHATS_FILE = CONFIG_DIR / "llm_chats.json"
    LLM_JOBS_DIR = CONFIG_DIR / "llm_jobs"
    st.session_state["runtime_paths"] = paths
    paths["storage_mode"] = "session"
    return paths


init_app_storage()


@st.cache_data(show_spinner=False)
def load_processed_data(path, active_updated_at=None):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["Ticker", "Date"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_summary(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def clear_cache():
    st.cache_data.clear()


@st.cache_data(show_spinner=False, ttl=30)
def cached_ollama_models(base_url):
    return list_ollama_models(base_url.replace("/v1", ""))


def get_model_state_key(provider_name, base_url):
    return f"provider_models::{provider_name}::{base_url}"


@st.cache_data(show_spinner=False, ttl=86400)
def get_us_symbol_options(force_refresh_token=0):
    result = market_tools.search_us_symbols(limit=12000, force_refresh=bool(force_refresh_token))
    records = result.get("records", [])
    options = []
    for record in records:
        ticker = record.get("yfinance_symbol") or record.get("symbol")
        name = record.get("name") or ticker
        exchange = record.get("exchange") or ""
        options.append(f"{ticker} | {name} | {exchange}")
    return {
        "options": options,
        "status": {key: value for key, value in result.items() if key != "records"},
    }


def get_load_ticker_options(force_refresh_token=0):
    raw_tickers = market_tools.list_local_raw_data().get("tickers", [])
    symbol_result = get_us_symbol_options(force_refresh_token=force_refresh_token)
    labeled_existing = set(symbol_result["options"])
    quick_options = [
        f"{ticker} | local/default | cached"
        for ticker in sorted(set(market_tools.DEFAULT_TICKERS + POPULAR_TICKERS + raw_tickers))
        if not any(option.startswith(f"{ticker} |") for option in labeled_existing)
    ]
    return quick_options + symbol_result["options"], symbol_result["status"]


def parse_selected_ticker_options(selected_values):
    tickers = []
    for value in selected_values or []:
        token = str(value).split("|", 1)[0].strip()
        ticker = market_tools.normalize_ticker(token)
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers


def load_llm_preferences():
    if not LLM_PREFERENCES_FILE.exists():
        return {}
    try:
        with LLM_PREFERENCES_FILE.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def save_llm_preferences(preferences):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with LLM_PREFERENCES_FILE.open("w", encoding="utf-8") as file:
        json.dump(preferences, file, ensure_ascii=False, indent=2)


def initialize_llm_ui_state(preferences):
    provider_name = preferences.get("provider_name", "Alibaba Bailian")
    if provider_name not in PROVIDER_PRESETS:
        provider_name = "Alibaba Bailian"

    preset = PROVIDER_PRESETS[provider_name]
    defaults = {
        "llm_provider_name": provider_name,
        "llm_base_url": preferences.get("base_url") or preset["base_url"],
        "llm_model": preferences.get("model") or preset["model"],
        "llm_api_key": preferences.get("api_key", "") if preferences.get("remember_api_key") else "",
        "llm_remember_api_key": bool(preferences.get("remember_api_key")),
        "llm_save_debug_log": True,
        "llm_limit_tool_rounds": False,
        "llm_max_tool_rounds": 5,
        "llm_limit_total_runtime": False,
        "llm_max_elapsed_seconds": 180,
        "llm_request_timeout": 120,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("llm_previous_provider_name", st.session_state["llm_provider_name"])


def prepare_llm_widget_state():
    widget_keys = {
        "llm_widget_base_url": "llm_base_url",
        "llm_widget_api_key": "llm_api_key",
        "llm_widget_remember_api_key": "llm_remember_api_key",
        "llm_widget_save_debug_log": "llm_save_debug_log",
        "llm_widget_limit_tool_rounds": "llm_limit_tool_rounds",
        "llm_widget_max_tool_rounds": "llm_max_tool_rounds",
        "llm_widget_limit_total_runtime": "llm_limit_total_runtime",
        "llm_widget_max_elapsed_seconds": "llm_max_elapsed_seconds",
        "llm_widget_request_timeout": "llm_request_timeout",
    }
    for widget_key, state_key in widget_keys.items():
        st.session_state.setdefault(widget_key, st.session_state.get(state_key))


def apply_provider_change_if_needed(preferences):
    provider_name = st.session_state.get("llm_provider_name", "Alibaba Bailian")
    previous_provider = st.session_state.get("llm_previous_provider_name")
    if provider_name == previous_provider:
        return

    preset = PROVIDER_PRESETS[provider_name]
    use_saved_preferences = provider_name == preferences.get("provider_name")
    st.session_state["llm_base_url"] = (preferences.get("base_url") if use_saved_preferences else None) or preset["base_url"]
    st.session_state["llm_model"] = (preferences.get("model") if use_saved_preferences else None) or preset["model"]
    if use_saved_preferences and preferences.get("remember_api_key"):
        st.session_state["llm_api_key"] = preferences.get("api_key", "")
    elif provider_name in {"Ollama Local", "LM Studio Local"}:
        st.session_state.setdefault("llm_api_key", "")
    st.session_state["llm_previous_provider_name"] = provider_name
    st.session_state["llm_widget_base_url"] = st.session_state["llm_base_url"]
    st.session_state["llm_widget_model_text"] = st.session_state["llm_model"]
    st.session_state["llm_widget_api_key"] = st.session_state.get("llm_api_key", "")


def make_chat_title(index):
    return f"Chat {index}"


def get_next_chat_number(chats=None):
    chats = chats or st.session_state.get("llm_chats", {})
    existing_titles = {chat.get("title") for chat in chats.values()}
    number = 1
    while make_chat_title(number) in existing_titles:
        number += 1
    return number


def create_empty_chat(title=None):
    chat_id = uuid4().hex[:8]
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "id": chat_id,
        "title": title or make_chat_title(get_next_chat_number()),
        "messages": [],
        "memory_summary": "",
        "last_result": None,
        "created_at": now,
        "updated_at": now,
    }


def sanitize_llm_result_for_storage(result):
    if not result:
        return None

    return {
        "answer": result.get("answer", ""),
        "provider": result.get("provider"),
        "model": result.get("model"),
        "tools": result.get("tools", []),
        "log_path": result.get("log_path"),
    }


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)
    temp_path.replace(path)


def read_json_file(path):
    try:
        with Path(path).open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def write_llm_job_status(job_file, status, **fields):
    payload = {
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        **fields,
    }
    write_json_atomic(job_file, payload)
    return payload


def run_llm_job_in_background(job_file, storage_mode, runtime_session_id, runtime_root, chat_id, question, llm_kwargs):
    write_llm_job_status(
        job_file,
        "running",
        chat_id=chat_id,
        question=question,
        started_at=datetime.now().isoformat(timespec="seconds"),
    )
    try:
        if storage_mode == "session":
            market_tools.configure_runtime_storage(session_id=runtime_session_id, root=runtime_root)
        result = run_llm_tool_chat(question=question, chat_id=chat_id, **llm_kwargs)
    except Exception as exc:
        write_llm_job_status(
            job_file,
            "failed",
            chat_id=chat_id,
            question=question,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
    else:
        write_llm_job_status(
            job_file,
            "completed",
            chat_id=chat_id,
            question=question,
            result=sanitize_llm_result_for_storage(result),
        )


def start_llm_background_job(chat, question, llm_kwargs):
    LLM_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = uuid4().hex[:12]
    job_file = LLM_JOBS_DIR / f"{job_id}.json"
    now = datetime.now().isoformat(timespec="seconds")
    chat["messages"].append({"role": "user", "content": question})
    chat["pending_job"] = {
        "job_id": job_id,
        "job_file": str(job_file),
        "question": question,
        "started_at": now,
    }
    chat["updated_at"] = now
    write_llm_job_status(job_file, "queued", chat_id=chat["id"], question=question, started_at=now)
    save_llm_chats_to_disk()

    worker = threading.Thread(
        target=run_llm_job_in_background,
        kwargs={
            "job_file": str(job_file),
            "storage_mode": st.session_state.get("storage_mode", "local"),
            "runtime_session_id": st.session_state.get("web_session_id"),
            "runtime_root": str(Path(os.getenv("FINTECH_RUNTIME_ROOT", WEB_RUNTIME_ROOT))),
            "chat_id": chat["id"],
            "question": question,
            "llm_kwargs": llm_kwargs,
        },
        daemon=True,
    )
    worker.start()
    return job_id


def reconcile_llm_job(chat):
    pending_job = chat.get("pending_job")
    if not pending_job:
        return None

    job_status = read_json_file(pending_job.get("job_file"))
    if not job_status:
        return {"status": "unknown", "message": "Waiting for job status file."}

    status = job_status.get("status")
    if status == "completed":
        result = job_status.get("result") or {}
        answer = result.get("answer", "")
        chat["messages"].append({"role": "assistant", "content": answer})
        chat["memory_summary"] = build_memory_summary(chat["messages"])
        chat["last_result"] = result
        chat["updated_at"] = job_status.get("updated_at") or datetime.now().isoformat(timespec="seconds")
        chat.pop("pending_job", None)
        save_llm_chats_to_disk()
        clear_cache()
    elif status == "failed":
        error = job_status.get("error", "Unknown error")
        chat["messages"].append({"role": "assistant", "content": f"LLM call failed: {error}"})
        chat["last_result"] = {
            "answer": f"LLM call failed: {error}",
            "provider": None,
            "model": None,
            "tools": [],
            "log_path": None,
        }
        chat["updated_at"] = job_status.get("updated_at") or datetime.now().isoformat(timespec="seconds")
        chat.pop("pending_job", None)
        save_llm_chats_to_disk()
    return job_status


@st.fragment(run_every="3s")
def render_llm_pending_job_status(chat_id):
    chat = st.session_state.get("llm_chats", {}).get(chat_id)
    if not chat or not chat.get("pending_job"):
        return

    job_status = reconcile_llm_job(chat) or {"status": "queued"}
    status = job_status.get("status", "queued")
    if status in {"completed", "failed"}:
        st.rerun(scope="app")

    st.info(
        "LLM job is running in the background. "
        f"Status: {status}. "
        "You can switch pages and come back later."
    )


def load_llm_chats_from_disk():
    if not LLM_CHATS_FILE.exists():
        return None

    try:
        with LLM_CHATS_FILE.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception:
        return None

    chats = payload.get("chats", {})
    current_chat_id = payload.get("current_chat_id")
    if not isinstance(chats, dict) or not chats:
        return None

    for chat in chats.values():
        chat.setdefault("messages", [])
        chat.setdefault("memory_summary", "")
        chat.setdefault("last_result", None)
        chat.setdefault("pending_job", None)
        chat.setdefault("title", "Chat")
        chat.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
        chat.setdefault("updated_at", chat["created_at"])

    if current_chat_id not in chats:
        current_chat_id = next(iter(chats))

    return {"chats": chats, "current_chat_id": current_chat_id}


def save_llm_chats_to_disk():
    if "llm_chats" not in st.session_state:
        return

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    chats = {}
    for chat_id, chat in st.session_state["llm_chats"].items():
        saved_chat = dict(chat)
        saved_chat["last_result"] = sanitize_llm_result_for_storage(chat.get("last_result"))
        chats[chat_id] = saved_chat

    with LLM_CHATS_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "current_chat_id": st.session_state.get("current_llm_chat_id"),
                "chats": chats,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


def init_llm_chats():
    if "llm_chats" in st.session_state and "current_llm_chat_id" in st.session_state:
        return

    saved = load_llm_chats_from_disk()
    if saved:
        st.session_state["llm_chats"] = saved["chats"]
        st.session_state["current_llm_chat_id"] = saved["current_chat_id"]
        return

    chat = create_empty_chat(title="Chat 1")
    st.session_state["llm_chats"] = {chat["id"]: chat}
    st.session_state["current_llm_chat_id"] = chat["id"]


def get_current_llm_chat():
    init_llm_chats()
    chat_id = st.session_state["current_llm_chat_id"]
    return st.session_state["llm_chats"][chat_id]


def create_llm_chat():
    chat = create_empty_chat(title=make_chat_title(get_next_chat_number(st.session_state["llm_chats"])))
    st.session_state["llm_chats"][chat["id"]] = chat
    st.session_state["current_llm_chat_id"] = chat["id"]
    save_llm_chats_to_disk()


def clear_current_llm_chat():
    chat = get_current_llm_chat()
    chat["messages"] = []
    chat["memory_summary"] = ""
    chat["last_result"] = None
    chat["pending_job"] = None
    if not chat["title"]:
        chat["title"] = make_chat_title(get_next_chat_number(st.session_state["llm_chats"]))
    chat["updated_at"] = datetime.now().isoformat(timespec="seconds")
    save_llm_chats_to_disk()


def delete_current_llm_chat():
    init_llm_chats()
    chat_id = st.session_state["current_llm_chat_id"]
    if len(st.session_state["llm_chats"]) == 1:
        clear_current_llm_chat()
        return

    del st.session_state["llm_chats"][chat_id]
    st.session_state["current_llm_chat_id"] = next(iter(st.session_state["llm_chats"]))
    save_llm_chats_to_disk()


def build_memory_summary(messages, keep_recent=6, max_lines=10, max_chars=180):
    older_messages = messages[:-keep_recent] if len(messages) > keep_recent else []
    lines = []
    for message in older_messages[-max_lines:]:
        role = "User" if message["role"] == "user" else "Assistant"
        content = " ".join(str(message["content"]).split())
        if len(content) > max_chars:
            content = content[:max_chars].rstrip() + "..."
        lines.append(f"- {role}: {content}")
    return "\n".join(lines)


def get_recent_context_messages(chat, keep_recent=6):
    return [
        {"role": message["role"], "content": message["content"]}
        for message in chat.get("messages", [])[-keep_recent:]
        if message.get("role") in {"user", "assistant"} and message.get("content")
    ]


def render_chat_messages(chat):
    if not chat.get("messages"):
        return

    with st.expander("Chat history", expanded=True):
        for message in chat["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


def render_llm_result(result):
    st.markdown(result["answer"])
    for tool_call in result["tools"]:
        tool_result = tool_call.get("result", {})
        figure_path = tool_result.get("figure_path")
        if figure_path and Path(figure_path).exists():
            st.image(figure_path, use_container_width=True)
    with st.expander("Tool calls", expanded=False):
        st.json(result["tools"])
    if result.get("log_path"):
        st.caption(f"Debug log saved to {result['log_path']}")


def get_filtered_ticker_data(df, ticker, date_range):
    if df.empty:
        return df

    start_date, end_date = date_range
    return df[
        (df["Ticker"] == ticker)
        & (df["Date"] >= pd.to_datetime(start_date))
        & (df["Date"] <= pd.to_datetime(end_date))
    ].copy()


def downsample_for_chart(df, frequency):
    if df.empty or frequency == "Daily":
        return df

    rule = {"Weekly": "W", "Monthly": "ME"}[frequency]
    numeric_columns = ["Close", "MA5", "MA20", "RSI", "MACD", "Volatility"]
    available_columns = [column for column in numeric_columns if column in df.columns]
    return (
        df.set_index("Date")[available_columns]
        .resample(rule)
        .last()
        .dropna(how="all")
        .reset_index()
    )


def show_status_cards(status):
    if not status["available"]:
        st.warning(status["message"])
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{status['rows']:,}")
    col2.metric("Tickers", status["ticker_count"])
    col3.metric("Start", status["start_date"])
    col4.metric("End", status["end_date"])


def show_current_data_snapshot():
    inventory = market_tools.get_local_data_inventory()
    runtime_status = market_tools.get_runtime_storage_status()
    active = inventory["active_analysis_dataset"]
    processed = inventory["processed_dataset"]
    raw_files = inventory["raw_files"]

    with st.expander("Current data snapshot", expanded=True):
        if runtime_status.get("enabled"):
            st.caption(
                "Session runtime storage: "
                f"session {runtime_status['session_id']} | SQLite {runtime_status['database']}"
            )
        else:
            st.caption("Local persistent storage: using project data/, reports/, models/, and config/ folders.")
        active_dataset = active.get("dataset", {})
        st.write(
            "Active app dataset: "
            f"{active['label']} "
            f"({active['processed_file']})."
        )
        if active_dataset.get("available"):
            st.caption(
                f"Active tickers: {', '.join(active_dataset['tickers'])} "
                f"({active_dataset['start_date']} to {active_dataset['end_date']}, "
                f"{active_dataset['rows']:,} rows)."
            )
        else:
            st.caption(active_dataset.get("message", "Active dataset is not available."))
        if active.get("note"):
            st.caption(f"Note: {active['note']}")
        st.divider()

        if processed["available"]:
            st.write(
                "Current processed dataset: "
                f"{', '.join(processed['tickers'])} "
                f"({processed['start_date']} to {processed['end_date']}, "
                f"{processed['rows']:,} rows)."
            )
        else:
            st.info("No processed dataset is currently loaded.")

        raw_tickers = raw_files.get("tickers", [])
        raw_only_tickers = inventory.get("raw_only_tickers", [])
        col1, col2 = st.columns(2)
        col1.metric("Session raw tickers", len(raw_tickers))
        col2.metric("Raw-only tickers", len(raw_only_tickers))

        if raw_only_tickers:
            st.caption("Raw-only tickers are available in this Web session cache but are not part of the current processed dataset.")
            st.write(", ".join(raw_only_tickers))

        raw_records = raw_files.get("records", [])
        if raw_records:
            with st.expander("Raw file details", expanded=False):
                st.dataframe(pd.DataFrame(raw_records), use_container_width=True, height=300)

        workspace = inventory.get("llm_workspace", {})
        workspace_processed = workspace.get("processed_dataset", {})
        st.divider()
        st.write("Current AI chat workspace:")
        st.caption(f"Workspace: {workspace.get('workspace_dir', 'N/A')}")
        if workspace_processed.get("available"):
            st.write(
                f"{', '.join(workspace_processed['tickers'])} "
                f"({workspace_processed['start_date']} to {workspace_processed['end_date']}, "
                f"{workspace_processed['rows']:,} rows)."
            )
        else:
            st.caption("No AI chat workspace processed dataset yet.")

        if runtime_status.get("enabled"):
            st.caption("Raw market data is scoped to this Web session. Chat processed/results are isolated per chat.")
        else:
            st.caption("Raw market data is stored persistently under data/raw/. Chat processed/results are isolated per chat under data/workspaces/.")


def page_data_setup(df):
    st.subheader("Data Setup")
    st.write("Load local cached data when available, or fetch missing market data for selected tickers and dates.")

    status = market_tools.get_dataset_status()
    show_status_cards(status)
    show_current_data_snapshot()

    active_status = market_tools.get_active_analysis_dataset_status()
    if active_status["source"] != "project":
        col_reset, col_merge = st.columns(2)
        if col_reset.button("Show main loaded dataset", use_container_width=True):
            result = market_tools.reset_app_pages_to_project_dataset()
            clear_cache()
            st.success(result["message"])
            st.rerun()
        with col_merge.expander("Developer merge"):
            st.caption("Optional: copy the active AI workspace into the main project dataset.")
            if st.button("Merge active AI workspace", use_container_width=True):
                with st.spinner("Merging AI chat workspace data into the main project dataset..."):
                    try:
                        result = market_tools.merge_llm_workspace_to_project(chat_id=active_status.get("chat_id"))
                    except Exception as exc:
                        st.error(f"Merge failed: {exc}")
                    else:
                        clear_cache()
                        st.success("AI chat workspace data merged into project dataset.")
                        st.json(result)
                        st.rerun()

    with st.expander("Load market data", expanded=False):
        st.session_state.setdefault("symbol_refresh_token", 0)
        col_symbols, col_refresh = st.columns([4, 1])
        ticker_options, symbol_status = get_load_ticker_options(st.session_state["symbol_refresh_token"])
        source_label = symbol_status.get("source", "symbol cache")
        row_count = symbol_status.get("rows", len(ticker_options))
        col_symbols.caption(f"Ticker universe: {row_count:,} symbols from {source_label}. You can still type any yfinance ticker manually.")
        if symbol_status.get("error"):
            col_symbols.warning(f"Symbol list refresh failed; using fallback options. {symbol_status['error']}")
        if col_refresh.button("Refresh symbols", use_container_width=True):
            st.session_state["symbol_refresh_token"] += 1
            st.cache_data.clear()
            st.rerun()

        with st.form("download_form"):
            selected_ticker_values = st.multiselect(
                "Tickers",
                ticker_options,
                default=[],
                accept_new_options=True,
                filter_mode="fuzzy",
                placeholder="Search or type a ticker, then press Enter",
                help="Search common/local tickers from the list, or type any yfinance ticker symbol and press Enter.",
            )
            col1, col2 = st.columns(2)
            start_date = col1.date_input("Start date", value=pd.to_datetime("2015-01-01"))
            end_date = col2.date_input("End date", value=pd.to_datetime("2025-12-31"))
            use_proxy = st.checkbox(
                "Use Clash proxy",
                value=os.getenv("USE_PROXY", "false").lower() in {"1", "true", "yes"},
                help="Only works when the server running this app has CLASH_PROXY configured. Your local Clash cannot be used by Streamlit Community.",
            )
            data_source_label = st.selectbox(
                "Data source",
                ["Auto (use local if valid)", "Online (download from yfinance)"],
                help="Auto reuses valid local raw CSV files first, then downloads only when needed.",
            )
            data_source = "auto" if data_source_label.startswith("Auto") else "download"
            submitted = st.form_submit_button("Load & Process Data", type="primary")

        if submitted:
            selected_tickers = parse_selected_ticker_options(selected_ticker_values)
            if not selected_tickers:
                st.error("Please select at least one ticker.")
            elif start_date >= end_date:
                st.error("Start date must be earlier than end date.")
            else:
                st.info(f"Requested tickers: {', '.join(selected_tickers)}")
                with st.spinner("Refreshing market data, features, EDA artifacts, and baselines..."):
                    try:
                        result = market_tools.refresh_market_data(
                            tickers=selected_tickers,
                            start_date=start_date.isoformat(),
                            end_date=end_date.isoformat(),
                            use_proxy=use_proxy,
                            run_eda_after=True,
                            data_source=data_source,
                        )
                    except Exception as exc:
                        st.error(f"Refresh failed: {exc}")
                    else:
                        market_tools.reset_app_pages_to_project_dataset(note="Activated newly loaded market data.")
                        clear_cache()
                        st.success("Data refresh completed.")
                        st.json(result)
                        st.rerun()

    if not df.empty:
        with st.expander("Processed data preview", expanded=False):
            preview_rows = st.slider("Preview rows", 5, 100, 20)
            st.dataframe(df.tail(preview_rows), use_container_width=True, height=360)


def page_explorer(df):
    st.subheader("Interactive Data Explorer")
    if df.empty:
        st.warning("No processed data available.")
        return

    tickers = sorted(df["Ticker"].unique())
    col1, col2, col3 = st.columns([1, 2, 1])
    ticker = col1.selectbox("Ticker", tickers)
    date_range = col2.date_input(
        "Date range",
        value=(df["Date"].min().date(), df["Date"].max().date()),
    )
    frequency = col3.selectbox("Chart frequency", ["Monthly", "Weekly", "Daily"])

    if not isinstance(date_range, tuple) or len(date_range) != 2:
        st.info("Select both start and end dates.")
        return

    ticker_df = get_filtered_ticker_data(df, ticker, date_range)
    chart_df = downsample_for_chart(ticker_df, frequency)

    if chart_df.empty:
        st.warning("No rows match the selected filters.")
        return

    chart_columns = [column for column in ["Close", "MA5", "MA20"] if column in chart_df.columns]
    st.line_chart(chart_df.set_index("Date")[chart_columns])

    with st.expander("Show filtered rows", expanded=False):
        st.dataframe(ticker_df.tail(50), use_container_width=True, height=360)


def page_eda():
    st.subheader("EDA Results")

    active_status = market_tools.get_active_analysis_dataset_status()
    st.caption(f"Using {active_status['label']}: {active_status['processed_file']}")
    eda_summary = load_summary(active_status["eda_summary_file"])
    quality_summary = load_summary(active_status["data_quality_file"])
    tab1, tab2, tab3 = st.tabs(["Asset Summary", "Data Quality", "Figures"])

    with tab1:
        if eda_summary.empty:
            st.info("EDA summary is not available yet.")
        else:
            st.dataframe(eda_summary, use_container_width=True, height=420)

    with tab2:
        if quality_summary.empty:
            st.info("Data quality summary is not available yet.")
        else:
            st.dataframe(quality_summary, use_container_width=True, height=420)

    with tab3:
        figure_result = market_tools.list_available_figures(data_scope="active")
        figures = figure_result["figures"]
        if not figures:
            st.info("No generated figures found.")
            return

        figure_name = st.selectbox("Figure", figures)
        figure_path = Path(figure_result["directory"]) / figure_name
        st.image(str(figure_path), use_container_width=True)


def page_ai_assistant():
    st.subheader("LLM Natural Language Analysis")
    st.write(
        "Choose an OpenAI-compatible provider or local model server. "
        "The model can call local project tools to answer questions."
    )

    init_llm_chats()
    preferences = load_llm_preferences()
    initialize_llm_ui_state(preferences)
    prepare_llm_widget_state()
    chats = st.session_state["llm_chats"]
    chat_column, main_column = st.columns([1.1, 3.2], gap="large")

    with chat_column:
        st.markdown("**Chats**")
        if st.button("New chat", use_container_width=True):
            create_llm_chat()
            st.rerun()

        sorted_chat_ids = sorted(chats, key=lambda chat_id: chats[chat_id]["updated_at"], reverse=True)
        for chat_id in sorted_chat_ids:
            chat_item = chats[chat_id]
            is_current = chat_id == st.session_state["current_llm_chat_id"]
            label = chat_item["title"] or "Untitled chat"
            if st.button(label, key=f"chat_select_{chat_id}", use_container_width=True, type="primary" if is_current else "secondary"):
                st.session_state["current_llm_chat_id"] = chat_id
                st.rerun()

        chat = get_current_llm_chat()
        new_title = st.text_input("Rename current chat", value=chat["title"], key=f"rename_{chat['id']}")
        if st.button("Rename", use_container_width=True):
            chat["title"] = new_title.strip() or chat["title"]
            chat["updated_at"] = datetime.now().isoformat(timespec="seconds")
            save_llm_chats_to_disk()
            st.rerun()

        col_clear, col_delete = st.columns(2)
        if col_clear.button("Clear", use_container_width=True):
            clear_current_llm_chat()
            st.rerun()
        if col_delete.button("Delete", use_container_width=True):
            delete_current_llm_chat()
            st.rerun()

    with main_column:
        chat = get_current_llm_chat()
        job_status = reconcile_llm_job(chat)
        if job_status and job_status.get("status") == "completed":
            st.success("LLM answer is ready.")
        elif job_status and job_status.get("status") == "failed":
            st.error(f"LLM call failed: {job_status.get('error', 'Unknown error')}")

        provider_names = list(PROVIDER_PRESETS.keys())
        preferred_provider = st.session_state.get("llm_provider_name", preferences.get("provider_name", "Alibaba Bailian"))
        provider_index = provider_names.index(preferred_provider) if preferred_provider in provider_names else 0
        provider_name = st.selectbox("Provider", provider_names, index=provider_index)
        st.session_state["llm_provider_name"] = provider_name
        apply_provider_change_if_needed(preferences)
        provider_name = st.session_state["llm_provider_name"]
        preset = PROVIDER_PRESETS[provider_name]
        use_saved_preferences = provider_name == preferences.get("provider_name")

        col1, col2 = st.columns(2)
        base_url = col1.text_input("Base URL", key="llm_widget_base_url")
        st.session_state["llm_base_url"] = base_url
        if provider_name in {"Ollama Local", "LM Studio Local"}:
            api_key = col2.text_input("API Key (optional for local providers)", type="password", key="llm_widget_api_key")
        else:
            api_key = col2.text_input("API Key", type="password", key="llm_widget_api_key")
        st.session_state["llm_api_key"] = api_key

        if provider_name == "Ollama Local":
            ollama_models = cached_ollama_models(base_url)
            if ollama_models["available"] and ollama_models["models"]:
                preferred_model = st.session_state.get("llm_model") or ((preferences.get("model") if use_saved_preferences else None) or preset["model"])
                model_index = ollama_models["models"].index(preferred_model) if preferred_model in ollama_models["models"] else 0
                model = st.selectbox("Model", ollama_models["models"], index=model_index, key="llm_widget_model_select")
            else:
                st.session_state.setdefault("llm_widget_model_text", st.session_state.get("llm_model"))
                model = st.text_input("Model", key="llm_widget_model_text")
                st.caption(f"Ollama model discovery failed: {ollama_models['error']}")
        else:
            state_key = get_model_state_key(provider_name, base_url)
            col_load, col_hint = st.columns([1.15, 4.2], vertical_alignment="center")
            if col_load.button("Load models", use_container_width=True):
                if provider_name not in {"LM Studio Local"} and not api_key:
                    st.warning("Enter an API key before loading remote provider models.")
                else:
                    st.session_state[state_key] = list_openai_compatible_models(base_url, api_key)

            model_result = st.session_state.get(state_key)
            if model_result and model_result["available"]:
                model_options = model_result["models"]
                preferred_model = st.session_state.get("llm_model") or ((preferences.get("model") if use_saved_preferences else None) or preset["model"])
                default_index = model_options.index(preferred_model) if preferred_model in model_options else 0
                model = st.selectbox("Model", model_options, index=default_index, key="llm_widget_model_select")
                col_hint.caption("Model list loaded from the provider's OpenAI-compatible /models endpoint.")
            elif model_result and model_result["error"]:
                st.session_state.setdefault("llm_widget_model_text", st.session_state.get("llm_model"))
                model = st.text_input("Model", key="llm_widget_model_text")
                col_hint.caption(f"Model discovery unavailable: {model_result['error']}")
            else:
                st.session_state.setdefault("llm_widget_model_text", st.session_state.get("llm_model"))
                model = st.text_input("Model", key="llm_widget_model_text")
        st.session_state["llm_model"] = model

        with st.expander("Advanced execution settings", expanded=False):
            limit_tool_rounds = st.checkbox("Limit tool-call rounds", key="llm_widget_limit_tool_rounds")
            st.session_state["llm_limit_tool_rounds"] = limit_tool_rounds
            max_tool_rounds = 0
            if limit_tool_rounds:
                max_tool_rounds = st.number_input(
                    "Maximum tool-call rounds",
                    min_value=1,
                    max_value=50,
                    step=1,
                    key="llm_widget_max_tool_rounds",
                )
                st.session_state["llm_max_tool_rounds"] = max_tool_rounds

            limit_total_runtime = st.checkbox("Limit total runtime", key="llm_widget_limit_total_runtime")
            st.session_state["llm_limit_total_runtime"] = limit_total_runtime
            max_elapsed_seconds = 0
            if limit_total_runtime:
                max_elapsed_seconds = st.number_input(
                    "Maximum total runtime (sec)",
                    min_value=30,
                    max_value=3600,
                    step=30,
                    key="llm_widget_max_elapsed_seconds",
                )
                st.session_state["llm_max_elapsed_seconds"] = max_elapsed_seconds

            request_timeout = st.number_input(
                "Single API request timeout (sec)",
                min_value=5,
                max_value=600,
                step=5,
                key="llm_widget_request_timeout",
                help="Network timeout for each LLM API request. This is still finite to avoid a frozen request.",
            )
            st.session_state["llm_request_timeout"] = request_timeout
        save_debug_log = st.checkbox("Save debug log", key="llm_widget_save_debug_log")
        st.session_state["llm_save_debug_log"] = save_debug_log
        remember_api_key = st.checkbox(
            "Remember API key locally",
            key="llm_widget_remember_api_key",
            help=f"Stores the key as plain text in {LLM_PREFERENCES_FILE}. Leave off on shared machines.",
        )
        st.session_state["llm_remember_api_key"] = remember_api_key
        if st.button("Save LLM settings"):
            save_llm_preferences(
                {
                    "provider_name": provider_name,
                    "base_url": base_url,
                    "model": model,
                    "remember_api_key": bool(remember_api_key),
                    "api_key": api_key if remember_api_key else "",
                }
            )
            st.success("LLM settings saved locally.")

        allow_write_tools = True
        if st.session_state.get("storage_mode") == "session":
            st.caption("AI analysis outputs are written to the current Chat workspace. Raw market data is cached only inside this Web session.")
        else:
            st.caption("AI analysis outputs are written to the current Chat workspace. Raw market data is stored persistently under data/raw/.")

        chat["memory_summary"] = build_memory_summary(chat.get("messages", []))
        if chat["memory_summary"]:
            with st.expander("Context memory summary", expanded=False):
                st.write(chat["memory_summary"])
        render_chat_messages(chat)
        pending_job = chat.get("pending_job")
        if pending_job:
            render_llm_pending_job_status(chat["id"])

        question = st.text_area("Question", height=120, key=f"question_{chat['id']}")

        if st.button("Ask LLM", type="primary", disabled=bool(pending_job)):
            if not api_key and provider_name not in {"Ollama Local", "LM Studio Local"}:
                st.error("Please enter your API key.")
            elif not question.strip():
                st.error("Please enter a question.")
            else:
                start_llm_background_job(
                    chat=chat,
                    question=question,
                    llm_kwargs={
                        "api_key": api_key,
                        "provider_name": provider_name,
                        "model": model,
                        "base_url": base_url,
                        "memory_summary": chat.get("memory_summary"),
                        "context_messages": get_recent_context_messages(chat),
                        "allow_write_tools": allow_write_tools,
                        "max_tool_rounds": max_tool_rounds,
                        "request_timeout": request_timeout,
                        "max_elapsed_seconds": max_elapsed_seconds,
                        "save_debug_log": save_debug_log,
                    },
                )
                st.success("LLM job started in the background. You can switch pages now.")
                st.rerun()
        elif chat.get("last_result"):
            st.divider()
            st.subheader("Last LLM Answer")
            render_llm_result(chat["last_result"])


def page_baselines():
    st.subheader("Baseline Strategy Results")
    st.write("Current strategies: Buy & Hold, Moving Average crossover, RSI threshold strategy, and lightweight Portfolio CEM.")

    active_status = market_tools.get_active_analysis_dataset_status()
    active_source = active_status["source"]
    active_is_workspace = active_source in {"llm_workspace", "chat_workspace"}
    active_chat_id = active_status.get("chat_id")
    workspace_paths = market_tools.get_chat_workspace_paths(active_chat_id)
    data_scope = "workspace" if active_is_workspace else "project"
    st.caption(f"Using {active_status['label']}: {active_status['processed_file']}")

    st.markdown("**Unified Strategy Comparison**")
    comparison_file = (
        workspace_paths["strategy_comparison_file"]
        if active_is_workspace
        else market_tools.STRATEGY_COMPARISON_FILE
    )
    comparison = load_summary(comparison_file)
    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("Build comparison"):
            with st.spinner("Building strategy comparison..."):
                result = market_tools.run_strategy_comparison(data_scope=data_scope, chat_id=active_chat_id)
                clear_cache()
                st.success("Strategy comparison updated.")
                st.json(result)
    with col_b:
        if comparison.empty:
            st.info("Strategy comparison is not available yet. Build it after baseline metrics exist.")
        else:
            st.caption("Portfolio CEM is portfolio-level; the other rows are single-asset strategy results.")
            st.dataframe(comparison, use_container_width=True, height=240)

    st.divider()

    st.markdown("**Portfolio RL Training**")
    rl_metrics_file = (
        workspace_paths["portfolio_rl_metrics_file"]
        if active_is_workspace
        else market_tools.PORTFOLIO_RL_METRICS_FILE
    )
    rl_metrics = load_summary(rl_metrics_file)
    rl_col_a, rl_col_b = st.columns([1, 3])
    with rl_col_a:
        if st.button("Run Portfolio CEM"):
            with st.spinner("Training lightweight portfolio policy..."):
                result = market_tools.run_portfolio_cem_training(
                    data_scope=data_scope,
                    chat_id=active_chat_id,
                )
                clear_cache()
                st.success("Portfolio CEM training completed.")
                st.json(result)
    with rl_col_b:
        if rl_metrics.empty:
            st.info("Portfolio RL metrics are not available yet.")
        else:
            st.dataframe(rl_metrics, use_container_width=True, height=160)
            rl_equity_result = market_tools.get_portfolio_rl_equity_curve(
                max_rows=5000,
                data_scope=data_scope,
                chat_id=active_chat_id,
            )
            if rl_equity_result["available"]:
                rl_equity_df = pd.DataFrame(rl_equity_result["records"])
                rl_equity_df["Date"] = pd.to_datetime(rl_equity_df["Date"])
                st.line_chart(rl_equity_df.set_index("Date")[["Portfolio_Value"]])

    st.divider()

    strategy_options = {
        "Buy & Hold": {
            "metrics_file": workspace_paths["buy_hold_metrics_file"] if active_is_workspace else market_tools.BUY_HOLD_METRICS_FILE,
            "run": market_tools.run_buy_hold_baseline,
            "equity": market_tools.get_buy_hold_equity_curve,
        },
        "Moving Average": {
            "metrics_file": workspace_paths["ma_metrics_file"] if active_is_workspace else market_tools.MA_METRICS_FILE,
            "run": market_tools.run_ma_baseline,
            "equity": market_tools.get_ma_equity_curve,
        },
        "RSI": {
            "metrics_file": workspace_paths["rsi_metrics_file"] if active_is_workspace else market_tools.RSI_METRICS_FILE,
            "run": market_tools.run_rsi_baseline,
            "equity": market_tools.get_rsi_equity_curve,
        },
    }
    strategy_name = st.selectbox("Strategy", list(strategy_options.keys()))
    strategy = strategy_options[strategy_name]

    metrics = load_summary(strategy["metrics_file"])
    if metrics.empty:
        st.info(f"{strategy_name} results are not available yet.")
        if st.button(f"Run {strategy_name} baseline"):
            with st.spinner(f"Running {strategy_name} baseline..."):
                result = strategy["run"](data_scope=data_scope, chat_id=active_chat_id)
                clear_cache()
                st.success(f"{strategy_name} baseline completed.")
                st.json(result)
        return

    st.dataframe(metrics, use_container_width=True, height=360)

    tickers = sorted(metrics["Ticker"].unique())
    ticker = st.selectbox("Equity curve ticker", tickers)
    equity_result = strategy["equity"](ticker, max_rows=5000, data_scope=data_scope, chat_id=active_chat_id)
    if not equity_result["available"]:
        st.warning(equity_result["message"])
        return

    equity_df = pd.DataFrame(equity_result["records"])
    equity_df["Date"] = pd.to_datetime(equity_df["Date"])
    st.line_chart(equity_df.set_index("Date")[["Portfolio_Value"]])


def format_bytes(size_bytes):
    size = float(size_bytes or 0)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024


def page_exports():
    st.subheader("Export Artifacts")
    st.write("Create a ZIP package with generated data, figures, strategy results, and portfolio policy files.")

    active_status = market_tools.get_active_analysis_dataset_status()
    current_chat = get_current_llm_chat()
    scope_options = {
        "Active app dataset": "active",
        "Main project dataset": "project",
        "Current AI chat workspace": "workspace",
    }

    col_scope, col_name = st.columns([1.2, 2])
    scope_label = col_scope.selectbox("Export scope", list(scope_options.keys()))
    export_name = col_name.text_input(
        "Export file name",
        value=f"finrl_insight_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    data_scope = scope_options[scope_label]
    export_chat_id = current_chat["id"] if data_scope == "workspace" else active_status.get("chat_id")

    col1, col2, col3, col4, col5 = st.columns(5)
    include_data = col1.checkbox("Processed data", value=True)
    include_raw_data = col2.checkbox("Raw data", value=False)
    include_figures = col3.checkbox("Figures", value=True)
    include_results = col4.checkbox("Strategy results", value=True)
    include_models = col5.checkbox("Models", value=True)
    include_research = st.checkbox("Research reports", value=True)
    col_filter, col_query = st.columns([1, 2])
    category_filter_label = col_filter.selectbox(
        "Artifact type",
        ["All", "data", "raw_data", "figures", "results", "models", "research"],
    )
    artifact_query = col_query.text_input(
        "Filter by extraction code or file name",
        placeholder="e.g. PRICE, STRATEGY_COMPARISON, TSLA",
    )
    category_filter = None if category_filter_label == "All" else category_filter_label

    inventory = market_tools.list_exportable_artifacts(
        data_scope=data_scope,
        chat_id=export_chat_id,
        include_data=include_data,
        include_raw_data=include_raw_data,
        include_figures=include_figures,
        include_strategy_results=include_results,
        include_models=include_models,
        include_research=include_research,
        category=category_filter,
        query=artifact_query,
    )
    st.caption(
        f"Scope: {inventory['data_scope']} | "
        f"Files: {inventory['file_count']} | "
        f"Size: {format_bytes(inventory['total_size_bytes'])}"
    )

    if inventory["records"]:
        option_labels = {
            record["artifact_code"]: (
                f"{record['artifact_code']} | {record['category']} | "
                f"{record['name']} | {format_bytes(record['size_bytes'])}"
            )
            for record in inventory["records"]
        }
        selected_codes = st.multiselect(
            "Precise extraction codes",
            list(option_labels.keys()),
            default=list(option_labels.keys()),
            format_func=lambda code: option_labels[code],
            help="Use these codes to export one specific image, dataset, strategy result, or model artifact.",
        )
        with st.expander("Files to export", expanded=False):
            st.dataframe(pd.DataFrame(inventory["records"]), use_container_width=True, height=300)
    else:
        selected_codes = []
        st.info(inventory["message"])

    if st.button("Create export ZIP", type="primary", disabled=not selected_codes):
        result = market_tools.export_analysis_artifacts(
            data_scope=data_scope,
            chat_id=export_chat_id,
            export_name=export_name,
            include_data=include_data,
            include_raw_data=include_raw_data,
            include_figures=include_figures,
            include_strategy_results=include_results,
            include_models=include_models,
            include_research=include_research,
            artifact_codes=selected_codes,
            category=category_filter,
            query=artifact_query,
        )
        if not result["exported"]:
            st.warning(result["message"])
        else:
            st.session_state["last_export_result"] = result
            st.success(f"Export package created: {Path(result['export_file']).name}")

    export_result = st.session_state.get("last_export_result")
    if export_result and Path(export_result["export_file"]).exists():
        export_path = Path(export_result["export_file"])
        st.download_button(
            "Download latest export",
            data=export_path.read_bytes(),
            file_name=export_path.name,
            mime="application/zip",
            use_container_width=True,
        )
        st.caption(f"Saved at: {export_path}")


def page_research_agent():
    st.subheader("Web Research Agent")
    st.write("Generate a lightweight research report with fundamentals, broad market context, and recent Yahoo Finance news sources.")

    active_status = market_tools.get_active_analysis_dataset_status()
    current_chat = get_current_llm_chat()
    scope_options = {
        "Active app dataset": "active",
        "Main project research folder": "project",
        "Current AI chat workspace": "workspace",
    }
    col_query, col_tickers = st.columns([2, 1])
    query = col_query.text_input("Research question", value="Analyze semiconductor market leaders")
    ticker_text = col_tickers.text_input("Tickers", value="NVDA, AMD, QQQ")
    data_scope_label = st.selectbox("Save report to", list(scope_options.keys()))
    data_scope = scope_options[data_scope_label]
    research_chat_id = current_chat["id"] if data_scope == "workspace" else active_status.get("chat_id")

    col_a, col_b, col_c, col_d = st.columns(4)
    include_fundamentals = col_a.checkbox("Fundamentals", value=True)
    include_macro = col_b.checkbox("Macro snapshot", value=True)
    include_news = col_c.checkbox("Recent news", value=True)
    news_limit = col_d.number_input("News per ticker", min_value=1, max_value=10, value=5, step=1)
    macro_period = st.selectbox("Macro lookback", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)

    if st.button("Run research", type="primary"):
        with st.spinner("Collecting fundamentals, macro context, and source links..."):
            try:
                result = market_tools.run_web_research_agent(
                    query=query,
                    tickers=market_tools.parse_ticker_input(ticker_text),
                    include_fundamentals=include_fundamentals,
                    include_macro=include_macro,
                    include_news=include_news,
                    news_limit=int(news_limit),
                    macro_period=macro_period,
                    data_scope=data_scope,
                    chat_id=research_chat_id,
                )
            except Exception as exc:
                st.error(f"Research failed: {exc}")
            else:
                st.session_state["last_research_result"] = result
                st.success("Research report created.")

    result = st.session_state.get("last_research_result")
    if not result:
        return

    st.caption(f"Report: {result.get('markdown_file')}")
    fundamentals = pd.DataFrame(result.get("fundamentals", {}).get("records", []))
    if not fundamentals.empty:
        st.markdown("**Fundamental Snapshot**")
        st.dataframe(fundamentals, use_container_width=True, height=260)

    macro = pd.DataFrame(result.get("macro", {}).get("records", []))
    if not macro.empty:
        st.markdown("**Macro Market Snapshot**")
        st.dataframe(macro, use_container_width=True, height=260)

    news = result.get("news", {}).get("records", [])
    if news:
        st.markdown("**Recent News Sources**")
        for item in news[:20]:
            title = item.get("title") or "Untitled"
            url = item.get("url") or ""
            st.markdown(f"- `{item.get('ticker')}` [{title}]({url})")

    markdown_file = result.get("markdown_file")
    if markdown_file and Path(markdown_file).exists():
        st.download_button(
            "Download Markdown report",
            data=Path(markdown_file).read_text(encoding="utf-8"),
            file_name=Path(markdown_file).name,
            mime="text/markdown",
            use_container_width=True,
        )


def page_tool_api_preview():
    st.subheader("LLM Tool API Preview")
    st.write("These are the Python functions that can later be registered as ChatGPT tools.")

    st.code(
        """
get_dataset_status()
get_project_capabilities()
get_runtime_storage_status()
list_tool_proposals(limit=50, status=None)
list_temp_composite_tools()
get_tool_promotion_status()
list_local_raw_data()
get_local_data_inventory()
get_llm_workspace_status(chat_id=None)
get_active_analysis_dataset_status()
search_us_symbols(query="", limit=500, force_refresh=False)
validate_ticker_candidates(query, candidates)
inspect_local_raw_ticker(ticker, start_date=None, end_date=None)
get_eda_summary()
get_data_quality_summary()
list_available_figures()
get_ticker_history(ticker, start_date=None, end_date=None, max_rows=500)
create_ticker_price_chart(ticker, start_date=None, end_date=None, months=12)
get_ticker_metrics(ticker)
screen_stock_candidates(query="", max_candidates=300, shortlist_size=10, lookback_period="1y")
prepare_ticker_analysis(tickers, start_date=None, end_date=None, lookback_period="1y")
analyze_theme_candidates(query, lookback_period="1y", shortlist_size=5)
propose_new_tool(tool_name, user_need, inputs=None, outputs=None)
register_temp_composite_tool(tool_name, base_tool, preset_arguments=None, filters=None)
promote_tool_proposal_local(proposal_id=None, proposal_file=None)
load_shortlist_for_analysis(tickers, start_date, end_date, chat_id=None)
refresh_llm_workspace_data(tickers, start_date, end_date, data_source="auto", use_proxy=False, chat_id=None)
refresh_llm_workspace_ticker(ticker, start_date, end_date, data_source="auto", use_proxy=False, chat_id=None)
push_llm_workspace_to_app_pages(note=None, chat_id=None)
reset_app_pages_to_project_dataset(note=None)
merge_llm_workspace_to_project(tickers=None, chat_id=None)
run_buy_hold_baseline(tickers=None, initial_cash=100000)
get_buy_hold_metrics(ticker=None)
get_buy_hold_equity_curve(ticker, max_rows=500)
run_ma_baseline(tickers=None, initial_cash=100000, short_window=5, long_window=20)
get_ma_metrics(ticker=None)
get_ma_equity_curve(ticker, max_rows=500)
run_rsi_baseline(tickers=None, initial_cash=100000, buy_threshold=30, sell_threshold=70)
get_rsi_metrics(ticker=None)
get_rsi_equity_curve(ticker, max_rows=500)
run_strategy_comparison(data_scope="project")
get_strategy_comparison(ticker=None, data_scope="auto")
run_portfolio_env_smoke_test(tickers=None, data_scope="auto", max_steps=5)
run_portfolio_cem_training(tickers=None, data_scope="auto", generations=4, population_size=12)
get_portfolio_rl_metrics(data_scope="auto")
get_portfolio_rl_equity_curve(max_rows=1000, data_scope="auto")
get_fundamental_snapshot(["AAPL", "MSFT"])
get_macro_market_snapshot(period="6mo")
get_market_news(["AAPL"], limit_per_ticker=5)
run_web_research_agent("Analyze Apple fundamentals and market context", tickers=["AAPL"])
list_exportable_artifacts(data_scope="active")
export_analysis_artifacts(data_scope="active", export_name=None)
export_selected_artifacts(["FIGURES_PRICE_CHART_TSLA"], data_scope="active")
        """.strip(),
        language="python",
    )

    status = market_tools.get_dataset_status()
    st.json(status)


def page_tool_proposals():
    st.subheader("Tool Proposals")
    st.write("LLM-generated tool ideas are saved here for developer review. They are not executed automatically.")
    st.caption("Local developer promotion command: python scripts/promote_tool_proposal.py --proposal-file <proposal.json>")
    promotion_status = market_tools.get_tool_promotion_status()
    if promotion_status["enabled"]:
        st.success("Local LLM-driven promotion is enabled for this runtime.")
    else:
        st.info("Local LLM-driven promotion is disabled. Set ENABLE_LOCAL_TOOL_PROMOTION=true when running locally to allow it.")

    proposals = market_tools.list_tool_proposals(limit=100)
    temp_tools = market_tools.list_temp_composite_tools()
    with st.expander("Temporary composite tools in this session", expanded=False):
        if temp_tools["records"]:
            st.dataframe(pd.DataFrame(temp_tools["records"]), use_container_width=True)
        else:
            st.info("No temporary composite tools registered yet.")

    st.caption(f"Directory: {proposals['directory']}")
    if not proposals["records"]:
        st.info("No tool proposals yet.")
        return

    for proposal in proposals["records"]:
        title = f"{proposal.get('display_name') or proposal.get('tool_name')} | {proposal.get('created_at')}"
        with st.expander(title, expanded=False):
            st.write(proposal.get("user_need", ""))
            col1, col2 = st.columns(2)
            col1.write("**Inputs**")
            col1.write(proposal.get("inputs", []))
            col2.write("**Outputs**")
            col2.write(proposal.get("outputs", []))
            st.write("**Required data**")
            st.write(proposal.get("required_data", []))
            st.write("**Implementation plan**")
            st.write(proposal.get("implementation_plan", []))
            st.write("**Safety notes**")
            st.write(proposal.get("safety_notes", []))
            if proposal.get("suggested_python_code"):
                st.code(proposal["suggested_python_code"], language="python")
            if proposal.get("suggested_tool_schema"):
                st.json(proposal["suggested_tool_schema"])
            st.caption(f"File: {proposal.get('file_path')}")


st.title("Intelligent Financial Market Analysis")
st.caption("Lightweight Streamlit demo with tool-ready backend functions.")

active_dataset_status = market_tools.get_active_analysis_dataset_status()
df = load_processed_data(
    active_dataset_status["processed_file"],
    active_dataset_status.get("updated_at"),
)

with st.sidebar:
    page_options = [
        "Data Setup",
        "Data Explorer",
        "EDA Results",
        "Baseline Results",
        "Research Agent",
        "Export Artifacts",
        "AI Assistant",
    ]
    show_developer_tools = st.checkbox("Developer tools", value=False)
    if show_developer_tools:
        page_options.extend(["LLM Tool API Preview", "Tool Proposals"])

    page = st.radio(
        "Page",
        page_options,
    )

if page == "Data Setup":
    page_data_setup(df)
elif page == "Data Explorer":
    page_explorer(df)
elif page == "EDA Results":
    page_eda()
elif page == "Baseline Results":
    page_baselines()
elif page == "Research Agent":
    page_research_agent()
elif page == "Export Artifacts":
    page_exports()
elif page == "AI Assistant":
    page_ai_assistant()
elif page == "LLM Tool API Preview":
    page_tool_api_preview()
else:
    page_tool_proposals()

st.divider()
if st.session_state.get("storage_mode") == "session":
    st.caption("Baseline strategies, portfolio environment checks, and lightweight RL results are saved as session runtime artifacts.")
else:
    st.caption("Baseline strategies, portfolio environment checks, and lightweight RL results are saved under the local reports/ and models/ folders.")
