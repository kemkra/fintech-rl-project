from datetime import datetime
import json
from pathlib import Path
import sys
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
    active = inventory["active_analysis_dataset"]
    processed = inventory["processed_dataset"]
    raw_files = inventory["raw_files"]

    with st.expander("Current data snapshot", expanded=True):
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
        col1.metric("Raw tickers", len(raw_tickers))
        col2.metric("Raw-only tickers", len(raw_only_tickers))

        if raw_only_tickers:
            st.caption("Raw-only tickers are available in data/raw but are not part of the current processed dataset.")
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

        st.caption("Raw data is shared in data/raw and is not duplicated per chat.")


def page_data_setup(df):
    st.subheader("Data Setup")
    st.write("Load local cached data when available, or fetch missing market data for selected tickers and dates.")

    status = market_tools.get_dataset_status()
    show_status_cards(status)
    show_current_data_snapshot()

    active_status = market_tools.get_active_analysis_dataset_status()
    active_chat_id = active_status.get("chat_id")
    col_merge, col_clear = st.columns(2)
    if col_merge.button("Merge AI chat workspace into project", use_container_width=True):
        with st.spinner("Merging AI chat workspace data into the main project dataset..."):
            try:
                result = market_tools.merge_llm_workspace_to_project(chat_id=active_chat_id)
            except Exception as exc:
                st.error(f"Merge failed: {exc}")
            else:
                clear_cache()
                st.success("AI chat workspace data merged into project dataset.")
                st.json(result)
                st.rerun()
    if col_clear.button("Clear AI chat workspace", use_container_width=True):
        result = market_tools.clear_llm_workspace(chat_id=active_chat_id)
        clear_cache()
        st.success(result["message"])
        st.rerun()

    if active_status["source"] != "project":
        if st.button("Reset pages to main project dataset", use_container_width=True):
            result = market_tools.reset_app_pages_to_project_dataset()
            clear_cache()
            st.success(result["message"])
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
                default=[
                    option for option in ticker_options
                    if option.split("|", 1)[0].strip() in {"AAPL", "MSFT", "QQQ", "SPY"}
                ][:4],
                accept_new_options=True,
                filter_mode="fuzzy",
                placeholder="Search or type a ticker, then press Enter",
                help="Search common/local tickers from the list, or type any yfinance ticker symbol and press Enter.",
            )
            col1, col2 = st.columns(2)
            start_date = col1.date_input("Start date", value=pd.to_datetime("2015-01-01"))
            end_date = col2.date_input("End date", value=pd.to_datetime("2025-12-31"))
            use_proxy = st.checkbox("Use Clash proxy", value=True)
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
                        clear_cache()
                        st.success("Data refresh completed.")
                        st.json(result)

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

    eda_summary = load_summary(market_tools.EDA_SUMMARY_FILE)
    quality_summary = load_summary(market_tools.DATA_QUALITY_FILE)
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
        figures = market_tools.list_available_figures()["figures"]
        if not figures:
            st.info("No generated figures found.")
            return

        figure_name = st.selectbox("Figure", figures)
        figure_path = market_tools.FIGURES_DIR / figure_name
        st.image(str(figure_path), use_container_width=True)


def page_ai_assistant():
    st.subheader("LLM Natural Language Analysis")
    st.write(
        "Choose an OpenAI-compatible provider or local model server. "
        "The model can call local project tools to answer questions."
    )

    init_llm_chats()
    preferences = load_llm_preferences()
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

        provider_names = list(PROVIDER_PRESETS.keys())
        preferred_provider = preferences.get("provider_name", "Alibaba Bailian")
        provider_index = provider_names.index(preferred_provider) if preferred_provider in provider_names else 0
        provider_name = st.selectbox("Provider", provider_names, index=provider_index)
        preset = PROVIDER_PRESETS[provider_name]
        use_saved_preferences = provider_name == preferences.get("provider_name")

        col1, col2 = st.columns(2)
        preferred_base_url = preferences.get("base_url") if use_saved_preferences else None
        preferred_base_url = preferred_base_url or preset["base_url"]
        base_url = col1.text_input("Base URL", value=preferred_base_url)
        remembered_api_key = preferences.get("api_key", "") if use_saved_preferences else ""
        if provider_name in {"Ollama Local", "LM Studio Local"}:
            api_key = col2.text_input("API Key (optional for local providers)", value=remembered_api_key, type="password")
        else:
            api_key = col2.text_input("API Key", value=remembered_api_key, type="password")

        if provider_name == "Ollama Local":
            ollama_models = cached_ollama_models(base_url)
            if ollama_models["available"] and ollama_models["models"]:
                preferred_model = preferences.get("model") if use_saved_preferences else None
                preferred_model = preferred_model or preset["model"]
                model_index = ollama_models["models"].index(preferred_model) if preferred_model in ollama_models["models"] else 0
                model = st.selectbox("Model", ollama_models["models"], index=model_index)
            else:
                preferred_model = preferences.get("model") if use_saved_preferences else None
                model = st.text_input("Model", value=preferred_model or preset["model"])
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
                preferred_model = preferences.get("model") if use_saved_preferences else None
                preferred_model = preferred_model or preset["model"]
                default_index = model_options.index(preferred_model) if preferred_model in model_options else 0
                model = st.selectbox("Model", model_options, index=default_index)
                col_hint.caption("Model list loaded from the provider's OpenAI-compatible /models endpoint.")
            elif model_result and model_result["error"]:
                preferred_model = preferences.get("model") if use_saved_preferences else None
                model = st.text_input("Model", value=preferred_model or preset["model"])
                col_hint.caption(f"Model discovery unavailable: {model_result['error']}")
            else:
                preferred_model = preferences.get("model") if use_saved_preferences else None
                model = st.text_input("Model", value=preferred_model or preset["model"])

        with st.expander("Advanced execution settings", expanded=False):
            limit_tool_rounds = st.checkbox("Limit tool-call rounds", value=False)
            max_tool_rounds = 0
            if limit_tool_rounds:
                max_tool_rounds = st.number_input(
                    "Maximum tool-call rounds",
                    min_value=1,
                    max_value=50,
                    value=5,
                    step=1,
                )

            limit_total_runtime = st.checkbox("Limit total runtime", value=False)
            max_elapsed_seconds = 0
            if limit_total_runtime:
                max_elapsed_seconds = st.number_input(
                    "Maximum total runtime (sec)",
                    min_value=30,
                    max_value=3600,
                    value=180,
                    step=30,
                )

            request_timeout = st.number_input(
                "Single API request timeout (sec)",
                min_value=5,
                max_value=600,
                value=120,
                step=5,
                help="Network timeout for each LLM API request. This is still finite to avoid a frozen request.",
            )
        save_debug_log = st.checkbox("Save debug log", value=True)
        remember_api_key = st.checkbox(
            "Remember API key locally",
            value=bool(preferences.get("remember_api_key")),
            help=f"Stores the key as plain text in {LLM_PREFERENCES_FILE}. Leave off on shared machines.",
        )
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
        st.caption("AI analysis outputs are written to the current Chat workspace. Raw market data is shared in data/raw.")

        chat["memory_summary"] = build_memory_summary(chat.get("messages", []))
        if chat["memory_summary"]:
            with st.expander("Context memory summary", expanded=False):
                st.write(chat["memory_summary"])
        render_chat_messages(chat)

        question = st.text_area("Question", height=120, key=f"question_{chat['id']}")

        if st.button("Ask LLM", type="primary"):
            if not api_key and provider_name not in {"Ollama Local", "LM Studio Local"}:
                st.error("Please enter your API key.")
            elif not question.strip():
                st.error("Please enter a question.")
            else:
                with st.spinner("Calling LLM and local tools..."):
                    try:
                        result = run_llm_tool_chat(
                            question=question,
                            api_key=api_key,
                            provider_name=provider_name,
                            model=model,
                            base_url=base_url,
                            memory_summary=chat.get("memory_summary"),
                            context_messages=get_recent_context_messages(chat),
                            allow_write_tools=allow_write_tools,
                            max_tool_rounds=max_tool_rounds,
                            request_timeout=request_timeout,
                            max_elapsed_seconds=max_elapsed_seconds,
                            save_debug_log=save_debug_log,
                            chat_id=chat["id"],
                        )
                    except Exception as exc:
                        st.error(f"LLM call failed: {exc}")
                    else:
                        now = datetime.now().isoformat(timespec="seconds")
                        chat["messages"].append({"role": "user", "content": question})
                        chat["messages"].append({"role": "assistant", "content": result["answer"]})
                        chat["memory_summary"] = build_memory_summary(chat["messages"])
                        chat["last_result"] = result
                        chat["updated_at"] = now
                        save_llm_chats_to_disk()
                        render_llm_result(result)
        elif chat.get("last_result"):
            st.divider()
            st.subheader("Last LLM Answer")
            render_llm_result(chat["last_result"])


def page_baselines():
    st.subheader("Baseline Strategy Results")
    st.write("Current baselines: Buy & Hold, Moving Average crossover, and RSI threshold strategy.")

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
            st.dataframe(comparison, use_container_width=True, height=240)

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


def page_tool_api_preview():
    st.subheader("LLM Tool API Preview")
    st.write("These are the Python functions that can later be registered as ChatGPT tools.")

    st.code(
        """
get_dataset_status()
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
load_shortlist_for_analysis(tickers, start_date, end_date, chat_id=None)
refresh_llm_workspace_data(tickers, start_date, end_date, data_source="auto", use_proxy=True, chat_id=None)
refresh_llm_workspace_ticker(ticker, start_date, end_date, data_source="auto", use_proxy=True, chat_id=None)
push_llm_workspace_to_app_pages(note=None, chat_id=None)
reset_app_pages_to_project_dataset(note=None)
merge_llm_workspace_to_project(tickers=None, chat_id=None)
clear_llm_workspace(chat_id=None)
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
        """.strip(),
        language="python",
    )

    status = market_tools.get_dataset_status()
    st.json(status)


st.title("Intelligent Financial Market Analysis")
st.caption("Lightweight Streamlit demo with tool-ready backend functions.")

active_dataset_status = market_tools.get_active_analysis_dataset_status()
df = load_processed_data(
    active_dataset_status["processed_file"],
    active_dataset_status.get("updated_at"),
)

with st.sidebar:
    page = st.radio(
        "Page",
        [
            "Data Setup",
            "Data Explorer",
            "EDA Results",
            "Baseline Results",
            "AI Assistant",
            "LLM Tool API Preview",
        ],
    )

if page == "Data Setup":
    page_data_setup(df)
elif page == "Data Explorer":
    page_explorer(df)
elif page == "EDA Results":
    page_eda()
elif page == "Baseline Results":
    page_baselines()
elif page == "AI Assistant":
    page_ai_assistant()
else:
    page_tool_api_preview()

st.divider()
st.caption("Baseline strategies, portfolio environment, and RL results will be added as saved artifacts.")
