# Project Structure

This file maps the repository for Codex. Update it when modules move or major responsibilities change.

## Top-Level Files

| Path | Purpose |
| --- | --- |
| `README.md` | User-facing setup, deployment, and feature overview. |
| `PROJECT_PLAN.md` | Current implementation status and next work. |
| `TODO.txt` | Short actionable backlog. |
| `pyproject.toml` | Project metadata and runtime dependencies. |
| `requirements.txt` | Pinned dependency list for Streamlit deployment. |
| `run_app.py` | Local helper entry point; not for Streamlit Cloud entry file. |
| `.gitignore` | Ignores runtime/session/generated cache files. |

## App Layer

| Path | Purpose |
| --- | --- |
| `app/streamlit_app.py` | Main Streamlit app. Contains page rendering and UI state helpers. |

Main Streamlit pages:

- `page_data_setup()` -> Data
- `page_explorer()` -> Explorer
- `page_eda()` -> Analysis
- `page_research_agent()` -> Research
- `page_baselines()` -> Strategy Lab
- `page_report_builder()` -> Report
- `page_ai_assistant()` -> AI Assistant
- `page_exports()` -> Export
- `page_tool_api_preview()` -> developer-only API preview
- `page_tool_proposals()` -> developer-only tool proposal review

## Source Modules

| Path | Purpose |
| --- | --- |
| `src/data/download_data.py` | Lower-level yfinance/Yahoo chart download helpers. |
| `src/features/feature_engineering.py` | Generates Daily_Return, MA5, MA20, RSI, MACD, Volatility. |
| `src/analysis/eda.py` | EDA summaries, data quality, and figures. |
| `src/strategies/buy_hold.py` | Buy & Hold baseline. |
| `src/strategies/moving_average.py` | MA crossover baseline. |
| `src/strategies/rsi.py` | RSI threshold baseline. |
| `src/environment/trading_env.py` | Simple single-asset trading environment. |
| `src/environment/portfolio_env.py` | Multi-asset portfolio environment. |
| `src/training/portfolio_cem.py` | Lightweight Portfolio CEM training. |
| `src/evaluation/metrics.py` | Shared performance metrics. |
| `src/evaluation/risk_analysis.py` | VaR, CVaR, rolling volatility, rolling Sharpe, drawdown, beta, and correlation. |
| `src/evaluation/strategy_comparison.py` | Unified strategy comparison table. |
| `src/storage/runtime_store.py` | Session runtime paths and SQLite dataset registry. |
| `src/tools/market_tools.py` | Main backend tool API for Streamlit and LLM. |
| `src/tools/generated_market_tools.py` | Generated/promoted tool functions. |
| `src/llm/assistant.py` | OpenAI-compatible LLM tool-calling loop and schemas. |
| `src/llm/generated_tool_schemas.py` | Generated/promoted tool schemas. |

## Documentation

| Path | Purpose |
| --- | --- |
| `docs/PROJECT_ARCHITECTURE.md` | Current architecture and data flow. |
| `docs/PROJECT_STRUCTURE.md` | This repository map. |
| `docs/AI_ASSISTANT_GUIDE.md` | User/developer guide for the AI Assistant. |
| `docs/FINAL_REPORT.md` | Course/final report narrative. |
| `docs/AI_USAGE_LOG.md` | Historical AI-assisted development log. |

## Tests

| Path | Purpose |
| --- | --- |
| `tests/test_core_tools.py` | Synthetic-data smoke tests for runtime storage, CSV upload, risk analytics, reports, strategy comparison, export ZIPs, and LLM tool registration. |

## Data and Artifact Directories

Local mode:

| Path | Purpose |
| --- | --- |
| `data/raw/` | Raw per-ticker market CSV files. |
| `data/processed/stock_features.csv` | Main processed feature dataset. |
| `data/reference/us_stock_symbols.csv` | Cached symbol universe. |
| `data/workspaces/chats/<chat_id>/` | Local Chat-scoped processed/results/figures/research. |
| `reports/figures/` | EDA and chart figures. |
| `reports/results/` | EDA, risk, strategy, comparison, and Portfolio CEM CSV outputs. |
| `reports/research/` | Markdown/JSON research reports. |
| `reports/generated/` | Unified Markdown/JSON reports built from data, EDA, research, and strategy artifacts. |
| `reports/exports/` | ZIP export packages. |
| `reports/tool_proposals/` | LLM-generated tool proposals. |
| `models/` | Local model/policy artifacts. |
| `config/` | Local app state, LLM preferences, chat history, job status. |

Session mode:

| Path | Purpose |
| --- | --- |
| `.streamlit_runtime/runtime.db` | SQLite registry for temporary runtime datasets. |
| `.streamlit_runtime/sessions/<session_id>/` | Isolated session root. |
| `.streamlit_runtime/sessions/<session_id>/raw/` | Session raw cache. |
| `.streamlit_runtime/sessions/<session_id>/processed/` | Session main processed dataset. |
| `.streamlit_runtime/sessions/<session_id>/workspaces/chats/<chat_id>/` | Session Chat workspaces. |
| `.streamlit_runtime/sessions/<session_id>/reports/` | Session figures/results/research/generated reports/exports/logs. |
| `.streamlit_runtime/sessions/<session_id>/config/` | Session preferences, chat history, jobs, active dataset pointer. |

## Most Important Functions

Data and storage:

- `market_tools.configure_runtime_storage()`
- `market_tools.get_runtime_diagnostics()`
- `market_tools.cleanup_runtime_sessions()`
- `market_tools.refresh_market_data()`
- `market_tools.import_uploaded_price_data()`
- `market_tools.normalize_uploaded_price_data()`
- `market_tools.refresh_llm_workspace_data()`
- `market_tools.get_active_analysis_dataset_status()`
- `market_tools.push_llm_workspace_to_app_pages()`

Analysis:

- `market_tools.get_eda_summary()`
- `market_tools.get_data_quality_summary()`
- `market_tools.run_risk_analysis()`
- `market_tools.get_risk_summary()`
- `market_tools.create_ticker_price_chart()`
- `market_tools.run_strategy_comparison()`
- `market_tools.run_portfolio_cem_training()`
- `market_tools.run_web_research_agent()`
- `market_tools.build_research_citations()`
- `market_tools.build_unified_report()`

Export:

- `market_tools.list_exportable_artifacts()`
- `market_tools.export_analysis_artifacts()`
- `market_tools.export_selected_artifacts()`

LLM:

- `run_llm_tool_chat()` in `src/llm/assistant.py`
- `execute_tool_call()` in `src/llm/assistant.py`
- `TOOL_FUNCTIONS`, `TOOL_SCHEMAS`, `WRITE_TOOL_SCHEMAS`

Streamlit LLM jobs:

- `start_llm_background_job()` in `app/streamlit_app.py`
- `reconcile_llm_job()` in `app/streamlit_app.py`
- `render_llm_job_dashboard()` in `app/streamlit_app.py`

Developer diagnostics:

- `page_runtime_diagnostics()` in `app/streamlit_app.py`

## Common Commands

Compile important modules:

```bash
.venv/bin/python -m py_compile app/streamlit_app.py src/tools/market_tools.py src/llm/assistant.py src/storage/runtime_store.py src/evaluation/risk_analysis.py
```

Run deterministic tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Run local app:

```bash
FINTECH_STORAGE_MODE=local .venv/bin/streamlit run app/streamlit_app.py
```

Run session-mode app:

```bash
FINTECH_STORAGE_MODE=session .venv/bin/streamlit run app/streamlit_app.py
```

Check whitespace:

```bash
git diff --check
```

## Editing Guidance

- Prefer small, scoped changes.
- Put reusable logic in `src/`, not in Streamlit rendering blocks.
- Preserve local/session storage behavior when changing paths.
- Keep LLM tool outputs JSON-serializable.
- After completing a module, update `PROJECT_PLAN.md`, `TODO.txt`, and this file if architecture or module ownership changed.
