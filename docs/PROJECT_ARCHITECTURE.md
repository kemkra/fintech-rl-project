# Project Architecture

This document is optimized for Codex and future maintainers. It explains the current architecture, not the historical path used to build it.

## 1. System Summary

FinRL Insight is a Streamlit financial analysis app with an LLM tool-calling layer.

Core workflow:

```text
market data
  -> feature engineering
  -> EDA / research / strategies / portfolio RL
  -> Streamlit pages
  -> LLM tools and export artifacts
```

The app supports local persistent storage and Web session storage.

## 2. Main Layers

| Layer | Responsibility | Main files |
| --- | --- | --- |
| Web UI | User pages, controls, charts, downloads | `app/streamlit_app.py` |
| Tool API | JSON-friendly functions for UI and LLM | `src/tools/market_tools.py` |
| LLM agent | OpenAI-compatible tool-calling loop | `src/llm/assistant.py` |
| Data | Download, fallback, raw normalization | `src/data/download_data.py`, `src/tools/market_tools.py` |
| Features | Technical indicators | `src/features/feature_engineering.py` |
| EDA | Summaries and figures | `src/analysis/eda.py` |
| Risk analytics | VaR, CVaR, drawdown, rolling risk, beta/correlation | `src/evaluation/risk_analysis.py`, `src/tools/market_tools.py` |
| Strategies | Buy & Hold, MA, RSI | `src/strategies/` |
| RL-style portfolio | PortfolioEnv and CEM trainer | `src/environment/portfolio_env.py`, `src/training/portfolio_cem.py` |
| Evaluation | Metrics and comparison tables | `src/evaluation/` |
| Report Builder | Unified Markdown/JSON report generation | `src/tools/market_tools.py`, `app/streamlit_app.py` |
| Storage | Runtime/session paths and SQLite registry | `src/storage/runtime_store.py` |
| Tests | Deterministic synthetic-data smoke coverage | `tests/test_core_tools.py` |
| Tool extension | Proposed/generated tools | `scripts/promote_tool_proposal.py`, `src/tools/generated_market_tools.py` |

## 3. Storage Architecture

Storage mode is controlled by `FINTECH_STORAGE_MODE`.

```text
auto     local folders on developer machine, session folders on Streamlit Cloud
local    force data/, reports/, models/, config/
session  force .streamlit_runtime/sessions/<session_id>/
```

Session mode runs a lightweight cleanup on app startup and preserves the active session while deleting old inactive session folders. The default cleanup age is controlled by `FINTECH_SESSION_CLEANUP_HOURS` and defaults to 24 hours.

Local mode:

```text
data/raw/
data/processed/stock_features.csv
data/workspaces/chats/<chat_id>/
reports/figures/
reports/results/
reports/research/
reports/generated/
reports/exports/
models/
config/
```

Session mode:

```text
.streamlit_runtime/runtime.db
.streamlit_runtime/sessions/<session_id>/raw/
.streamlit_runtime/sessions/<session_id>/processed/stock_features.csv
.streamlit_runtime/sessions/<session_id>/workspaces/chats/<chat_id>/
.streamlit_runtime/sessions/<session_id>/reports/figures/
.streamlit_runtime/sessions/<session_id>/reports/results/
.streamlit_runtime/sessions/<session_id>/reports/research/
.streamlit_runtime/sessions/<session_id>/reports/generated/
.streamlit_runtime/sessions/<session_id>/reports/exports/
.streamlit_runtime/sessions/<session_id>/models/
.streamlit_runtime/sessions/<session_id>/config/
```

Rules:

- Local runs should be durable and use project folders by default.
- Web deployments should isolate visitor data with session storage.
- Raw CSVs are reusable source data.
- Processed/features/results/research are analysis views and may be project-scoped or Chat-scoped.
- Do not require or recreate `merged_stock_data.csv` as a long-lived artifact.

## 4. Data Flow

Manual Data page flow:

```text
Data page ticker/date form
  -> market_tools.refresh_market_data()
  -> raw validation or yfinance/Yahoo chart download
  -> raw CSV cache
  -> feature engineering
  -> EDA artifacts
  -> risk summary and rolling risk artifacts
  -> baseline strategy artifacts
  -> active dataset reset to main loaded dataset
```

CSV upload flow:

```text
Data page CSV upload
  -> normalize_uploaded_price_data()
  -> import_uploaded_price_data()
  -> raw per-ticker CSV cache
  -> feature engineering
  -> EDA + risk + baseline strategy artifacts
  -> active dataset reset to uploaded main dataset
```

Risk analytics flow:

```text
Analysis page or run_risk_analysis()
  -> active/project/Chat-scoped processed dataset
  -> per-ticker VaR, CVaR, volatility, drawdown, Sharpe/Sortino, beta and correlation
  -> risk_summary.csv + risk_rolling.csv in the active results directory
  -> Report Builder and Export can include these CSV artifacts
```

AI Assistant data flow:

```text
user asks natural-language question
  -> app/streamlit_app.py starts a background job file under config/llm_jobs or session config/llm_jobs
  -> src/llm/assistant.py
  -> tool call into market_tools
  -> Chat-scoped workspace when write tools generate analysis data
  -> AI Assistant job dashboard reads queued/running/completed/failed JSON status files
  -> optional push_llm_workspace_to_app_pages()
  -> normal pages read active dataset pointer
```

Research flow:

```text
Research page or run_web_research_agent()
  -> yfinance fundamentals
  -> macro ETF/index snapshot
  -> Yahoo Finance RSS news links
  -> structured citations with source quality scores
  -> Markdown + JSON report in reports/research or Chat workspace research dir
```

Unified report flow:

```text
Report page or build_unified_report()
  -> active/project/Chat-scoped processed dataset
  -> EDA summary + data quality + risk summary + latest research + strategy comparison + Portfolio CEM metrics
  -> Markdown + JSON report in reports/generated or Chat workspace generated_reports dir
  -> optional Export ZIP entry
```

Export flow:

```text
list_exportable_artifacts()
  -> data / figures / results / research / reports / models inventory
  -> stable artifact_code per file
  -> export_analysis_artifacts() or export_selected_artifacts()
  -> ZIP with metadata.json
```

## 5. Active Dataset Pointer

The app pages read from the active dataset pointer rather than hardcoding `data/processed/stock_features.csv`.

Local mode:

```text
config/active_analysis_dataset.json
```

Session mode:

```text
.streamlit_runtime/sessions/<session_id>/config/active_analysis_dataset.json
```

The pointer records:

- `source`: `project` or `chat_workspace`
- `chat_id`: when a Chat workspace is active
- `note`
- `updated_at`

Pages using this pointer:

- Explorer
- Analysis
- Research save scope when set to active/project/workspace
- Strategy Lab
- Export

## 6. Streamlit UI Contract

Main pages:

```text
Data
Explorer
Analysis
Research
Strategy Lab
Report
AI Assistant
Export
```

Developer pages:

```text
Runtime Diagnostics
LLM Tool API Preview
Tool Proposals
```

UI rules:

- Keep common user paths simple.
- Hide paths, raw cache details, workspace internals, and merge tools behind advanced/developer sections.
- Do not put heavy business logic in Streamlit callbacks.
- Use `st.fragment(run_every="3s")` only for lightweight LLM pending-job polling.

## 7. LLM Tool Architecture

LLM entry point:

```text
src/llm/assistant.py::run_llm_tool_chat()
```

Tool registry:

```text
src/llm/assistant.py::TOOL_FUNCTIONS
src/llm/assistant.py::TOOL_SCHEMAS
src/llm/assistant.py::WRITE_TOOL_SCHEMAS
```

Tool implementation:

```text
src/tools/market_tools.py
```

Important tool families:

- Dataset status and inventory
- Ticker search and validation
- Data refresh and Chat workspace refresh
- Price chart and ticker history
- EDA summaries
- Risk analytics
- Strategy baselines and comparison
- Portfolio CEM training and metrics
- Fundamental/macro/news research
- Structured citations and source quality scores
- Unified report generation
- Export inventory and ZIP creation
- Tool proposal, promotion, and temporary composite tools

Write tools are enabled in the Streamlit AI Assistant by default, but generated data is scoped by storage mode and Chat workspace logic.

## 8. Current Limitations

- Portfolio CEM is a lightweight RL-style baseline, not a full PPO/DQN deep RL agent.
- yfinance/Yahoo data can be delayed, missing, rate-limited, or structurally inconsistent.
- Research Agent uses public yfinance/Yahoo RSS by default with structured citations; it is not a full paid-search deep research system.
- Streamlit Community session storage is temporary and not durable; old inactive sessions are cleaned up by age.
- The app is educational and should not provide investment advice.

## 9. Verification

Core checks:

```bash
.venv/bin/python -m py_compile app/streamlit_app.py src/tools/market_tools.py src/llm/assistant.py src/storage/runtime_store.py src/evaluation/risk_analysis.py
.venv/bin/python -m unittest discover -s tests -v
git diff --check
```

The initial test suite uses synthetic data and no network calls. It covers runtime registry writes, CSV upload import, risk metrics, unified report generation, strategy comparison, export ZIP creation, and LLM risk-tool registration.

## 10. Recommended Extension Points

| Goal | Best extension point |
| --- | --- |
| Add sector concentration risk | Extend `src/evaluation/risk_analysis.py` with sector metadata and concentration summaries |
| Add PDF/DOCX report export | Extend `build_unified_report()` output conversion and export integration |
| Add search-provider research | Add Tavily/Brave/SerpAPI provider functions in `market_tools.py` or new `src/research/` package |
| Add PPO/DQN | New training module under `src/training/`; keep PortfolioEnv contract stable |
| Add tests | New `tests/` using synthetic data and no network dependency |
