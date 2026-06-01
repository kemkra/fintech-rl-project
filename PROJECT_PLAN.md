# FinRL Insight Project Plan

This file is the compact working plan for Codex and human maintainers. It describes the current product, completed modules, and the next useful work. Historical implementation notes live in `docs/AI_USAGE_LOG.md`.

## 1. Product Goal

Build an intelligent financial market analysis Web app that can:

- Load stock / ETF market data for custom tickers and date ranges.
- Engineer technical indicators and run EDA.
- Compare traditional strategy baselines and a lightweight portfolio RL-style policy.
- Provide fundamental, macro, and recent-news research context.
- Let an LLM call local tools to prepare data, run analysis, explain results, and export artifacts.

This is an educational research system, not investment advice.

## 2. Current User-Facing Pages

The Streamlit app is implemented in `app/streamlit_app.py`.

| Page | Purpose |
| --- | --- |
| `Data` | Load market data and show the current dataset. Advanced storage/raw/workspace details are hidden by default. |
| `Explorer` | Inspect price and indicator time series for one ticker. |
| `Analysis` | Show EDA summaries, data quality, risk analytics, and generated figures. |
| `Research` | Generate fundamental, macro, and recent-news research reports. |
| `Strategy Lab` | Run/view Buy & Hold, MA, RSI, strategy comparison, and Portfolio CEM. |
| `Report` | Build a unified Markdown/JSON report from active data, EDA, research, strategies, and Portfolio CEM. |
| `AI Assistant` | Natural-language tool-calling assistant with multi-chat history. |
| `Export` | Export selected data, figures, research reports, strategy results, and model files as ZIP. |

Developer-only pages are hidden behind the `Developer tools` checkbox:

- `LLM Tool API Preview`
- `Tool Proposals`

## 3. Storage Modes

Storage mode is selected in `app/streamlit_app.py`.

| Mode | Trigger | Output location |
| --- | --- | --- |
| `local` | `FINTECH_STORAGE_MODE=local` or default local run | `data/`, `reports/`, `models/`, `config/` |
| `session` | `FINTECH_STORAGE_MODE=session` or detected Streamlit Cloud | `.streamlit_runtime/sessions/<session_id>/` |
| `auto` | default | local on developer machine, session on Streamlit Cloud |

Use `FINTECH_STORAGE_MODE = "session"` in Streamlit secrets if cloud detection fails.

## 4. Implemented Modules

| Area | Status | Main files |
| --- | --- | --- |
| Data download | Done | `src/data/download_data.py`, `src/tools/market_tools.py` |
| CSV upload | Done | `app/streamlit_app.py`, `src/tools/market_tools.py` |
| Feature engineering | Done | `src/features/feature_engineering.py` |
| EDA | Done | `src/analysis/eda.py` |
| Risk analytics | Done | `src/evaluation/risk_analysis.py`, `src/tools/market_tools.py` |
| Single-asset env | Done | `src/environment/trading_env.py` |
| Portfolio env | Done | `src/environment/portfolio_env.py` |
| Buy & Hold baseline | Done | `src/strategies/buy_hold.py` |
| MA baseline | Done | `src/strategies/moving_average.py` |
| RSI baseline | Done | `src/strategies/rsi.py` |
| Strategy comparison | Done | `src/evaluation/strategy_comparison.py` |
| Portfolio CEM training | Done | `src/training/portfolio_cem.py` |
| Web UI | Done, actively refined | `app/streamlit_app.py` |
| LLM tool layer | Done | `src/llm/assistant.py`, `src/tools/market_tools.py` |
| LLM job dashboard | Done, lightweight | `app/streamlit_app.py` |
| Research Agent | Done, lightweight | `src/tools/market_tools.py`, `app/streamlit_app.py` |
| Unified report builder | Done | `src/tools/market_tools.py`, `app/streamlit_app.py` |
| Export system | Done | `src/tools/market_tools.py`, `app/streamlit_app.py` |
| Runtime storage | Done | `src/storage/runtime_store.py` |
| Deployment hardening | Done, lightweight | `src/storage/runtime_store.py`, `app/streamlit_app.py` |
| Tool proposal/promotion | Done | `scripts/promote_tool_proposal.py`, generated tool files |
| Deterministic smoke tests | Done, initial suite | `tests/test_core_tools.py` |

## 5. Current Architecture Rules

- Business logic belongs in `src/tools/market_tools.py` or dedicated `src/` modules, not directly in Streamlit page code.
- Streamlit page code should call tool functions and render their JSON-friendly results.
- LLM-callable tools must return JSON-serializable dictionaries.
- Local runs should preserve project files; cloud runs should isolate data per session.
- AI-generated processed data, figures, strategy results, research reports, and unified reports should be Chat-scoped when launched from a Chat workflow.
- Raw market CSVs may be shared within the current storage mode; processed/results/research artifacts represent an analysis view and should stay scoped.
- Avoid requiring `merged_stock_data.csv`; combine individual raw ticker frames in memory.
- Keep user-facing UI simple; put paths, raw cache, merge, runtime, and developer controls behind advanced sections.

## 6. Verification Checklist

Run these after code changes:

```bash
.venv/bin/python -m py_compile app/streamlit_app.py src/tools/market_tools.py src/llm/assistant.py src/evaluation/risk_analysis.py
.venv/bin/python -m unittest discover -s tests -v
git diff --check
```

For UI changes, start Streamlit and inspect the relevant page:

```bash
.venv/bin/streamlit run app/streamlit_app.py --server.port 8526 --server.headless true
```

For storage changes, verify both modes:

```bash
FINTECH_STORAGE_MODE=local .venv/bin/streamlit run app/streamlit_app.py
FINTECH_STORAGE_MODE=session .venv/bin/streamlit run app/streamlit_app.py
```

## 7. High-Value Backlog

| Priority | Task | Notes |
| --- | --- | --- |
| P1 | Broader tests | Add deterministic tests for data refresh with mocked downloads, Streamlit helpers, and LLM chat job persistence. |
| P1 | Sector concentration | Extend risk analytics with sector metadata and concentration limits. |
| P2 | Research search-provider upgrade | Optional Tavily/Brave/SerpAPI multi-source web search providers beyond current yfinance/Yahoo citation layer. |
| P2 | Full DRL training | Add PPO/DQN/A2C only if time and dependency budget allow. |
| P3 | Export retention polish | Add separate expiry controls for old ZIP exports and generated reports if needed. |

## 8. Documentation Map

- `README.md`: user setup, deployment, and feature overview.
- `PROJECT_PLAN.md`: current implementation status and next work.
- `TODO.txt`: short actionable backlog.
- `docs/PROJECT_ARCHITECTURE.md`: Codex-readable architecture and data flow.
- `docs/PROJECT_STRUCTURE.md`: directory and module map.
- `docs/AI_ASSISTANT_GUIDE.md`: how to use the LLM assistant.
- `docs/FINAL_REPORT.md`: final course/report narrative.
- `docs/AI_USAGE_LOG.md`: historical AI-assisted development log.
