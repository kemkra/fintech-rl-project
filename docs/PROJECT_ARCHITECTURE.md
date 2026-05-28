# Project Architecture

## 1. Course Requirement Alignment

The course project is an "intelligent data-questioning Web application". Therefore, this project should not be presented only as an RL model experiment. It should be presented as a financial market data analysis Web system with an RL trading agent as the advanced analysis module.

Required course modules and project mapping:

| Course Requirement | Project Module | Output |
| --- | --- | --- |
| Data file reading | `src/data/download_data.py` | `data/raw/*.csv` |
| Data preprocessing | `src/features/feature_engineering.py` | `data/processed/stock_features.csv` |
| Data analysis methods | EDA, baseline strategies, RL agent | summary tables, strategy results |
| Data visualization | `src/analysis/`, `src/visualization/`, Streamlit charts | `reports/figures/`, Web charts |
| Interactive analysis and basic Q&A | `app/streamlit_app.py`, AI assistant module | user questions, generated explanations |

The dataset also satisfies the course data requirements:

* More than 1000 rows
* More than 10 semantic columns after feature engineering
* Real-world financial time-series data
* Reproducible data collection and processing scripts

## 2. Final Product Positioning

Final product:

```text
An Intelligent Financial Market Analysis and Trading Strategy Web System
```

The system will allow users to:

1. Load and inspect historical stock / ETF data
2. Explore market trends and technical indicators
3. Compare baseline trading strategies
4. View RL trading agent behavior and performance
5. Ask natural-language questions about the dataset and analysis results

## 3. System Layers

```text
Data Layer
  raw market CSVs
  processed feature dataset

Analysis Layer
  EDA
  baseline strategies
  RL trading / portfolio environment
  backtesting metrics

Artifact Layer
  figures
  metrics tables
  equity curves
  action logs
  trained models

Web/UI Layer
  Streamlit dashboard
  interactive filters
  charts
  basic AI question answering

Documentation Layer
  README
  PPT
  demo video
  AI usage report
```

## 4. Data Flow

The system supports two data modes:

1. Stable demo mode

   * Use already downloaded and processed local data.
   * Recommended for classroom demo because it avoids network instability.

2. Interactive download mode

   * User selects tickers and date range in the Web UI.
   * The app checks whether matching local raw CSV files can satisfy the requested range.
   * The user can choose `auto` or `download`.
   * The app then triggers data preparation, feature engineering, and EDA refresh.
   * This demonstrates interactive analysis and makes the system more realistic.

```text
Web UI data controls
  ↓
selected tickers / start date / end date / proxy option
  ↓
local raw validation or src/data/download_data.py
  ↓
data/raw/*.csv
  ↓
src/features/feature_engineering.py
  Directly combines the selected individual raw CSV files in memory.
  The pipeline no longer keeps a saved merged raw CSV as a required artifact.
  ↓
data/processed/stock_features.csv
  ↓
EDA / strategies / RL environment / backtesting
  ↓
reports/figures/
reports/results/
models/
  ↓
src/tools/market_tools.py
  ↓
app/streamlit_app.py
```

The Web app should read saved artifacts by default. User-triggered data refresh is allowed, but long-running tasks such as RL training should remain offline before the demo. Business logic should live in reusable Python tools instead of Streamlit page code.

Data refresh source selection:

```text
auto
  Use valid local raw CSV first; download from yfinance only when local data is missing or incompatible.

download
  Force yfinance download.
```

Local raw validation checks required columns, date coverage, and whether the file looks like daily market data. A small date tolerance is allowed because requested calendar dates may fall on non-trading days.

Raw storage rule:

* Keep individual ticker CSV files in `data/raw/`.
* Do not keep `merged_stock_data.csv` as a long-lived artifact.
* Build combined raw frames in memory only when generating a processed dataset.

## 5. Planned Web Pages

### 5.1 Data Overview

Purpose:

* Prove that the system can read and summarize the dataset.
* Show ticker list, date range, row count, columns, and missing values.
* Allow the user to choose a custom market data period and refresh the dataset.

Core interactions:

* Select tickers
* Search ticker symbols from the cached Nasdaq Trader Symbol Directory
* Select start date and end date
* Choose stable local data mode or online download mode
* Optionally enable Clash proxy
* Trigger `Load & Process Data`
* Show raw / processed data preview
* Show current processed dataset status

### 5.2 EDA Dashboard

Purpose:

* Connect data analysis to financial insight.

Charts:

* Close price trend
* Cumulative return
* Daily return distribution
* Volatility comparison
* Asset correlation heatmap
* MA / RSI / MACD indicator chart

### 5.3 Strategy Backtest

Purpose:

* Compare classical strategies before introducing RL.

Strategies:

* Buy-and-Hold
* Moving Average crossover
* RSI strategy
* Random strategy

Outputs:

* Equity curve
* Performance metrics
* Trade/action table

### 5.4 RL Agent Demo

Purpose:

* Show the advanced module and project innovation.

Outputs:

* RL action timeline
* Portfolio value curve
* Comparison with baseline strategies
* Explanation of state, action, and reward design

### 5.5 Intelligent Q&A

Purpose:

* Satisfy the course requirement for interactive analysis / basic question answering.

Current implementation uses the LLM Tool-Calling Extension as the main interaction path. The earlier rule-based Offline Assistant has been removed to keep the Web UI focused.

## 5.6 LLM Tool-Calling Extension

The project has a tool-ready backend layer:

```text
src/tools/market_tools.py
```

These functions return JSON-friendly dictionaries and can later be registered as ChatGPT / OpenAI tool calls:

```text
get_dataset_status
list_local_raw_data
get_local_data_inventory
get_llm_workspace_status
search_us_symbols
validate_ticker_candidates
get_eda_summary
get_data_quality_summary
list_available_figures
get_ticker_history
create_ticker_price_chart
get_ticker_metrics
screen_stock_candidates
load_shortlist_for_analysis
refresh_llm_workspace_data
refresh_llm_workspace_ticker
merge_llm_workspace_to_project
clear_llm_workspace
run_buy_hold_baseline
get_buy_hold_metrics
get_buy_hold_equity_curve
run_ma_baseline
get_ma_metrics
get_ma_equity_curve
run_rsi_baseline
get_rsi_metrics
get_rsi_equity_curve
run_strategy_comparison
get_strategy_comparison
get_active_analysis_dataset_status
push_llm_workspace_to_app_pages
reset_app_pages_to_project_dataset
```

This keeps Streamlit as the presentation layer while the analysis tools can be reused by:

* Streamlit pages
* future OpenAI function calling
* future MCP server wrapper
* command-line scripts

Current LLM integration:

```text
src/llm/assistant.py
```

The first implementation uses OpenAI-compatible Chat Completions and tool-calling APIs. The Streamlit page lets the user choose:

* OpenAI
* DeepSeek
* Alibaba Bailian / DashScope compatible endpoint
* Ollama Local
* LM Studio Local
* Custom OpenAI-compatible endpoint

Users enter their own API key at runtime. The key is not saved by the project. Local providers such as Ollama and LM Studio can use placeholder keys because local OpenAI-compatible servers usually do not require real API keys.

Local LLM support:

* Ollama model discovery uses the local `/api/tags` endpoint and fills a model dropdown when available.
* Alibaba Bailian, LM Studio, and custom OpenAI-compatible endpoints can optionally load models from `/models`.
* Manual model input remains available as a fallback when model discovery is unavailable.
* Local providers do not require a real API key; a placeholder is used if the field is empty.
* Provider, base URL, and model preferences can be saved locally in `config/llm_preferences.json`.
* API keys are saved only when the user explicitly enables local key remembering.

Chat context:

* The AI Assistant supports multiple independent chats in the current Streamlit session.
* Each chat stores its own messages, last result, and compact memory summary.
* The UI presents a chat-list layout with create, switch, rename, clear, and delete actions.
* Chat history is saved locally in `config/llm_chats.json`, separate from API key preferences.
* LLM calls receive the compact memory summary plus recent messages from the active chat only.
* Full tool results are kept in debug logs and are not automatically injected into future context.

Debug logs:

* AI Assistant can save each LLM/tool-call run to `reports/logs/`.
* Logs include tool calls, compacted tool results, and final answer metadata for debugging.

For the local demo workflow, AI-triggered data loading uses shared raw files and per-chat analysis workspaces. The AI Assistant can refresh a Chat workspace without overwriting the main project dataset. Users can review workspace status in Data Setup and explicitly merge workspace data into the main project dataset when desired.

The Web app also has an active dataset pointer:

```text
config/active_analysis_dataset.json
```

This file records whether app pages should currently read the main project processed dataset or a specific Chat workspace processed dataset. It lets the LLM push an analysis dataset to Data Explorer / Baseline pages for inspection without overwriting `data/processed/stock_features.csv`.

Workspace storage:

```text
data/raw/                         shared raw cache
data/processed/                   main project processed dataset
data/workspaces/chats/<chat_id>/  per-chat processed/results/figures
```

Raw market CSV files should be shared across chats. Chat-level separation is most useful for processed datasets, figures, and strategy results because those represent a temporary analytical view rather than reusable source data.

Candidate discovery workflow:

```text
User asks for promising stocks
  ↓
screen_stock_candidates()
  Uses the cached symbol universe, local raw files, and yfinance fallback
  ↓
Shortlist of candidates by historical risk/return metrics
  ↓
load_shortlist_for_analysis()
  Loads only the shortlist into the current Chat workspace
  ↓
Feature engineering, baseline strategies, charts, and LLM explanation
```

The screener output should be described as historical candidate discovery, not direct investment advice.

Ticker resolution workflow:

```text
User mentions company name
  ↓
LLM proposes one or more ticker candidates
  ↓
validate_ticker_candidates checks local dataset and yfinance validity
  ↓
Only validated tickers are used for download, charting, or analysis
```

This avoids relying on a hard-coded translation dictionary and keeps the workflow extensible to new companies and markets.

## 6. RL Scope

The RL component should be positioned as an advanced analysis feature, not the only project goal.

Current stage:

* `TradingEnv`: single-asset trading environment
* Discrete actions: hold / buy / sell

Planned extension:

* `PortfolioEnv`: multi-asset portfolio management environment
* Continuous actions: allocation weights across assets

Practical delivery path:

1. First finish EDA and baseline strategies.
2. Build multi-asset portfolio environment.
3. Add RL training if time permits.
4. Use Web UI to compare RL against baselines.

## 7. Required Submission Materials

The final submission should include:

* Source code
* Test data with more than 1000 rows
* Data processing scripts
* README with setup and run instructions
* PPT for class presentation
* Demo video or live demo
* AI usage report and reflection
* Code source notes at function or file level where practical

## 8. AI Usage Documentation Plan

The course explicitly requires transparent AI usage disclosure. We will maintain:

```text
docs/AI_USAGE_LOG.md
```

Each important AI collaboration record should include:

* Task goal
* Prompt summary
* AI output summary
* Human decision / modification
* Verification method
* Final result

At least five key cases should be recorded before final submission.
