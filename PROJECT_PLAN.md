# PROJECT_PLAN

## Project Title

Intelligent Financial Market Analysis and RL Trading Web System

---

# 1. Project Goal

The objective of this project is to build an intelligent financial market analysis Web system.

The system reads historical market data, preprocesses it, performs exploratory analysis, visualizes market behavior, compares trading strategies, and provides an interactive question-answering interface. Reinforcement Learning (RL) trading is the advanced / innovative module of the system.

Main goals:

* Collect and process historical financial market data
* Build an interactive financial data analysis Web app
* Provide exploratory data analysis and visualizations
* Implement baseline trading strategies
* Construct an RL trading environment
* Design state, action, and reward representations
* Train and evaluate an RL agent
* Compare performance against benchmark strategies
* Provide a basic intelligent Q&A interface over dataset and analysis results

Course requirement mapping:

* Data file reading → `src/data/download_data.py`
* Data preprocessing → `src/features/feature_engineering.py`
* Data analysis methods → EDA, baseline strategies, RL backtesting
* Data visualization → figures and Streamlit charts
* Interactive analysis / Q&A → Streamlit UI and question-answering module

Detailed architecture:

* See `docs/PROJECT_ARCHITECTURE.md`

---

# 2. Current Progress

### Completed

* [x] Project direction selected:

  * FinTech + Reinforcement Learning

* [x] Initial project architecture designed

* [x] Dataset source determined:

  * yfinance

* [x] Selected assets:

  Equity stocks:

  * AAPL
  * MSFT
  * GOOGL
  * AMZN
  * TSLA

  Financial sector stocks:

  * JPM
  * BAC
  * GS

  Market ETFs:

  * SPY
  * QQQ

* [x] Data download module planned

* [x] Download historical market data

  * Implemented `src/data/download_data.py`
  * Uses yfinance with optional Clash proxy support
  * Stores individual ticker CSV files in `data/raw/`

* [x] Store raw data into data/raw/

* [x] Feature engineering

  * Implemented `src/features/feature_engineering.py`
  * Generated Daily_Return, MA5, MA20, RSI, MACD, and Volatility
  * Reads individual raw ticker CSV files directly instead of relying on a saved merged raw file
  * Stores processed data in `data/processed/stock_features.csv`

* [x] Trading environment construction

  * Implemented `src/environment/trading_env.py`
  * Supports single-asset Hold/Buy/Sell actions
  * State includes market features, technical indicators, and portfolio information

* [x] Web/UI architecture planning

  * Designed offline artifact pipeline for data, models, reports, and Streamlit UI
  * Defined Web UI pages and artifact dependencies

* [x] Existing Python module quality review

  * Rechecked data download, feature engineering, and trading environment modules
  * Added safer proxy configuration, raw CSV filtering, and environment input validation

* [x] EDA and exploratory visualization

  * Implemented `src/analysis/eda.py`
  * Generated data quality summary and asset-level EDA summary
  * Saved reusable figures to `reports/figures/`

* [x] Interactive Web data refresh architecture

  * Designed stable local-data demo mode
  * Designed user-triggered online download mode for selected tickers and date range

* [x] Streamlit Web Demo skeleton

  * Implemented `app/streamlit_app.py`
  * Added local data mode, custom download controls, EDA preview, generated figures, and basic Q&A

* [x] Lightweight Web optimization and tool-ready backend

  * Split Streamlit into sidebar-selected pages to avoid rendering every section at once
  * Added downsampled chart rendering and collapsed data previews
  * Implemented `src/tools/market_tools.py` with JSON-friendly functions for future ChatGPT tool calling

* [x] Buy-and-Hold baseline strategy

  * Implemented `src/strategies/buy_hold.py`
  * Added initial reusable performance metrics in `src/evaluation/metrics.py`
  * Saved equity curves and metrics to `reports/results/`
  * Exposed Buy & Hold tools through `src/tools/market_tools.py`

* [x] Moving Average baseline strategy

  * Implemented `src/strategies/moving_average.py`
  * Added MA5/MA20 crossover baseline using a one-day delayed signal to avoid look-ahead bias
  * Saved MA equity curves and metrics to `reports/results/`
  * Exposed MA strategy tools through `src/tools/market_tools.py`, LLM tool schemas, and Streamlit Baseline Results

* [x] RSI baseline strategy

  * Implemented `src/strategies/rsi.py`
  * Added RSI threshold baseline using a one-day delayed signal to avoid look-ahead bias
  * Saved RSI equity curves and metrics to `reports/results/`
  * Exposed RSI strategy tools through `src/tools/market_tools.py`, LLM tool schemas, and Streamlit Baseline Results
  * Made RSI outputs workspace-aware for LLM-downloaded tickers in per-chat AI workspaces

* [x] Unified backtesting metrics and strategy comparison

  * Implemented `src/evaluation/strategy_comparison.py`
  * Aggregates Buy & Hold, Moving Average, and RSI metrics into one comparison table
  * Adds per-ticker ranking by total return, Sharpe ratio, and max drawdown
  * Saves comparison output to `reports/results/strategy_comparison.csv`
  * Exposed strategy comparison through `src/tools/market_tools.py`, LLM tool schemas, and Streamlit Baseline Results

* [x] Active app dataset switching for LLM analysis

  * Added `config/active_analysis_dataset.json` as a local UI state pointer
  * Added LLM tools for pushing the AI workspace dataset to app pages and resetting back to the main project dataset
  * Updated Streamlit pages to read the active dataset without overwriting `data/processed/stock_features.csv`
  * Added Data Setup visibility and reset controls for the active app dataset

* [x] Removed saved merged raw data dependency

  * Removed generation of `merged_stock_data.csv` from the download and refresh workflows
  * Updated feature engineering to load directly from individual `data/raw/*.csv` files
  * Updated project and AI workspace refresh flows to build processed datasets from in-memory raw frames
  * Deleted existing generated merged raw CSV files

* [x] Searchable US ticker universe for data loading

  * Added Nasdaq Trader Symbol Directory loading and local caching
  * Updated Data Setup wording from download-first language to load/process language
  * Replaced the small ticker picker with a searchable symbol universe while preserving manual yfinance ticker entry
  * Exposed `search_us_symbols()` to the LLM tool layer

* [x] Per-chat AI analysis workspace

  * Uses per-session runtime raw storage as the shared raw market data cache for both UI and LLM workflows
  * Stores AI-generated processed data, figures, and strategy results under `.streamlit_runtime/sessions/<session_id>/workspaces/chats/<chat_id>/`
  * Automatically passes the active Chat ID into LLM workspace tools
  * Updated active dataset push/reset so normal app pages can inspect a specific Chat workspace

* [x] Stock screener for LLM candidate discovery

  * Added `screen_stock_candidates()` for broad historical screening over a sampled US stock universe
  * Uses local raw history first and falls back to yfinance for missing candidates
  * Ranks candidates by return, volatility, drawdown, volume, and risk-adjusted return
  * Saves screener results to the current Chat workspace
  * Added `load_shortlist_for_analysis()` to load only the shortlist into the Chat workspace for deeper feature/strategy analysis

* [x] Streamlit Community runtime storage

  * Added per-session runtime directories under `.streamlit_runtime/sessions/<session_id>/`
  * Added `src/storage/runtime_store.py` with a lightweight SQLite runtime database
  * Updated Streamlit startup to configure project tools into the current Web session runtime
  * Routes generated raw files, processed features, reports, figures, active dataset state, LLM preferences, and Chat history away from shared project paths during Web use
  * Records temporary raw/feature datasets in SQLite while keeping CSV outputs for existing EDA, baseline, and chart modules
  * Keeps the design deploy-friendly for Streamlit Community without requiring persistent cloud storage yet
  * Changed the Web data-loading proxy checkbox to default off unless `USE_PROXY=true`, because Streamlit Community cannot use a local Clash proxy
  * Fixed EDA helper defaults so runtime sessions read the session processed file instead of the old `data/processed/stock_features.csv`
  * Fixed processed-data reads for baseline strategy generation under runtime storage
  * Added more explicit per-ticker diagnostics when yfinance returns empty data or is rate-limited
  * Added a guard to `run_app.py` so it cannot recursively start extra Streamlit servers if accidentally selected as the Cloud entry file
  * Added a Yahoo chart API fallback for online data loading when yfinance returns empty data under rate limits
  * Changed proxy handling so Streamlit Cloud does not accidentally try to use the developer's local Clash proxy
  * Hardened LLM stock screening against missing `Date` columns in downloaded history
  * Added semiconductor/chip thematic seed tickers so broad prompts like "芯片科技股票" can screen a relevant universe instead of relying on exact symbol-name matching
  * Added intent-level LLM workflows:
    * `prepare_ticker_analysis()` automatically loads requested tickers, generates features, runs baselines, creates charts, and can push the dataset to app pages
    * `analyze_theme_candidates()` automatically screens a theme, loads the shortlist, runs deeper analysis, and returns records for the final answer
  * Updated the LLM system prompt so normal data refreshes are treated as internal analysis steps rather than user-facing confirmation chores
  * Fixed chart/history tools when the main project dataset is empty but the current Chat workspace has data
  * Added a safe Tool Proposal workflow:
    * `propose_new_tool()` saves non-executable JSON proposals for missing tools
    * `list_tool_proposals()` lists saved proposals
    * Streamlit includes a Tool Proposals page for review
    * Proposals are stored in the current runtime session and are not dynamically executed
  * Added a local-only Tool Promotion workflow:
    * Reviewed proposals can be promoted with `python scripts/promote_tool_proposal.py --proposal-file <proposal.json>`
    * Promoted functions are written to `src/tools/generated_market_tools.py`
    * Promoted schemas are written to `src/llm/generated_tool_schemas.py`
    * `src/llm/assistant.py` loads generated tools on startup so they become available after commit/push/redeploy
    * The promotion script validates one function definition, blocks risky names/imports, updates proposal status, and compiles generated files
    * Local LLM-driven promotion is available only when `ENABLE_LOCAL_TOOL_PROMOTION=true`
    * The assistant refreshes generated tool modules before building tool schemas and before executing tool calls, so promoted local tools can be used in the next LLM turn
  * Added Web-safe temporary composite tools:
    * `register_temp_composite_tool()` creates session-scoped JSON wrappers around approved base tools
    * Temporary tools support preset arguments, simple record filters, sorting, selected columns, and row limits
    * They are exposed to the LLM in the current Web session without writing Python code or affecting other sessions
    * Tool Proposals page now shows temporary composite tools registered in the session
  * Refined active dataset flow across Web pages:
    * Manual Data Setup loads now reset pages to the main loaded project dataset and rerun immediately
    * AI workspace refreshes generate workspace EDA summaries and figures
    * `EDA Results` follows the active dataset's summaries and figures instead of always reading project reports
    * Price chart generation from a Chat workspace automatically activates that workspace for app-page inspection
    * Removed the ordinary `Clear AI workspace` control from Data Setup and moved merge behind a developer expander

* [x] LLM natural-language analysis prototype

  * Implemented `src/llm/assistant.py`
  * Added OpenAI-compatible tool-calling support for OpenAI, DeepSeek, Alibaba Bailian, local model servers, and custom providers
  * Added Streamlit AI Assistant page where users enter their own API key
  * Connected LLM tool calls to local market analysis functions
  * Added optional model discovery for providers that expose a compatible `/models` endpoint, with one unified model selector in the UI
  * Made tool-call round limits and LLM request timeouts configurable from the Web UI
  * Persisted the latest LLM answer in Streamlit session state so it remains visible after page switches
  * Added multiple independent AI chats with compact per-chat context memory
  * Reworked the AI Assistant into a chat-list layout with current-chat rename support
  * Persisted AI chat history locally in `config/llm_chats.json` with numbered default chat names
  * Added local LLM setting persistence for provider, base URL, model, and optional API key

* [x] Custom ticker input for Web data refresh

  * Streamlit now supports both quick ticker selection and arbitrary custom ticker input
  * Added ticker parsing helpers in `src/tools/market_tools.py`
  * Replaced separate quick/custom ticker fields with one searchable picker that includes an expanded common ticker pool and accepts new symbols

* [x] Local free LLM provider presets

  * Added Ollama Local and LM Studio Local OpenAI-compatible presets
  * Removed the earlier rule-based Offline Assistant after the LLM tool interface became the main interaction path

* [x] Expanded LLM tool interfaces for ticker validation, charting, and data refresh

  * Added LLM-proposed ticker candidate validation through local data and yfinance
  * Added chart generation tool for ticker price history
  * Added one-ticker data refresh tool gated behind write-tool permission
  * Isolated LLM automatic analysis outputs so AI analysis does not overwrite the main project dataset
  * Added explicit merge/clear controls for AI workspace data in Data Setup
  * Made Buy & Hold and MA tools workspace-aware so LLM-downloaded tickers can be evaluated without merging first

* [x] Optimized LLM workflow, local raw reuse, and debug logging

  * Made tool-round and total-runtime limits optional instead of always-on numeric limits
  * Added local raw CSV validation and simplified data refresh modes to `auto/download`
  * Added combined local data inventory so the AI can report both processed data and raw CSV files
  * Added a Data Setup snapshot showing the current processed dataset and local raw-file inventory
  * Added Ollama local model discovery
  * Added optional LLM/tool-call logs under `reports/logs/`

* [x] Multi-asset PortfolioEnv

  * Implemented `src/environment/portfolio_env.py`
  * Supports multiple assets with overlapping dates and continuous non-negative portfolio weights
  * Includes cash allocation, transaction cost, turnover tracking, portfolio value, log-return reward, and optional risk penalty
  * Observation combines normalized market features with current portfolio state
  * Added `run_portfolio_env_smoke_test()` to the tool layer for lightweight validation from the app/LLM side

* [x] Lightweight portfolio RL training scaffold

  * Implemented `src/training/portfolio_cem.py`
  * Uses a linear softmax allocation policy over cash plus selected assets
  * Trains with Cross-Entropy Method as a lightweight, dependency-free RL-style baseline before heavier DQN/PPO work
  * Saves equity curve, metrics, training history, and policy weights as runtime artifacts
  * Exposed `run_portfolio_cem_training()`, `get_portfolio_rl_metrics()`, and `get_portfolio_rl_equity_curve()` to the tool layer and LLM function-calling interface
  * Added a Portfolio RL Training block to the Baseline Results page

* [x] Strategy comparison including portfolio RL

  * Extended `src/evaluation/strategy_comparison.py` to include Portfolio CEM metrics when available
  * Added `Asset_Set` and `comparison_level` fields so single-asset strategy rows and portfolio-level RL rows are not confused
  * Rebuilds the comparison table automatically after Portfolio CEM training
  * Updated Streamlit and LLM descriptions to present the comparison as traditional baselines plus portfolio RL

* [x] User-facing capability explanation

  * Added `get_project_capabilities()` to the tool layer
  * Exposed it to the LLM function-calling interface
  * Updated the assistant prompt so questions like "你可以做什么" or "这个项目能解决什么问题" produce a structured project capability overview

* [x] Background LLM job handling in Streamlit

  * Replaced synchronous AI Assistant calls with background thread jobs
  * Stores job status and sanitized results under the current runtime config directory
  * Keeps pending jobs attached to the current Chat so users can switch pages and return later
  * Merges completed answers back into chat history and clears cached data views after tool-generated artifacts are ready

* [x] Project handoff readiness audit

  * Added a top-level `README.md` with local setup, Streamlit Community deployment, runtime storage, LLM providers, and example questions
  * Updated `docs/PROJECT_ARCHITECTURE.md` to match the current runtime/session architecture and Portfolio CEM scope
  * Updated `pyproject.toml` project description
  * Rechecked Python syntax, ignored runtime files, and deployment entry-point guidance

* [x] Final end-to-end acceptance test

  * Verified the runtime tool pipeline with deterministic synthetic market data
  * Generated EDA summaries and figures
  * Ran Buy & Hold, Moving Average, RSI, and Portfolio CEM
  * Verified the unified strategy comparison includes all four strategy types
  * Verified key LLM tool schemas are exposed
  * Started the Streamlit app and confirmed the Web UI loads successfully

---

### In Progress

* [x] Core project complete

---

### Future Tasks

* [ ] Add heavier RL algorithms such as PPO or DQN if time permits
* [x] Strategy comparison and performance analysis
* [ ] Streamlit Web UI refinement
* [x] Final report
* [x] AI Assistant usage guide

---

# 3. Target System Architecture

The project will be built as an offline research pipeline plus a lightweight Web/UI layer. This aligns with the course requirement of an intelligent data-questioning Web application.

Core idea:

```text
Data / Feature / Training / Backtesting run offline
↓
Artifacts are saved to data/, models/, reports/
↓
Web UI reads saved artifacts and presents results interactively
```

The Web UI should not retrain models on every page load. It should load processed data, precomputed backtest results, trained model outputs, and generated figures.

## 3.1 Module Responsibilities

```text
src/data/
  Download historical market data and save raw CSV files.

src/features/
  Generate technical indicators and save processed feature datasets.

src/analysis/
  Run EDA and save reusable charts / summary tables.

src/strategies/
  Implement baseline strategies such as Buy-and-Hold, MA crossover, RSI, and random strategy.

src/environment/
  Provide trading environments for RL agents.
  trading_env.py: single-asset discrete trading environment.
  portfolio_env.py: future multi-asset portfolio management environment.

src/models/
  Train RL agents and save model files plus prediction/action logs.

src/evaluation/
  Run backtests and compute common performance metrics.

src/visualization/
  Build reusable plotting functions for reports and the Web UI.

app/
  Streamlit Web UI for interactive exploration and final demo.
```

## 3.2 Artifact Flow

```text
.streamlit_runtime/sessions/<session_id>/raw/*.csv
  Produced or reused by src/data/download_data.py through the Web/runtime tool layer

.streamlit_runtime/sessions/<session_id>/processed/stock_features.csv
  Produced by src/features/feature_engineering.py through the Web/runtime tool layer

.streamlit_runtime/sessions/<session_id>/reports/figures/
  Produced by src/analysis/eda.py and chart-generation tools

.streamlit_runtime/sessions/<session_id>/reports/results/
  Stores strategy equity curves, metrics tables, EDA summaries, and comparison tables

.streamlit_runtime/sessions/<session_id>/models/
  Stores lightweight portfolio policy artifacts

app/
  Reads the active runtime dataset and artifacts selected by the active dataset pointer
```

## 3.3 Web / UI Design

The final Web UI will be implemented with Streamlit.

Planned pages:

1. Data Overview

   * Ticker selector
   * Price chart
   * Volume chart
   * Missing data / date range summary

2. EDA Dashboard

   * Cumulative return comparison
   * Daily return distribution
   * Volatility comparison
   * Correlation heatmap
   * Technical indicator chart for MA, RSI, MACD

3. Strategy Backtest

   * Select Buy & Hold, MA, RSI, or Portfolio CEM strategy
   * Show equity curve
   * Show trade markers
   * Show performance metrics

4. RL Agent Demo

   * Select ticker or portfolio setting
   * Load trained model outputs
   * Display action timeline
   * Display portfolio value over time

5. Comparison Report

   * Compare baseline strategies and RL agent
   * Rank by Total Return, Sharpe Ratio, Max Drawdown, and Volatility
   * Export or display final summary charts

## 3.4 Current Project Structure

```text
fintech_rl_project/

├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── reports/
│   ├── figures/
│   └── results/
│
├── src/
│   ├── data/
│   │   └── download_data.py
│   │
│   ├── features/
│   │   └── feature_engineering.py
│   │
│   ├── environment/
│   │   └── trading_env.py
│   │
│   ├── analysis/
│   │   └── eda.py
│   │
│   ├── strategies/
│   │   └── baseline_strategies.py
│   │
│   ├── models/
│   │   └── train_agent.py
│   │
│   ├── evaluation/
│   │   ├── backtest.py
│   │   └── metrics.py
│   │
│   └── visualization/
│       └── plots.py
│
├── PROJECT_PLAN.md
├── pyproject.toml
└── README.md
```

---

# 4. Data Information

Data source:

* yfinance

Frequency:

* Daily data (1d)

Time period:

* 2015–2025

Fields:

* Open
* High
* Low
* Close
* Volume

Additional features to be generated later:

* Daily returns
* Moving averages
* RSI
* MACD
* Volatility
* Momentum indicators

---

# 5. Reinforcement Learning Design

## State Space

Possible state variables:

Market information:

* Open
* High
* Low
* Close
* Volume

Technical indicators:

* MA5
* MA20
* RSI
* MACD
* Volatility

Portfolio information:

* Current cash
* Current holdings
* Portfolio value

---

## Action Space

Discrete actions:

0 → Hold

1 → Buy

2 → Sell

Possible future extension:

Continuous actions:

* Buy x%
* Sell x%

---

## Reward Function

Initial reward:

Daily portfolio return

Example:

reward = (portfolio_value_today − portfolio_value_yesterday)

Possible improvements:

* Risk-adjusted reward
* Sharpe ratio reward
* Transaction cost penalty
* Drawdown penalty

---

# 6. Baseline Methods

The RL agent will be compared against:

1. Buy-and-Hold strategy

2. Moving Average strategy

3. Random trading strategy

---

# 7. Evaluation Metrics

Performance metrics:

* Total Return
* Annualized Return
* Sharpe Ratio
* Maximum Drawdown
* Win Rate

---

# 8. Development Order

Step 1

Data Collection
(Completed)

↓

Step 2

Feature Engineering
(Completed)

↓

Step 3

Trading Environment Construction
(Completed)

↓

Step 4

Course Requirement Alignment and Architecture Planning
(Completed)

↓

Step 5

EDA and Exploratory Visualization
(Current stage)

↓

Step 6

Baseline Strategy Construction

↓

Step 7

Unified Backtesting and Metrics

↓

Step 8

Portfolio Management Environment

↓

Step 9

RL Agent Training

↓

Step 10

Strategy Comparison and Analysis

↓

Step 11

Streamlit Web UI

↓

Step 12

Intelligent Q&A and AI Usage Documentation

↓

Step 13

Final Report

---

# 9. Notes for Codex

Important implementation requirements:

* Use modular code structure
* Add comments and docstrings
* Add basic error handling
* Save intermediate outputs
* Keep functions reusable
* Use pandas/numpy for processing
* Use matplotlib for visualization
* Prefer reproducible experiments

Future RL libraries to consider:

* stable-baselines3
* gymnasium
* pytorch

```
```
