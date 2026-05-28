# AI Usage Log

This document records how AI tools were used during the project. It will be expanded continuously before final submission.

## Case 1: Project Planning and Module Decomposition

* Task goal: Break down the financial RL project into manageable modules.
* Prompt summary: Asked AI to read `PROJECT_PLAN.md` and continue implementation step by step.
* AI output summary: Suggested modular pipeline including data collection, feature engineering, trading environment, baselines, backtesting, RL training, and Web UI.
* Human decision / modification: Chose to keep a double-layer `src/` structure and implement modules incrementally.
* Verification method: Checked generated file structure and ran Python syntax checks.
* Final result: Initial project roadmap and module structure.

## Case 2: Feature Engineering Implementation

* Task goal: Generate technical indicators for market data.
* Prompt summary: Asked AI to implement MA5, MA20, RSI, MACD, and volatility.
* AI output summary: Created `src/features/feature_engineering.py`.
* Human decision / modification: Kept indicators grouped by ticker to avoid cross-asset rolling-window leakage.
* Verification method: Ran synthetic data tests and confirmed processed output schema.
* Final result: `data/processed/stock_features.csv`.

## Case 3: Data Download Debugging with Proxy

* Task goal: Fix yfinance download issues and use local Clash proxy.
* Prompt summary: Asked AI to simplify download code to yfinance only and add proxy support.
* AI output summary: Updated `src/data/download_data.py` with proxy environment variables and robust date normalization.
* Human decision / modification: Set actual proxy port to `127.0.0.1:7897`.
* Verification method: Tested QQQ download and confirmed standard output columns.
* Final result: Working yfinance data download pipeline.

## Case 4: Trading Environment Design

* Task goal: Implement a simple RL trading environment.
* Prompt summary: Asked AI to continue the next small module after data and features.
* AI output summary: Created `src/environment/trading_env.py` with hold / buy / sell actions.
* Human decision / modification: Clarified that this is a single-asset environment and that portfolio management should be a future environment.
* Verification method: Ran environment smoke tests over processed AAPL data.
* Final result: Reusable `TradingEnv` for later training.

## Case 5: Course Requirement and Web Architecture Alignment

* Task goal: Align the project with the final project PDF requirements.
* Prompt summary: Asked AI to reference the project PDF and improve project architecture.
* AI output summary: Extracted course requirements and mapped them to project modules.
* Human decision / modification: Repositioned the project as an intelligent financial data analysis Web system with RL as an innovation module.
* Verification method: Updated `PROJECT_PLAN.md`, `TODO.txt`, and `docs/PROJECT_ARCHITECTURE.md`.
* Final result: Clear final-product architecture and submission plan.

## Case 6: EDA Module Implementation

* Task goal: Build the exploratory data analysis module for the financial dataset.
* Prompt summary: Asked AI to continue the next module after architecture planning.
* AI output summary: Created `src/analysis/eda.py` to generate summary CSV files and reusable PNG charts.
* Human decision / modification: Chose static chart artifacts so they can be reused in Streamlit, PPT, and final report.
* Verification method: Ran the EDA script, checked generated CSV summaries, and verified PNG dimensions.
* Final result: EDA outputs in `reports/results/` and `reports/figures/`.

## Case 7: Streamlit Web Demo Skeleton

* Task goal: Add an early Web demo that supports custom ticker and date selection.
* Prompt summary: Asked AI whether Web interaction should allow selecting a time range and downloading data.
* AI output summary: Proposed a stable local-data mode plus an online download mode, then generated `app/streamlit_app.py`.
* Human decision / modification: Decided to keep online download user-triggered only, while preserving local artifacts for stable classroom demo.
* Verification method: Compiled the Streamlit app source and updated the architecture documents.
* Final result: Initial Web demo skeleton with data setup controls, EDA preview, figure viewer, and basic Q&A.

## Case 8: Streamlit Optimization and Tool-Ready Backend

* Task goal: Reduce Streamlit lag and make project tools easier to call from future LLM interfaces.
* Prompt summary: Asked AI to optimize Streamlit and wrap existing tools for ChatGPT-style calling.
* AI output summary: Reworked `app/streamlit_app.py` into page-based rendering and created `src/tools/market_tools.py`.
* Human decision / modification: Kept Streamlit as the Web framework but moved business logic out of the UI layer.
* Verification method: Ran syntax checks and called the new tool functions directly from Python.
* Final result: Faster Web skeleton and reusable JSON-friendly backend tools.

## Case 9: Buy & Hold Baseline

* Task goal: Implement the first baseline strategy for later comparison with RL.
* Prompt summary: Asked AI to continue the next project module while respecting the architecture and course requirements.
* AI output summary: Created `src/strategies/buy_hold.py` and `src/evaluation/metrics.py`, then connected outputs to tool functions and Streamlit.
* Human decision / modification: Kept Buy & Hold as a simple interpretable benchmark and saved both equity curves and metrics for Web/PPT reuse.
* Verification method: Ran the strategy over 10 tickers, checked generated CSV files, and called tool functions directly.
* Final result: Buy & Hold metrics and equity curves saved in `reports/results/`.

## Case 10: LLM Natural-Language Analysis Prototype

* Task goal: Let users ask natural-language questions and let an LLM call local project tools.
* Prompt summary: Asked AI to integrate multiple common LLM providers and connect them to existing project functions.
* AI output summary: Implemented an OpenAI-compatible tool-calling layer in `src/llm/assistant.py` and added a Streamlit AI Assistant page with common provider presets.
* Human decision / modification: Chose API-key input instead of full account login, and disabled write tools by default for safety.
* Verification method: Ran syntax checks and tested local tool dispatch without calling external APIs.
* Final result: Users can select provider, enter API key, ask a question, and allow the model to call local analysis tools.

## Case 11: Custom Ticker Input

* Task goal: Allow users to analyze stocks beyond the default project ticker list.
* Prompt summary: Asked AI to remove the limitation that only ten predefined tickers can be downloaded.
* AI output summary: Added ticker parsing helpers and updated the Streamlit data setup form.
* Human decision / modification: Kept the original ten tickers as quick selections while adding free-form custom ticker input.
* Verification method: Tested comma, space, semicolon, and newline-separated ticker parsing.
* Final result: Users can enter symbols such as `NVDA, META, NFLX, AMD` and run the same data pipeline.

## Case 12: Free Local LLM Options

* Task goal: Provide alternatives to paid OpenAI API usage.
* Prompt summary: Asked AI whether free alternatives exist and then asked it to implement the change.
* AI output summary: Added Ollama Local and LM Studio Local provider presets. The earlier rule-based assistant was later removed after the LLM tool interface became the primary interaction path.
* Human decision / modification: Kept cloud provider support but made local OpenAI-compatible servers first-class options.
* Verification method: Checked provider presets and compiled the Streamlit and LLM modules.
* Final result: The app can use local LLM servers for demos without paid API calls.

## Case 13: Expanded LLM Tool Interfaces

* Task goal: Let the LLM generate charts and refresh missing ticker data when permitted.
* Prompt summary: User reported that hard-coded ticker translation is too limited and suggested LLM-proposed candidates plus validation.
* AI output summary: Added ticker candidate validation, ticker price chart generation, and one-ticker data refresh tools.
* Human decision / modification: Replaced the primary hard-coded alias workflow with LLM candidate proposal and tool-based validation.
* Verification method: Tested candidate validation for `特斯拉 -> TSLA`, checked tool schemas, and confirmed disabled write tools return a safety message.
* Final result: The LLM can propose ticker candidates, tools validate them, and validated tickers can be used for charting or data refresh.

## Case 14: LLM Speed, Local Raw Reuse, and Debug Logs

* Task goal: Make LLM interaction faster and make data refresh safer and more transparent.
* Prompt summary: User requested shorter LLM wait time, local raw reuse before yfinance download, Ollama model discovery, optional API key for local models, and debug logs.
* AI output summary: Added local raw validation, combined local data inventory, simplified `auto/download` refresh modes, compacted tool results, optional tool/runtime limits, Ollama model discovery, and JSON debug logs.
* Human decision / modification: Kept yfinance download available but made local raw reuse the default in `auto` mode.
* Verification method: Checked local raw validation for AAPL/TSLA, verified Ollama model discovery, and compiled Streamlit/LLM/tool modules.
* Final result: Faster and more controllable AI Assistant workflow with inspectable logs under `reports/logs/`.

## Case 15: Consolidate Bailian Provider and Add Model Discovery

* Task goal: Avoid duplicate provider entries for the same Alibaba compatible API family and make model selection easier.
* Prompt summary: User noted that Qwen DashScope and Alibaba Bailian use the same kind of endpoint and asked whether available models can be fetched automatically.
* AI output summary: Kept one Alibaba Bailian provider preset, added OpenAI-compatible `/models` discovery, exposed a Streamlit button to load available models on demand, and made LLM tool-round/runtime settings configurable through explicit limit toggles.
* Human decision / modification: Decided not to keep two provider names when one configuration path is enough.
* Verification method: Compiled the Streamlit and LLM modules and tested provider preset metadata locally.
* Final result: Cleaner provider configuration with manual model fallback, optional automatic model selection through a unified Model control, and configurable LLM execution limits.

## Case 16: Streamline Assistant UI and Refresh Baselines

* Task goal: Remove redundant assistant UI and keep baseline results aligned with refreshed data.
* Prompt summary: User said Offline Assistant was redundant, AI example questions were unnecessary, and Baseline Results did not update after Data Setup changes.
* AI output summary: Removed the Offline Assistant page and rule-based helper, removed AI Assistant example questions, made market data refresh automatically rerun Buy & Hold baseline artifacts, added a Data Setup snapshot for current processed/raw data, and persisted the latest LLM answer across Streamlit page switches.
* Human decision / modification: Chose the LLM tool-calling interface as the single question-answering path.
* Verification method: Compiled Streamlit, LLM, and market tool modules; checked tool schemas and stale references.
* Final result: Cleaner Web navigation, visible current data state, baseline metrics/equity curves that update after data refresh, and LLM answers that remain visible after navigating away and back.

## Case 17: Multi-Chat Context and LLM Preferences

* Task goal: Support separate AI conversations and avoid re-entering model settings after every app restart.
* Prompt summary: User asked for multiple chats instead of one ever-growing context and requested a memory feature for API/model choices.
* AI output summary: Added persisted multi-chat management, compact per-chat context summaries, recent-message context passing to the LLM API, chat-list UI with numbered chats and rename support, a larger unified searchable ticker picker, and local LLM preferences saved to `config/llm_preferences.json`.
* Human decision / modification: API keys are only saved when the user explicitly enables local key remembering.
* Verification method: Compiled Streamlit and LLM modules; checked that context parameters are accepted by the LLM layer.
* Final result: Users can create, switch, rename, clear, delete, and reopen independent chats after restart, while saved provider/model settings are restored and ticker selection is handled through one search/add control.

## Case 18: Moving Average Baseline Strategy

* Task goal: Add the next traditional baseline strategy for later comparison with RL.
* Prompt summary: User asked to continue the next project module after the Web/LLM interface refinements.
* AI output summary: Implemented a MA5/MA20 crossover strategy with one-day delayed execution, saved equity curves and metrics, and connected the outputs to Streamlit and LLM tools.
* Human decision / modification: Kept the strategy simple and interpretable as a baseline, not an optimized trading system.
* Verification method: Ran `src/strategies/moving_average.py`, compiled changed modules, and checked MA tool schemas and result files.
* Final result: MA baseline metrics and equity curves saved in `reports/results/` and available in Baseline Results.

## Case 19: Isolated LLM Data Workspace

* Task goal: Let the LLM download missing ticker data without unexpectedly replacing the main project dataset.
* Prompt summary: User suggested separating AI-operated files from non-LLM analysis data.
* AI output summary: Added `data/llm_workspace/`, changed LLM refresh tools to write there, added main-data fallback for analysis, made baseline tools workspace-aware, and added Data Setup controls to view, clear, or merge workspace data.
* Human decision / modification: Chose explicit merge into the main project dataset instead of letting the LLM overwrite project artifacts automatically.
* Verification method: Refreshed AAPL into the LLM workspace and confirmed the main processed file timestamp did not change; verified ORCL Buy & Hold and MA metrics load from workspace; compiled Streamlit, tools, and LLM modules.
* Final result: LLM analysis can auto-download data and evaluate baseline strategies in an isolated workspace, while main project data changes only through Data Setup or explicit merge.

## Case 20: RSI Baseline Strategy

* Task goal: Add the next traditional baseline strategy before unified strategy comparison and RL training.
* Prompt summary: User asked AI to continue the next project module according to the existing architecture, paper context, and course requirements.
* AI output summary: Implemented `src/strategies/rsi.py`, connected RSI baseline outputs to market tools, LLM tool schemas, and Streamlit Baseline Results.
* Human decision / modification: Kept the strategy as a simple interpretable RSI threshold baseline with buy threshold 30, sell threshold 70, and one-day delayed execution.
* Verification method: Ran the RSI strategy script, compiled changed modules, and checked that project and LLM workspace RSI metrics can be generated.
* Final result: RSI metrics and equity curves are available in `reports/results/` and in `data/llm_workspace/results/` for AI-downloaded tickers.

## Case 21: Unified Strategy Comparison

* Task goal: Combine baseline strategy results into one comparison artifact for Web display, LLM answers, and final reporting.
* Prompt summary: User asked AI to continue the next project module after completing traditional baseline strategies.
* AI output summary: Added `src/evaluation/strategy_comparison.py`, exposed comparison tools, and added a unified comparison section to Streamlit Baseline Results.
* Human decision / modification: Compared only implemented baseline strategies first: Buy & Hold, Moving Average, and RSI.
* Verification method: Ran the comparison script and called `run_strategy_comparison()` / `get_strategy_comparison()` from the tool layer.
* Final result: Strategy comparison is saved to `reports/results/strategy_comparison.csv` with per-ticker rankings by return, Sharpe ratio, and max drawdown.

## Case 22: Active Dataset Push From LLM

* Task goal: Let users inspect an AI-generated analysis dataset in normal Web pages without merging it into the main project dataset.
* Prompt summary: User asked whether analysis datasets should be separated by Chat and whether the LLM API can push the current analysis dataset to other pages.
* AI output summary: Added an active app dataset pointer, exposed push/reset tools to the LLM, and updated Streamlit to read the active dataset.
* Human decision / modification: Chose a lightweight first step: active dataset switching now, per-chat processed/result workspaces as a later refinement.
* Verification method: Compiled Streamlit, tools, and LLM modules; tested active dataset status, push, and reset tool functions.
* Final result: LLM can call `push_llm_workspace_to_app_pages()` so Data Explorer and related pages show AI workspace data without overwriting main project files.

## Case 23: Remove Saved Merged Raw Data

* Task goal: Simplify the data pipeline by removing `merged_stock_data.csv` as a persistent intermediate artifact.
* Prompt summary: User asked to implement the previously discussed change for the merged raw data issue.
* AI output summary: Removed merged raw file generation from download, project refresh, and AI workspace refresh; updated feature engineering to read individual raw ticker CSV files directly.
* Human decision / modification: Kept individual raw files as the reusable data cache and treated merged raw frames as temporary in-memory objects only.
* Verification method: Compiled changed modules, searched the project for stale merged-data references, deleted generated merged CSV files, and smoke-tested feature generation from raw files.
* Final result: Processed datasets are now generated from `data/raw/*.csv` or in-memory selected raw frames without requiring `merged_stock_data.csv`.

## Case 24: Searchable Ticker Universe and Load Data Wording

* Task goal: Improve the data loading interface so users can search a broad stock symbol universe instead of choosing from a small hard-coded list.
* Prompt summary: User asked whether the data page could automatically pull and match all stock codes, and suggested renaming download wording to loading.
* AI output summary: Added Nasdaq Trader Symbol Directory fetch/cache utilities, integrated a searchable ticker universe into Streamlit Data Setup, renamed the UI flow to Load market data, and exposed symbol search to the LLM tool layer.
* Human decision / modification: Kept manual yfinance ticker entry as a fallback because not every useful market symbol is present in the US symbol directory.
* Verification method: Refreshed the symbol cache, searched for Oracle, compiled changed modules, and checked the UI code path.
* Final result: The data loading UI can search thousands of US-listed ticker symbols while still supporting local cache reuse and online yfinance loading.

## Case 25: Per-Chat AI Analysis Workspace

* Task goal: Reduce data-workspace confusion by sharing raw market data while isolating AI analysis outputs by Chat.
* Prompt summary: User confirmed the plan to refactor AI workspace into shared raw data plus per-chat processed/results/figures.
* AI output summary: Added `data/workspaces/chats/<chat_id>/`, routed LLM workspace tools through the active Chat ID, and changed AI refresh flows to save raw downloads in `data/raw/` while saving processed/results in the Chat workspace.
* Human decision / modification: Kept existing tool names for compatibility but changed their internal behavior to be Chat-scoped.
* Verification method: Compiled Streamlit/tools/LLM modules and smoke-tested refresh, baseline generation, strategy comparison, push-to-pages, reset, and clear for a test Chat workspace.
* Final result: Each Chat can now keep its own temporary processed dataset and results, while raw data is shared and reusable across the whole project.

## Case 26: LLM Stock Screener

* Task goal: Let the LLM answer broad stock-discovery questions without requiring all stocks to be fully downloaded and processed first.
* Prompt summary: User asked for a better way to answer questions like finding several stocks worth buying or researching.
* AI output summary: Added `screen_stock_candidates()` for lightweight historical screening and `load_shortlist_for_analysis()` for loading only selected candidates into the current Chat workspace.
* Human decision / modification: Framed the output as candidate discovery for further research, not direct investment advice.
* Verification method: Compiled changed modules and smoke-tested screening with local raw fallback during yfinance rate limiting.
* Final result: The LLM can now screen a broad candidate universe, return a shortlist, and then trigger deeper analysis only for that shortlist.

## Final Reflection

### Key human decisions

The student made several important design decisions during the project:

* Position the project as an intelligent financial market analysis Web application, with RL as an advanced module rather than the only focus.
* Keep Streamlit as the main Web framework because it is fast to prototype and suitable for course demonstration.
* Separate business logic from Streamlit page code by moving reusable operations into `src/tools/market_tools.py`.
* Use per-session runtime storage for Streamlit deployment so other users can access the app without depending on the developer's local data files.
* Treat AI-generated analysis data as temporary Chat workspace data unless explicitly pushed or merged.
* Use lightweight Portfolio CEM training as a practical RL-style scaffold before adding heavier PPO/DQN algorithms.

### AI suggestions accepted

The student accepted AI suggestions that improved modularity, deployability, and usability:

* Feature engineering with grouped ticker operations to avoid cross-asset leakage.
* JSON-friendly tool functions for future LLM / MCP expansion.
* Per-chat AI workspace design.
* Runtime storage for Streamlit Community compatibility.
* Background LLM jobs to reduce page-switch interruption.
* A capability tool so the assistant can clearly answer what the system can do.

### AI suggestions modified or rejected

Some AI suggestions were adjusted by the student:

* The early rule-based Offline Assistant was removed because the LLM tool-calling assistant became the main interaction path.
* Hard-coded ticker translation was replaced with LLM-proposed candidates plus validation.
* The project did not move immediately to a heavier Web stack because Streamlit was still sufficient after optimization.
* Full PPO/DQN training was postponed because the project first needed a reliable end-to-end data, Web, and evaluation workflow.

### Where AI was most helpful

AI was most useful for:

* breaking the project into implementable modules
* debugging yfinance data formats and proxy behavior
* designing reusable tool interfaces
* keeping the Web app aligned with the backend architecture
* expanding LLM workflows from simple Q&A to tool-using analysis
* updating documentation and TODO / plan files as the architecture evolved

### Where AI needed human correction

AI sometimes needed correction when:

* it assumed the local dataset was the only possible source of analysis
* it initially relied on limited ticker-name mappings
* it produced workflows that were too manual for broad stock-discovery questions
* it suggested or preserved older workspace paths after the runtime/session architecture changed

These issues were corrected through user feedback and code revisions.

### Verification methods

The project used several verification methods:

* Python syntax checks with `py_compile`
* `git diff --check` for formatting and whitespace issues
* smoke tests with synthetic market data
* direct tool calls from Python
* Streamlit runtime checks
* search-based audits for stale paths and outdated documentation
* manual review of generated CSV/JSON/figure artifacts

### Final assessment

AI significantly accelerated implementation and documentation, but the student remained responsible for architecture decisions, safety boundaries, project scope, verification, and final acceptance of generated code.
