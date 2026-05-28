# Final Report

## 1. Project Title

FinRL Insight: An Intelligent Financial Market Analysis and Reinforcement Learning Trading System

## 2. Project Motivation

Financial market analysis is a suitable domain for an intelligent data-questioning Web application because it combines real-world data, feature engineering, visualization, strategy evaluation, and interactive user questions.

This project is designed as a financial analysis system with a reinforcement-learning trading component. The goal is not only to train an RL model, but also to build a usable workflow:

```text
market data -> features -> EDA -> strategy backtesting -> portfolio RL -> Web UI -> LLM interaction
```

## 3. Dataset

The system uses daily OHLCV market data from yfinance / Yahoo-compatible sources. Users can choose tickers and date ranges in the Web UI.

The feature dataset includes:

* Date
* Ticker
* OHLCV columns
* daily return
* MA5 and MA20
* RSI
* MACD and signal line
* rolling volatility

The app uses runtime session storage for Web deployment. This avoids relying on fixed local CSV files when deployed to Streamlit Community.

## 4. System Architecture

The system has five main layers:

| Layer | Responsibility |
| --- | --- |
| Data layer | Download or reuse market data and store runtime raw CSV files |
| Feature layer | Generate technical indicators and processed feature datasets |
| Analysis layer | Run EDA, baseline strategies, and portfolio RL training |
| Tool layer | Expose analysis functions as JSON-friendly tools for the LLM |
| Web/UI layer | Provide Streamlit pages for data loading, charts, results, and AI interaction |

Key implementation files:

| Module | File |
| --- | --- |
| Web app | `app/streamlit_app.py` |
| Data loading | `src/data/download_data.py` |
| Feature engineering | `src/features/feature_engineering.py` |
| EDA | `src/analysis/eda.py` |
| Baseline strategies | `src/strategies/` |
| Portfolio environment | `src/environment/portfolio_env.py` |
| Portfolio CEM training | `src/training/portfolio_cem.py` |
| Strategy comparison | `src/evaluation/strategy_comparison.py` |
| LLM tools | `src/tools/market_tools.py` |
| LLM assistant | `src/llm/assistant.py` |
| Runtime storage | `src/storage/runtime_store.py` |

## 5. Web Application

The Streamlit app contains these main pages:

* Data Setup
* Data Explorer
* EDA Results
* Baseline Results
* AI Assistant
* LLM Tool API Preview
* Tool Proposals

The Data Setup page lets users load tickers and date ranges. The EDA and Baseline pages follow the active dataset pointer, so both manual datasets and AI-generated Chat workspaces can be inspected.

## 6. Strategy Modules

Implemented traditional baselines:

* Buy & Hold
* Moving Average crossover
* RSI threshold strategy

Each strategy saves:

* equity curve
* performance metrics
* strategy-specific parameters

Common metrics include:

* total return
* annualized return
* annualized volatility
* Sharpe ratio
* maximum drawdown
* win rate

## 7. Portfolio RL Component

The project includes two trading environments:

* `TradingEnv`: a simple single-asset environment with hold / buy / sell actions
* `PortfolioEnv`: a multi-asset portfolio environment with continuous allocation weights

The current training module is a lightweight Portfolio CEM policy. It uses:

* cash plus multiple asset weights as the action
* normalized market features plus portfolio state as the observation
* portfolio return / log return as the reward signal
* Cross-Entropy Method to optimize a linear softmax allocation policy

This is a practical RL-style scaffold for the course project. It is not a full PPO/DQN deep reinforcement learning implementation, but it completes the full loop from data to environment, training, evaluation, Web display, and LLM tool access.

## 8. Strategy Comparison

The unified comparison table includes:

* BuyHold
* MovingAverage
* RSI
* PortfolioCEM

Because Portfolio CEM is portfolio-level while the traditional baselines are single-asset rows, the comparison table includes:

* `Asset_Set`
* `comparison_level`

This prevents portfolio-level and single-asset results from being mixed without context.

## 9. LLM Integration

The AI Assistant uses OpenAI-compatible Chat Completions and tool calling. Supported providers include:

* OpenAI
* DeepSeek
* Alibaba Bailian / DashScope
* Ollama Local
* LM Studio Local
* Custom OpenAI-compatible endpoints

The assistant can call project tools to:

* inspect data
* validate ticker candidates
* search symbol directories
* generate charts
* run EDA and strategies
* screen theme-based stock candidates
* train Portfolio CEM
* push Chat workspace data to app pages
* explain project capabilities

LLM requests run as background jobs in the Streamlit app, reducing the chance that page navigation interrupts long model/tool calls.

## 10. Deployment Considerations

For Streamlit Community deployment:

* main file path should be `app/streamlit_app.py`
* `run_app.py` should only be used for local command-line startup
* local Clash proxy addresses such as `127.0.0.1:7890` cannot be used from cloud servers
* runtime data is temporary and stored under `.streamlit_runtime/`

The system currently prioritizes deployability and usability over persistent cloud data storage.

## 11. Limitations

* Online data loading can fail because of yfinance/Yahoo rate limits.
* The current RL module is a lightweight CEM scaffold, not a full deep RL algorithm.
* Theme screening is historical candidate discovery, not fundamental analysis.
* The app does not provide investment advice.
* Runtime data on Streamlit Community is temporary and should not be treated as durable storage.

## 12. Future Improvements

Potential future work:

* add PPO, DQN, DDPG, or SAC training
* add persistent cloud storage
* improve long-running job management with a durable queue
* add richer chart interactions
* add portfolio constraints and transaction-cost sensitivity analysis
* add final report exports directly from the Web UI

## 13. Conclusion

FinRL Insight satisfies the course goal of an intelligent data-questioning Web application by connecting real market data, feature engineering, analysis, strategy comparison, reinforcement-learning-style portfolio training, and LLM tool-based interaction in one coherent system.
