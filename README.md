# FinRL Insight

An intelligent financial market analysis and reinforcement-learning trading research app.

The project combines market data loading, technical feature engineering, exploratory data analysis, baseline trading strategies, a lightweight portfolio RL-style training scaffold, and an LLM tool-calling assistant in a Streamlit Web UI.

## What It Can Do

- Load US stock / ETF market data for user-selected tickers and date ranges.
- Generate technical indicators: MA5, MA20, RSI, MACD, daily return, and volatility.
- Run EDA summaries and charts for the active dataset.
- Generate lightweight fundamental, macro, and recent-news research reports.
- Backtest Buy & Hold, Moving Average crossover, and RSI threshold strategies.
- Run a multi-asset `PortfolioEnv` and train a lightweight Portfolio CEM policy.
- Compare traditional single-asset strategies with portfolio-level RL results.
- Let an LLM call local tools to prepare data, create charts, screen candidates, run strategies, and explain results.

The system is for education and research only. It does not provide investment advice.

## Project Structure

```text
app/                         Streamlit Web app
src/data/                    Market data download helpers
src/features/                Feature engineering
src/analysis/                EDA summaries and figures
src/strategies/              Buy & Hold, MA, RSI strategy baselines
src/environment/             Trading environments
src/training/                Lightweight portfolio RL training
src/evaluation/              Metrics and strategy comparison
src/tools/                   LLM-callable local tools
src/llm/                     OpenAI-compatible LLM tool-calling layer
src/storage/                 Runtime/session storage
docs/                        Architecture and AI usage notes
reports/                     Local debug logs and optional generated reports
```

## Local Setup

Use Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the app:

```bash
python run_app.py
```

Or run Streamlit directly:

```bash
streamlit run app/streamlit_app.py
```

## Streamlit Community Deployment

Set the main file path to:

```text
app/streamlit_app.py
```

Do not use `run_app.py` as the Streamlit Cloud entry file. It is only a local convenience wrapper.

Cloud deployments cannot use a local Clash proxy such as `127.0.0.1:7890`. Leave proxy usage off unless the server itself has a reachable proxy configured.

## Runtime Data

By default, local runs use the normal persistent project folders:

```text
data/
reports/
models/
config/
```

Streamlit Community/Web deployments should use temporary session data under:

```text
.streamlit_runtime/
```

This includes per-session raw data, processed feature files, figures, strategy results, model artifacts, LLM chat settings, and LLM background job status.

Storage mode is controlled by `FINTECH_STORAGE_MODE`:

- `auto` default: local runs use project folders; detected Streamlit Cloud runs use session runtime storage.
- `local`: force persistent project folders.
- `session`: force `.streamlit_runtime/sessions/<session_id>/` storage.

If cloud detection does not trigger in a hosted environment, add this to Streamlit secrets:

```toml
FINTECH_STORAGE_MODE = "session"
```

The `data/reference/us_stock_symbols.csv` file is a local symbol-search cache. If missing, the app can regenerate it from Nasdaq symbol directories when network access is available.

## LLM Providers

The AI Assistant supports OpenAI-compatible APIs:

- OpenAI
- DeepSeek
- Alibaba Bailian / DashScope compatible endpoint
- Ollama Local
- LM Studio Local
- Custom OpenAI-compatible endpoint

Remote providers require the user's own API key. Local providers can usually use an empty or placeholder key.

LLM calls run as background jobs in the Streamlit app, so switching pages should not interrupt ordinary in-progress requests. If the whole Streamlit process restarts, the background thread is still lost.

## Useful Questions

- What can you do?
- What local data is currently available?
- Load NVDA, AMD, and QQQ for the last year and compare strategies.
- Show Tesla's price chart for the last year.
- Research Apple's fundamentals and recent market context.
- Screen semiconductor stocks and compare the best candidates.
- Train the portfolio CEM strategy on the active dataset.

## Notes

- The current RL module is a lightweight Portfolio CEM scaffold, not a full PPO/DQN implementation.
- Strategy results are historical backtests and may not generalize to future markets.
- Public data loading depends on Yahoo/yfinance availability and rate limits.
