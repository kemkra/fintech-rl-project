import importlib
import json
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.tools import market_tools

try:
    from src.llm.generated_tool_schemas import GENERATED_TOOL_SCHEMAS
    from src.tools.generated_market_tools import (
        GENERATED_CHAT_WORKSPACE_TOOL_NAMES,
        GENERATED_TOOL_FUNCTIONS,
        GENERATED_WRITE_TOOL_NAMES,
    )
except Exception:
    GENERATED_CHAT_WORKSPACE_TOOL_NAMES = set()
    GENERATED_TOOL_FUNCTIONS = {}
    GENERATED_TOOL_SCHEMAS = []
    GENERATED_WRITE_TOOL_NAMES = set()


PROVIDER_PRESETS = {
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "DeepSeek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "Alibaba Bailian": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
    "Ollama Local": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5:7b",
    },
    "LM Studio Local": {
        "base_url": "http://localhost:1234/v1",
        "model": "local-model",
    },
    "Custom OpenAI-Compatible": {
        "base_url": "",
        "model": "",
    },
}

LOGS_DIR = Path("reports/logs")


SYSTEM_PROMPT = """
You are a financial data analysis assistant for a student project.
Your job is to deliver the final analysis result, not to make the user manage data files.
Treat cached/raw/processed data as an internal working medium. Do not repeatedly ask for permission to download or refresh ordinary market data when write tools are enabled.
Use tools when the user asks about dataset status, ticker metrics, EDA results, baseline strategy comparison, figures, or ticker history.
If the user asks what this project/app/assistant can do, what problems it can solve, or how to use it, call get_project_capabilities and answer from that capability map.
If the user asks what local data is available, call get_local_data_inventory so the answer includes both processed data and raw CSV files.
If the user asks for a chart, visual, recent performance, baseline strategy results, or a comparison for known tickers, use prepare_ticker_analysis when write tools are enabled.
If the user asks to discover promising stocks, find stocks worth researching, screen buy candidates, or analyze a theme/industry, use analyze_theme_candidates when write tools are enabled.
For follow-up confirmations such as "start training", "continue", "confirm", or "run portfolio management" after data has already been prepared in the current chat, do not invent or replace tickers. First use the current Chat workspace data, and call run_portfolio_cem_training with data_scope="workspace" or without a ticker list.
If write tools are disabled, use read-only tools and explain that automatic data preparation is unavailable.
If the user's request cannot be handled well with the existing tools, call propose_new_tool to create a tool proposal for developer review. Do not claim the proposed tool has been implemented or executed.
If the user asks what new tools have been proposed, call list_tool_proposals.
If the user asks for a temporary Web-session tool that can be built safely from existing tools, call register_temp_composite_tool. These temporary tools are JSON-configured wrappers around approved base tools, not arbitrary Python code.
If the user explicitly asks to merge/promote a reviewed proposal into the local toolset, call get_tool_promotion_status first. If local promotion is enabled, call promote_tool_proposal_local. If disabled, explain the local command/environment variable.
If the user uses a company name instead of a ticker, propose one or more likely ticker candidates and call validate_ticker_candidates to verify them.
If the user asks to find possible stock tickers, call search_us_symbols before validation.
Use the selected_ticker returned by validate_ticker_candidates in later tools.
If write tools are enabled and a missing ticker is validated, refresh the current Chat workspace data directly instead of asking for another confirmation.
If validation fails because of network/rate limits, ask the user to confirm the ticker or retry later instead of pretending the ticker was verified.
For "recent", "last year", or similar requests, choose a reasonable default lookback period such as 1y unless the user specifies dates.
Only ask follow-up questions when the request is genuinely ambiguous, very broad/expensive, requests real trading instructions, or needs paid/private credentials.
If the user wants to inspect the AI Chat workspace data in other app pages, call push_llm_workspace_to_app_pages.
When you generate charts or refresh workspace data for the user's analysis, make sure the workspace is active for app pages. The chart tool and workflow tools usually do this automatically.
Explain results clearly and mention whether outputs are based on the main project data or the current Chat workspace.
Do not provide investment advice. Frame conclusions as historical analysis.
""".strip()


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_project_capabilities",
            "description": "Get a clear user-facing overview of what this financial analysis and RL trading project can currently do, including examples and limitations.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_dataset_status",
            "description": "Get current processed dataset status, including tickers, dates, rows, and columns.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_runtime_storage_status",
            "description": "Get the active Web/runtime storage status, including the SQLite runtime database and temporary datasets for this session.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_local_raw_data",
            "description": "List valid and invalid raw CSV files in the active runtime cache, including tickers, date ranges, and row counts.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_local_data_inventory",
            "description": "Get a combined inventory of current processed data and all local raw CSV files.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_llm_workspace_status",
            "description": "Get the current Chat workspace status. Raw files live in the active runtime cache; processed/results are Chat-scoped.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_active_analysis_dataset_status",
            "description": "Get which processed dataset is currently shown by the Web app pages.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tool_proposals",
            "description": "List saved tool proposals that were generated for developer review.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
                    "status": {"type": "string", "description": "Optional proposal status filter, such as proposed."},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_tool_promotion_status",
            "description": "Check whether local developer tool promotion is enabled for this running app.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_temp_composite_tools",
            "description": "List temporary session-scoped composite tools registered in the current Web/runtime session.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_ticker_candidates",
            "description": "Validate one or more LLM-proposed ticker candidates using yfinance and the local dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Original user phrase, such as Tesla, 特斯拉, or Nvidia."},
                    "candidates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Likely ticker candidates proposed by the LLM, such as TSLA or NVDA.",
                    },
                    "validation_mode": {
                        "type": "string",
                        "enum": ["local_first", "local_only", "yfinance"],
                        "default": "local_first",
                    },
                },
                "required": ["query", "candidates"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_us_symbols",
            "description": "Search the cached US stock symbol universe by ticker or company name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Ticker or company name search text."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 25},
                    "force_refresh": {"type": "boolean", "default": False},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_local_raw_ticker",
            "description": "Inspect whether a local raw CSV for one ticker can satisfy the requested date range and interval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "interval": {"type": "string", "default": "1d"},
                },
                "required": ["ticker"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_eda_summary",
            "description": "Get the asset-level EDA summary table.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_data_quality_summary",
            "description": "Get data quality summary by ticker.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_figures",
            "description": "List generated EDA figure files.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ticker_history",
            "description": "Get recent historical rows for one ticker from the processed feature dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "start_date": {"type": "string", "description": "Optional YYYY-MM-DD start date."},
                    "end_date": {"type": "string", "description": "Optional YYYY-MM-DD end date."},
                    "max_rows": {"type": "integer", "minimum": 1, "maximum": 200},
                },
                "required": ["ticker"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_ticker_price_chart",
            "description": "Generate a local PNG price chart for one validated ticker over a selected date range or recent month window.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Validated ticker symbol, such as TSLA."},
                    "start_date": {"type": "string", "description": "Optional YYYY-MM-DD start date."},
                    "end_date": {"type": "string", "description": "Optional YYYY-MM-DD end date."},
                    "months": {"type": "integer", "minimum": 1, "maximum": 120, "default": 12},
                    "include_ma": {"type": "boolean", "default": True},
                },
                "required": ["ticker"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ticker_metrics",
            "description": "Get EDA metrics for one ticker.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "screen_stock_candidates",
            "description": "Screen a broad US stock universe using recent historical price/volume metrics and return a shortlist for further research. Use automatically for broad candidate-discovery questions when write tools are available. This is not investment advice.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Optional ticker/company-name filter, such as semiconductor, bank, apple, or oracle."},
                    "max_candidates": {"type": "integer", "minimum": 10, "maximum": 1000, "default": 300},
                    "shortlist_size": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                    "lookback_period": {"type": "string", "default": "1y", "description": "yfinance period such as 6mo, 1y, 2y, or 5y."},
                    "min_avg_volume": {"type": "number", "default": 500000},
                    "min_price": {"type": "number", "default": 5},
                    "include_etfs": {"type": "boolean", "default": False},
                    "sort_by": {
                        "type": "string",
                        "enum": ["risk_adjusted_return", "total_return", "annualized_return", "low_drawdown", "volume"],
                        "default": "risk_adjusted_return",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_buy_hold_metrics",
            "description": "Get Buy & Hold strategy metrics, optionally filtered by ticker.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_buy_hold_equity_curve",
            "description": "Get Buy & Hold equity curve records for one ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "max_rows": {"type": "integer", "minimum": 1, "maximum": 500},
                    "data_scope": {"type": "string", "enum": ["auto", "project", "workspace"], "default": "auto"},
                },
                "required": ["ticker"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ma_metrics",
            "description": "Get Moving Average crossover strategy metrics, optionally filtered by ticker.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ma_equity_curve",
            "description": "Get Moving Average crossover strategy equity curve records for one ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "max_rows": {"type": "integer", "minimum": 1, "maximum": 500},
                    "data_scope": {"type": "string", "enum": ["auto", "project", "workspace"], "default": "auto"},
                },
                "required": ["ticker"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_rsi_metrics",
            "description": "Get RSI threshold strategy metrics, optionally filtered by ticker.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_rsi_equity_curve",
            "description": "Get RSI threshold strategy equity curve records for one ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "max_rows": {"type": "integer", "minimum": 1, "maximum": 500},
                    "data_scope": {"type": "string", "enum": ["auto", "project", "workspace"], "default": "auto"},
                },
                "required": ["ticker"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_strategy_comparison",
            "description": "Get the unified strategy comparison table across Buy & Hold, Moving Average, RSI, and Portfolio CEM when available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "data_scope": {"type": "string", "enum": ["auto", "project", "workspace"], "default": "auto"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_portfolio_env_smoke_test",
            "description": "Run a lightweight smoke test for the multi-asset PortfolioEnv using equal-weight actions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {"type": "array", "items": {"type": "string"}},
                    "data_scope": {"type": "string", "enum": ["auto", "project", "workspace"], "default": "auto"},
                    "max_steps": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_portfolio_rl_metrics",
            "description": "Get saved metrics for the lightweight multi-asset portfolio RL/CEM training run.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_scope": {"type": "string", "enum": ["auto", "project", "workspace"], "default": "auto"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_portfolio_rl_equity_curve",
            "description": "Get the saved equity curve from the lightweight multi-asset portfolio RL/CEM training run.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_rows": {"type": "integer", "minimum": 1, "maximum": 5000, "default": 1000},
                    "data_scope": {"type": "string", "enum": ["auto", "project", "workspace"], "default": "auto"},
                },
                "additionalProperties": False,
            },
        },
    },
]


WRITE_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "run_buy_hold_baseline",
            "description": "Run or rerun the Buy & Hold baseline strategy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of tickers. If omitted, run all available tickers.",
                    },
                    "initial_cash": {"type": "number", "default": 100000},
                    "data_scope": {"type": "string", "enum": ["auto", "project", "workspace"], "default": "auto"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_ma_baseline",
            "description": "Run or rerun the Moving Average crossover baseline strategy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of tickers. If omitted, run all available tickers.",
                    },
                    "initial_cash": {"type": "number", "default": 100000},
                    "short_window": {"type": "integer", "default": 5},
                    "long_window": {"type": "integer", "default": 20},
                    "data_scope": {"type": "string", "enum": ["auto", "project", "workspace"], "default": "auto"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_rsi_baseline",
            "description": "Run or rerun the RSI threshold baseline strategy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of tickers. If omitted, run all available tickers.",
                    },
                    "initial_cash": {"type": "number", "default": 100000},
                    "buy_threshold": {"type": "number", "default": 30},
                    "sell_threshold": {"type": "number", "default": 70},
                    "data_scope": {"type": "string", "enum": ["auto", "project", "workspace"], "default": "auto"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_strategy_comparison",
            "description": "Build or rebuild the unified strategy comparison table across traditional baselines and Portfolio CEM when available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_scope": {"type": "string", "enum": ["project", "workspace"], "default": "project"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_portfolio_cem_training",
            "description": "Train and evaluate a lightweight multi-asset portfolio policy using Cross-Entropy Method. For follow-up commands after a Chat workspace has been prepared, omit tickers or use data_scope='workspace' so the current workspace universe is used.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of at least two tickers. If omitted, train on the selected data scope.",
                    },
                    "data_scope": {"type": "string", "enum": ["auto", "project", "workspace"], "default": "auto"},
                    "initial_cash": {"type": "number", "default": 100000},
                    "transaction_cost_pct": {"type": "number", "default": 0.001},
                    "generations": {"type": "integer", "minimum": 1, "maximum": 50, "default": 4},
                    "population_size": {"type": "integer", "minimum": 4, "maximum": 200, "default": 12},
                    "elite_fraction": {"type": "number", "minimum": 0.05, "maximum": 0.8, "default": 0.25},
                    "noise_scale": {"type": "number", "minimum": 0.001, "maximum": 5, "default": 0.2},
                    "train_ratio": {"type": "number", "minimum": 0.3, "maximum": 0.9, "default": 0.7},
                    "random_seed": {"type": "integer", "default": 42},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "prepare_ticker_analysis",
            "description": "Automatically prepare workspace data for known tickers, generate features, run baseline strategies, optionally create charts, and optionally push the dataset to app pages. Use this instead of asking the user whether to download missing ordinary market data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {"type": "array", "items": {"type": "string"}},
                    "start_date": {"type": "string", "description": "Optional YYYY-MM-DD start date."},
                    "end_date": {"type": "string", "description": "Optional YYYY-MM-DD end date."},
                    "lookback_period": {"type": "string", "default": "1y", "description": "Used when dates are omitted, such as 6mo, 1y, 2y, or 5y."},
                    "include_chart": {"type": "boolean", "default": True},
                    "push_to_app_pages": {"type": "boolean", "default": True},
                    "run_baseline_after": {"type": "boolean", "default": True},
                    "data_source": {"type": "string", "enum": ["auto", "download"], "default": "auto"},
                    "use_proxy": {"type": "boolean", "default": False},
                },
                "required": ["tickers"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_theme_candidates",
            "description": "Automatically screen a theme or industry, load the shortlist into the current Chat workspace, run feature engineering and baselines, create charts, and return records for final analysis. Use this for questions like promising chip stocks or AI-related stock candidates without asking the user to manage data refreshes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Theme, industry, or search phrase, such as semiconductor, chip technology, AI hardware, banks, or renewable energy."},
                    "lookback_period": {"type": "string", "default": "1y"},
                    "max_candidates": {"type": "integer", "minimum": 10, "maximum": 300, "default": 80},
                    "shortlist_size": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                    "min_avg_volume": {"type": "number", "default": 500000},
                    "min_price": {"type": "number", "default": 5},
                    "sort_by": {
                        "type": "string",
                        "enum": ["risk_adjusted_return", "total_return", "annualized_return", "low_drawdown", "volume"],
                        "default": "risk_adjusted_return",
                    },
                    "include_chart": {"type": "boolean", "default": True},
                    "push_to_app_pages": {"type": "boolean", "default": True},
                    "run_baseline_after": {"type": "boolean", "default": True},
                    "data_source": {"type": "string", "enum": ["auto", "download"], "default": "auto"},
                    "use_proxy": {"type": "boolean", "default": False},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refresh_llm_workspace_data",
            "description": "Load market data into the current Chat workspace and regenerate workspace features without modifying the main project dataset. Use automatically when lower-level data refresh is needed. Raw files are stored in the active runtime cache.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Any valid yfinance ticker symbols, such as AAPL, NVDA, META, SPY, or QQQ.",
                    },
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "use_proxy": {"type": "boolean", "default": False},
                    "run_baseline_after": {"type": "boolean", "default": True},
                    "data_source": {
                        "type": "string",
                        "enum": ["auto", "download"],
                        "default": "auto",
                        "description": "auto uses valid runtime raw files first, then downloads if needed.",
                    },
                },
                "required": ["tickers", "start_date", "end_date"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_shortlist_for_analysis",
            "description": "Load a screened shortlist into the current Chat workspace, generate features, and optionally run baseline strategies. Use automatically after candidate screening when deeper analysis is needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {"type": "array", "items": {"type": "string"}},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "use_proxy": {"type": "boolean", "default": False},
                    "data_source": {"type": "string", "enum": ["auto", "download"], "default": "auto"},
                    "run_baseline_after": {"type": "boolean", "default": True},
                },
                "required": ["tickers", "start_date", "end_date"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refresh_llm_workspace_ticker",
            "description": "Load one validated ticker into the current Chat workspace and regenerate workspace features without modifying the main project dataset. Use automatically when a requested ticker is missing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Validated ticker symbol, such as TSLA."},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "use_proxy": {"type": "boolean", "default": False},
                    "run_baseline_after": {"type": "boolean", "default": True},
                    "data_source": {
                        "type": "string",
                        "enum": ["auto", "download"],
                        "default": "auto",
                    },
                },
                "required": ["ticker", "start_date", "end_date"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "push_llm_workspace_to_app_pages",
            "description": "Make the current AI Chat workspace processed dataset active in other app pages without overwriting the main project dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string", "description": "Optional short reason for pushing the dataset."}
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reset_app_pages_to_project_dataset",
            "description": "Reset Web app pages to show the main project processed dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string", "description": "Optional short reason for resetting the dataset."}
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_new_tool",
            "description": "Create a non-executable tool proposal when the current toolset cannot satisfy the user's request well. This saves a JSON proposal for developer review; it does not dynamically run or install code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string", "description": "Proposed Python function name."},
                    "user_need": {"type": "string", "description": "What user need this tool would address."},
                    "inputs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Suggested input parameters and brief meanings.",
                    },
                    "outputs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Suggested output fields or artifacts.",
                    },
                    "required_data": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Data needed by the tool.",
                    },
                    "implementation_plan": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Implementation steps.",
                    },
                    "safety_notes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Safety, cost, privacy, or deployment notes.",
                    },
                    "suggested_python_code": {"type": "string"},
                    "suggested_tool_schema": {"type": "object"},
                },
                "required": ["tool_name", "user_need"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "promote_tool_proposal_local",
            "description": "Local developer-only command that promotes a reviewed proposal into generated tool files. It is disabled unless ENABLE_LOCAL_TOOL_PROMOTION=true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "proposal_id": {"type": "string", "description": "Proposal id to find under runtime/report proposal directories."},
                    "proposal_file": {"type": "string", "description": "Direct path to a proposal JSON file."},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "register_temp_composite_tool",
            "description": "Register a temporary session-scoped tool by safely composing an approved base tool with preset arguments, filters, sorting, selected columns, and a row limit. This does not write or execute Python code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string"},
                    "base_tool": {
                        "type": "string",
                        "enum": [
                            "screen_stock_candidates",
                            "get_strategy_comparison",
                            "get_eda_summary",
                            "get_data_quality_summary",
                            "get_ticker_history",
                            "get_buy_hold_metrics",
                            "get_ma_metrics",
                            "get_rsi_metrics",
                        ],
                    },
                    "description": {"type": "string"},
                    "preset_arguments": {"type": "object"},
                    "filters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {"type": "string"},
                                "operator": {"type": "string", "enum": [">", ">=", "<", "<=", "==", "!=", "contains", "gt", "gte", "lt", "lte", "ne"]},
                                "value": {},
                            },
                            "required": ["column", "operator", "value"],
                            "additionalProperties": False,
                        },
                    },
                    "select_columns": {"type": "array", "items": {"type": "string"}},
                    "sort_by": {"type": "string"},
                    "sort_ascending": {"type": "boolean", "default": False},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 20},
                },
                "required": ["tool_name", "base_tool"],
                "additionalProperties": False,
            },
        },
    },
]


TOOL_FUNCTIONS = {
    "get_project_capabilities": market_tools.get_project_capabilities,
    "get_dataset_status": market_tools.get_dataset_status,
    "get_runtime_storage_status": market_tools.get_runtime_storage_status,
    "list_local_raw_data": market_tools.list_local_raw_data,
    "get_local_data_inventory": market_tools.get_local_data_inventory,
    "get_llm_workspace_status": market_tools.get_llm_workspace_status,
    "get_active_analysis_dataset_status": market_tools.get_active_analysis_dataset_status,
    "list_tool_proposals": market_tools.list_tool_proposals,
    "list_temp_composite_tools": market_tools.list_temp_composite_tools,
    "get_tool_promotion_status": market_tools.get_tool_promotion_status,
    "search_us_symbols": market_tools.search_us_symbols,
    "validate_ticker_candidates": market_tools.validate_ticker_candidates,
    "inspect_local_raw_ticker": market_tools.inspect_local_raw_ticker,
    "get_eda_summary": market_tools.get_eda_summary,
    "get_data_quality_summary": market_tools.get_data_quality_summary,
    "list_available_figures": market_tools.list_available_figures,
    "get_ticker_history": market_tools.get_ticker_history,
    "create_ticker_price_chart": market_tools.create_ticker_price_chart,
    "get_ticker_metrics": market_tools.get_ticker_metrics,
    "screen_stock_candidates": market_tools.screen_stock_candidates,
    "get_buy_hold_metrics": market_tools.get_buy_hold_metrics,
    "get_buy_hold_equity_curve": market_tools.get_buy_hold_equity_curve,
    "get_ma_metrics": market_tools.get_ma_metrics,
    "get_ma_equity_curve": market_tools.get_ma_equity_curve,
    "get_rsi_metrics": market_tools.get_rsi_metrics,
    "get_rsi_equity_curve": market_tools.get_rsi_equity_curve,
    "get_strategy_comparison": market_tools.get_strategy_comparison,
    "run_portfolio_env_smoke_test": market_tools.run_portfolio_env_smoke_test,
    "run_portfolio_cem_training": market_tools.run_portfolio_cem_training,
    "get_portfolio_rl_metrics": market_tools.get_portfolio_rl_metrics,
    "get_portfolio_rl_equity_curve": market_tools.get_portfolio_rl_equity_curve,
    "run_buy_hold_baseline": market_tools.run_buy_hold_baseline,
    "run_ma_baseline": market_tools.run_ma_baseline,
    "run_rsi_baseline": market_tools.run_rsi_baseline,
    "run_strategy_comparison": market_tools.run_strategy_comparison,
    "prepare_ticker_analysis": market_tools.prepare_ticker_analysis,
    "analyze_theme_candidates": market_tools.analyze_theme_candidates,
    "refresh_llm_workspace_data": market_tools.refresh_llm_workspace_data,
    "refresh_llm_workspace_ticker": market_tools.refresh_llm_workspace_ticker,
    "load_shortlist_for_analysis": market_tools.load_shortlist_for_analysis,
    "push_llm_workspace_to_app_pages": market_tools.push_llm_workspace_to_app_pages,
    "reset_app_pages_to_project_dataset": market_tools.reset_app_pages_to_project_dataset,
    "propose_new_tool": market_tools.propose_new_tool,
    "promote_tool_proposal_local": market_tools.promote_tool_proposal_local,
    "register_temp_composite_tool": market_tools.register_temp_composite_tool,
}
TOOL_FUNCTIONS.update(GENERATED_TOOL_FUNCTIONS)


def refresh_generated_tool_registry():
    global GENERATED_CHAT_WORKSPACE_TOOL_NAMES, GENERATED_TOOL_FUNCTIONS, GENERATED_TOOL_SCHEMAS, GENERATED_WRITE_TOOL_NAMES

    try:
        import src.llm.generated_tool_schemas as generated_schema_module
        import src.tools.generated_market_tools as generated_tool_module

        generated_schema_module = importlib.reload(generated_schema_module)
        generated_tool_module = importlib.reload(generated_tool_module)
    except Exception:
        return False

    GENERATED_CHAT_WORKSPACE_TOOL_NAMES = generated_tool_module.GENERATED_CHAT_WORKSPACE_TOOL_NAMES
    GENERATED_TOOL_FUNCTIONS = generated_tool_module.GENERATED_TOOL_FUNCTIONS
    GENERATED_TOOL_SCHEMAS = generated_schema_module.GENERATED_TOOL_SCHEMAS
    GENERATED_WRITE_TOOL_NAMES = generated_tool_module.GENERATED_WRITE_TOOL_NAMES
    TOOL_FUNCTIONS.update(GENERATED_TOOL_FUNCTIONS)
    return True


def get_provider_config(provider_name, custom_base_url=None, custom_model=None):
    preset = PROVIDER_PRESETS[provider_name].copy()
    if custom_base_url:
        preset["base_url"] = custom_base_url
    if custom_model:
        preset["model"] = custom_model
    return preset


def build_tool_schemas(allow_write_tools=False):
    refresh_generated_tool_registry()
    generated_schemas = [
        schema for schema in GENERATED_TOOL_SCHEMAS
        if allow_write_tools or schema.get("function", {}).get("name") not in GENERATED_WRITE_TOOL_NAMES
    ]
    temp_schemas = market_tools.get_temp_composite_tool_schemas(allow_write_tools=allow_write_tools)
    read_schemas = TOOL_SCHEMAS + generated_schemas + temp_schemas
    if allow_write_tools:
        return read_schemas + WRITE_TOOL_SCHEMAS
    return read_schemas


def execute_tool_call(name, arguments, allow_write_tools=False, chat_id=None):
    refresh_generated_tool_registry()
    write_tool_names = {
        "run_buy_hold_baseline",
        "run_ma_baseline",
        "run_rsi_baseline",
        "run_strategy_comparison",
        "run_portfolio_env_smoke_test",
        "run_portfolio_cem_training",
        "screen_stock_candidates",
        "prepare_ticker_analysis",
        "analyze_theme_candidates",
        "refresh_llm_workspace_data",
        "refresh_llm_workspace_ticker",
        "load_shortlist_for_analysis",
        "push_llm_workspace_to_app_pages",
        "reset_app_pages_to_project_dataset",
        "propose_new_tool",
        "promote_tool_proposal_local",
        "register_temp_composite_tool",
    }
    write_tool_names.update(GENERATED_WRITE_TOOL_NAMES)
    if name in write_tool_names and not allow_write_tools:
        return {"error": f"Tool {name} is disabled because write tools are not allowed."}

    if name not in TOOL_FUNCTIONS and market_tools.is_temp_composite_tool(name):
        return market_tools.execute_temp_composite_tool(name, arguments=arguments, chat_id=chat_id)

    if name not in TOOL_FUNCTIONS:
        return {"error": f"Unknown tool: {name}"}

    chat_workspace_tool_names = {
        "get_llm_workspace_status",
        "get_local_data_inventory",
        "get_ticker_history",
        "create_ticker_price_chart",
        "get_buy_hold_metrics",
        "get_buy_hold_equity_curve",
        "get_ma_metrics",
        "get_ma_equity_curve",
        "get_rsi_metrics",
        "get_rsi_equity_curve",
        "get_strategy_comparison",
        "get_portfolio_rl_metrics",
        "get_portfolio_rl_equity_curve",
        "run_buy_hold_baseline",
        "run_ma_baseline",
        "run_rsi_baseline",
        "run_strategy_comparison",
        "run_portfolio_env_smoke_test",
        "run_portfolio_cem_training",
        "screen_stock_candidates",
        "prepare_ticker_analysis",
        "analyze_theme_candidates",
        "refresh_llm_workspace_data",
        "refresh_llm_workspace_ticker",
        "load_shortlist_for_analysis",
        "push_llm_workspace_to_app_pages",
        "propose_new_tool",
        "register_temp_composite_tool",
        "clear_llm_workspace",
        "merge_llm_workspace_to_project",
    }
    chat_workspace_tool_names.update(GENERATED_CHAT_WORKSPACE_TOOL_NAMES)
    if name in chat_workspace_tool_names and chat_id and "chat_id" not in arguments:
        arguments = dict(arguments)
        arguments["chat_id"] = chat_id

    function = TOOL_FUNCTIONS[name]
    return function(**arguments)


def list_ollama_models(base_url="http://localhost:11434"):
    url = f"{base_url.rstrip('/')}/api/tags"
    request = Request(url=url, method="GET")
    try:
        with urlopen(request, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"available": False, "models": [], "error": str(exc)}

    models = [item.get("name") for item in data.get("models", []) if item.get("name")]
    return {"available": True, "models": models, "error": None}


def list_openai_compatible_models(base_url, api_key=None):
    if not base_url:
        return {"available": False, "models": [], "error": "base_url is required"}

    url = f"{base_url.rstrip('/')}/models"
    request = Request(
        url=url,
        headers={
            "Authorization": f"Bearer {api_key or 'local-placeholder'}",
            "Content-Type": "application/json",
        },
        method="GET",
    )

    try:
        with urlopen(request, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"available": False, "models": [], "error": str(exc)}

    model_items = data.get("data", [])
    models = []
    for item in model_items:
        if isinstance(item, dict) and item.get("id"):
            models.append(item["id"])

    return {"available": bool(models), "models": models, "error": None if models else "No models returned"}


def _compact_tool_result(result, max_records=20):
    if isinstance(result, dict):
        compacted = {}
        for key, value in result.items():
            if key == "records" and isinstance(value, list):
                compacted[key] = value[:max_records]
                compacted["record_count"] = len(value)
                compacted["records_truncated"] = len(value) > max_records
            elif isinstance(value, list) and len(value) > max_records:
                compacted[key] = value[:max_records]
                compacted[f"{key}_count"] = len(value)
                compacted[f"{key}_truncated"] = True
            else:
                compacted[key] = value
        return compacted
    return result


def _save_debug_log(log_payload):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"llm_tool_call_{timestamp}.json"
    with log_path.open("w", encoding="utf-8") as file:
        json.dump(log_payload, file, ensure_ascii=False, indent=2, default=str)
    return str(log_path)


def call_openai_compatible_chat(base_url, api_key, payload, timeout=60):
    if not base_url:
        raise ValueError("base_url is required.")
    api_key = api_key or "local-placeholder"

    url = f"{base_url.rstrip('/')}/chat/completions"
    request = Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"LLM API connection failed: {exc}") from exc


def run_llm_tool_chat(
    question,
    api_key,
    provider_name="OpenAI",
    model=None,
    base_url=None,
    memory_summary=None,
    context_messages=None,
    allow_write_tools=True,
    temperature=0.2,
    max_tool_rounds=2,
    request_timeout=60,
    max_elapsed_seconds=0,
    save_debug_log=False,
    compact_tool_results=True,
    chat_id=None,
):
    provider = get_provider_config(provider_name, custom_base_url=base_url, custom_model=model)
    if not provider["model"]:
        raise ValueError("model is required.")

    max_tool_rounds = int(max_tool_rounds) if max_tool_rounds is not None else 0
    request_timeout = int(request_timeout)
    max_elapsed_seconds = int(max_elapsed_seconds or 0)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if memory_summary:
        messages.append(
            {
                "role": "system",
                "content": f"Conversation memory summary for this chat:\n{memory_summary}",
            }
        )

    for message in context_messages or []:
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": question})
    tool_schemas = build_tool_schemas(allow_write_tools=allow_write_tools)
    executed_tools = []
    debug_events = []
    started_at = time.monotonic()
    round_index = 0

    while max_tool_rounds <= 0 or round_index < max_tool_rounds:
        if max_elapsed_seconds > 0 and time.monotonic() - started_at >= max_elapsed_seconds:
            break

        round_index += 1
        payload = {
            "model": provider["model"],
            "messages": messages,
            "tools": tool_schemas,
            "tool_choice": "auto",
            "temperature": temperature,
        }
        response = call_openai_compatible_chat(
            base_url=provider["base_url"],
            api_key=api_key,
            payload=payload,
            timeout=request_timeout,
        )
        message = response["choices"][0]["message"]
        tool_calls = message.get("tool_calls") or []
        debug_events.append(
            {
                "round": round_index,
                "request": {
                    "model": provider["model"],
                    "message_count": len(messages),
                    "tool_count": len(tool_schemas),
                    "temperature": temperature,
                    "request_timeout": request_timeout,
                },
                "response_message": message,
            }
        )

        if not tool_calls:
            log_path = None
            if save_debug_log:
                log_path = _save_debug_log(
                    {
                        "question": question,
                        "provider": provider_name,
                        "model": provider["model"],
                        "allow_write_tools": allow_write_tools,
                        "max_tool_rounds": max_tool_rounds,
                        "request_timeout": request_timeout,
                        "max_elapsed_seconds": max_elapsed_seconds,
                        "tools": executed_tools,
                        "events": debug_events,
                        "final_answer": message.get("content", ""),
                    }
                )
            return {
                "answer": message.get("content", ""),
                "provider": provider_name,
                "model": provider["model"],
                "tools": executed_tools,
                "log_path": log_path,
                "raw_response": response,
            }

        messages.append(message)
        for tool_call in tool_calls:
            function_call = tool_call["function"]
            name = function_call["name"]
            raw_arguments = function_call.get("arguments") or "{}"
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {}

            result = execute_tool_call(
                name=name,
                arguments=arguments,
                allow_write_tools=allow_write_tools,
                chat_id=chat_id,
            )
            compacted_result = _compact_tool_result(result) if compact_tool_results else result
            executed_tools.append({"name": name, "arguments": arguments, "result": compacted_result})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(compacted_result, ensure_ascii=False, default=str),
                }
            )

    log_path = None
    if save_debug_log:
        if max_elapsed_seconds > 0 and time.monotonic() - started_at >= max_elapsed_seconds:
            final_answer = "The assistant reached the configured maximum runtime."
        else:
            final_answer = "The assistant reached the maximum number of tool-calling rounds."
        log_path = _save_debug_log(
            {
                "question": question,
                "provider": provider_name,
                "model": provider["model"],
                "allow_write_tools": allow_write_tools,
                "max_tool_rounds": max_tool_rounds,
                "request_timeout": request_timeout,
                "max_elapsed_seconds": max_elapsed_seconds,
                "tools": executed_tools,
                "events": debug_events,
                "final_answer": final_answer,
            }
        )

    if max_elapsed_seconds > 0 and time.monotonic() - started_at >= max_elapsed_seconds:
        answer = "The assistant reached the configured maximum runtime."
    else:
        answer = "The assistant reached the maximum number of tool-calling rounds."

    return {
        "answer": answer,
        "provider": provider_name,
        "model": provider["model"],
        "tools": executed_tools,
        "log_path": log_path,
    }
