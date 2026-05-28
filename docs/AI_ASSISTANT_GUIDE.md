# AI Assistant Guide

This guide explains how to use the AI Assistant page in FinRL Insight.

## 1. Purpose

The AI Assistant is a natural-language interface on top of the project's local analysis tools. It can help users:

* understand what the project can do
* inspect available market data
* load or refresh ticker data
* generate charts and EDA summaries
* run baseline strategies
* train the lightweight Portfolio CEM policy
* compare historical strategy performance
* screen theme-based stock candidates for further research

The assistant is not an investment advisor. Its answers should be treated as historical data analysis and educational output.

## 2. Provider Setup

The app supports OpenAI-compatible providers:

| Provider | API Key |
| --- | --- |
| OpenAI | Required |
| DeepSeek | Required |
| Alibaba Bailian / DashScope | Required |
| Ollama Local | Usually optional |
| LM Studio Local | Usually optional |
| Custom OpenAI-Compatible | Depends on server |

For Alibaba Bailian, a lightweight default model such as `qwen-plus` is usually enough for this project. For local Ollama, choose a locally installed model from the model dropdown or type its exact name.

The app can remember provider/model settings locally. API keys are saved only if the user explicitly enables local key remembering.

## 3. Recommended Questions

Capability overview:

```text
你可以做什么？
这个项目能解决哪些问题？
```

Data inspection:

```text
目前本地有哪些数据？
当前页面正在使用哪个数据集？
```

Ticker analysis:

```text
请分析特斯拉近一年的走势，并给出图表。
帮我比较 NVDA、AMD、QQQ 最近一年的策略表现。
```

Theme screening:

```text
帮我挖掘几支芯片科技相关股票，并比较它们。
找一些 AI 基础设施相关股票作为研究候选。
```

Strategy comparison:

```text
对当前数据运行 Buy & Hold、MA、RSI，并比较结果。
训练 Portfolio CEM，并和传统策略做对比。
```

Tool extension:

```text
你生成过哪些新工具草案？
把这个筛选逻辑注册成一个临时工具。
```

## 4. Data Behavior

Manual Data Setup loads data into the current Web session's main dataset.

AI-triggered analysis uses the current Chat workspace by default. This keeps AI-generated processed data, figures, and strategy results separate from the session's main loaded dataset.

When useful, the assistant can push the current Chat workspace to the normal app pages so Data Explorer, EDA Results, and Baseline Results show the AI-generated dataset without overwriting the main dataset.

Runtime data is stored under:

```text
.streamlit_runtime/sessions/<session_id>/
```

This is temporary Web-session data. It is suitable for Streamlit Community deployment because visitors do not depend on the developer's local `data/` files.

## 5. Background Jobs

LLM calls run as background jobs. After pressing `Ask LLM`, users can switch pages and return later. The current Chat will show a pending job until the answer is ready.

Limitations:

* If the whole Streamlit server process restarts, the background job is lost.
* Some remote model calls can still fail because of provider timeout, API key errors, rate limits, or unavailable models.
* Online market data may fail if Yahoo/yfinance rate-limits the server.

## 6. Debugging

Enable `Save debug log` to store LLM/tool-call logs. Logs include:

* question
* provider and model
* tool calls
* compacted tool results
* final answer metadata

Use these logs to verify whether the LLM called the expected tools and whether tool outputs were interpreted correctly.

## 7. Safety Notes

* All outputs are based on historical data and tool results.
* The system does not provide investment advice.
* Write tools are enabled in the app so the assistant can prepare data automatically, but outputs are written to runtime/session storage.
* Local tool promotion is disabled unless the developer explicitly enables it with `ENABLE_LOCAL_TOOL_PROMOTION=true`.
