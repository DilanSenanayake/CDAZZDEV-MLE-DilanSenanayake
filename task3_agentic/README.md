# Task 3 - Agentic Financial Research System

Multi-agent research workflow with tools, memory, and observability.

## Quick start

```bash
pip install -r requirements.txt
python task3_agentic/run_task3.py AAPL
```

Notebook: [`multi_agent.ipynb`](multi_agent.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake/blob/master/task3_agentic/multi_agent.ipynb)

https://colab.research.google.com/github/DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake/blob/master/task3_agentic/multi_agent.ipynb

## Architecture

### Task 3A - Single research agent

Five tools (all traced):

1. `get_price_data`
2. `get_news`
3. `calculate_volatility`
4. `llm_sentiment`
5. `web_search`

Tool order is chosen from **observations** (what is still missing in memory), with explicit observe/replan events in the trace - not a blind fixed script.

### Task 3B - Multi-agent

| Agent | Tools | Output |
|-------|-------|--------|
| Data Analyst | price, volatility, sentiment | `DataAnalystBrief` (Pydantic) |
| Research Writer | web_search, get_news | `FinalResearchReport` |

Includes a visible **critique loop**: Writer asks once -> Analyst answers -> Writer incorporates.

### Task 3C - Memory & observability

- Short-term memory: follow-ups answered without re-fetch
- Persistent cache: `memory/cache/{TICKER}_{date}.json`
- Trace log: [`logs/agent_trace.jsonl`](logs/agent_trace.jsonl)

## Bonus dashboard

```bash
streamlit run task3_agentic/dashboard.py
```
