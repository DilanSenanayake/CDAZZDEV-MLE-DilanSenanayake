# Task 1 - Financial AI Equity Research Assistant

LLM-powered equity research pipeline for **AAPL** with structured outputs.

## Quick start

```bash
# from repo root
pip install -r requirements.txt
cp .env.example .env   # add GROQ_API_KEY for live LLM mode
python task1_financial/run_task1.py AAPL
```

Or open [`equity_research.ipynb`](equity_research.ipynb) in Jupyter / [Google Colab](https://colab.research.google.com/).

## Modules

| File | Role |
|------|------|
| `data_pipeline.py` | OHLCV (>=2y), indicators from first principles, news, summary dict |
| `schemas.py` | Pydantic models for sentiment + trade signal |
| `prompts.py` | System/user prompt templates (separated from business logic) |
| `llm_reasoning.py` | Groq inference + validation; offline fallback if no API key |
| `report.py` | Bonus Markdown/HTML brief with matplotlib chart |
| `run_task1.py` | End-to-end runner |

## Indicators (no TA-Lib)

- SMA 50 / 200
- RSI 14 (Wilder smoothing)
- MACD (12, 26, 9)
- Bollinger Bands (20, 2?)

## Outputs

See `reports/AAPL_research_brief.html` and `reports/AAPL_run_summary.json`.

## Notes

- Dates use yfinance `period="2y"` - no hardcoded calendar strings.
- Set `GROQ_API_KEY` for production LLM mode; without it the notebook still runs via validated offline fallback so cell outputs remain visible.
