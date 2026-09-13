"""
Task 3 tools - wrappers around Task 1 logic + volatility + web search.
Each tool logs to agent_trace.jsonl via the observability helper.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TASK1 = ROOT / "task1_financial"
if str(TASK1) not in sys.path:
    sys.path.insert(0, str(TASK1))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_pipeline import (  # noqa: E402
    add_indicators,
    fetch_news_headlines,
    fetch_ohlcv,
)
from llm_reasoning import analyze_headlines  # noqa: E402

from task3_agentic.observability import traced  # noqa: E402


@traced
def get_price_data(ticker: str, period: str = "2y") -> dict[str, Any]:
    """Return OHLCV with computed indicators (latest snapshot + series stats)."""
    df = add_indicators(fetch_ohlcv(ticker, period=period))
    latest = df.iloc[-1]
    return {
        "ticker": ticker.upper(),
        "period": period,
        "rows": int(len(df)),
        "latest_close": float(latest["close"]) if pd.notna(latest["close"]) else None,
        "sma_50": float(latest["sma_50"]) if pd.notna(latest.get("sma_50")) else None,
        "sma_200": float(latest["sma_200"]) if pd.notna(latest.get("sma_200")) else None,
        "rsi_14": float(latest["rsi_14"]) if pd.notna(latest.get("rsi_14")) else None,
        "macd": float(latest["macd"]) if pd.notna(latest.get("macd")) else None,
        "macd_hist": float(latest["macd_hist"])
        if pd.notna(latest.get("macd_hist"))
        else None,
        "bb_upper": float(latest["bb_upper"]) if pd.notna(latest.get("bb_upper")) else None,
        "bb_lower": float(latest["bb_lower"]) if pd.notna(latest.get("bb_lower")) else None,
    }


@traced
def get_news(ticker: str, n: int = 10) -> list[dict[str, str]]:
    headlines = fetch_news_headlines(ticker, min_count=n)
    if not headlines:
        return [{"headline": f"No news found for {ticker}", "error": "empty"}]
    return [{"headline": h} for h in headlines[:n]]


@traced
def calculate_volatility(ticker: str, window: int = 21) -> dict[str, Any]:
    """Annualised historical volatility from daily log returns."""
    df = fetch_ohlcv(ticker, period="1y")
    close = df["close"].dropna()
    if len(close) < window + 1:
        return {"ticker": ticker, "error": "insufficient_data", "window": window}
    log_ret = np.log(close / close.shift(1)).dropna()
    daily_std = float(log_ret.tail(window).std(ddof=1))
    ann = daily_std * math.sqrt(252)
    return {
        "ticker": ticker.upper(),
        "window": window,
        "daily_std": daily_std,
        "annualised_volatility": ann,
    }


@traced
def llm_sentiment(headlines: list[str] | str) -> dict[str, Any]:
    if isinstance(headlines, str):
        try:
            headlines = json.loads(headlines)
        except json.JSONDecodeError:
            headlines = [headlines]
    if not headlines:
        return {"error": "empty_headlines", "overall_sentiment_score": 0.0}
    result = analyze_headlines(list(headlines))
    return result.model_dump()


@traced
def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return [{"title": "No results", "body": "", "href": "", "error": "empty"}]
        return [
            {
                "title": r.get("title", ""),
                "body": r.get("body", ""),
                "href": r.get("href", ""),
            }
            for r in results
        ]
    except Exception as exc:  # noqa: BLE001
        return [{"title": "web_search_failed", "body": str(exc), "href": "", "error": "exception"}]
