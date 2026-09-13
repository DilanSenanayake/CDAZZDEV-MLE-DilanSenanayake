"""
Task 1B  LLM sentiment analysis and Buy/Hold/Sell signal reasoning.
Uses Groq free-tier inference with Pydantic validation.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from .prompts import SENTIMENT_SYSTEM, SENTIMENT_USER, SIGNAL_SYSTEM, SIGNAL_USER
    from .schemas import HeadlineSentiment, SentimentBatchResult, TradeSignal
except ImportError:  # script-style execution from task1_financial/
    from prompts import SENTIMENT_SYSTEM, SENTIMENT_USER, SIGNAL_SYSTEM, SIGNAL_USER
    from schemas import HeadlineSentiment, SentimentBatchResult, TradeSignal

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
SENTIMENT_WEIGHT = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


def has_groq_key() -> bool:
    key = os.getenv("GROQ_API_KEY", "").strip()
    return bool(key) and not key.startswith("your_")


def _client() -> Groq:
    if not has_groq_key():
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def _offline_headline_sentiment(headline: str) -> HeadlineSentiment:
    """Deterministic lexicon fallback when no API key (keeps notebooks executable)."""
    text = headline.lower()
    pos = ("beat", "surge", "record", "growth", "upgrade", "rally", "profit", "win")
    neg = ("miss", "fall", "cut", "probe", "lawsuit", "downgrade", "loss", "risk")
    p = sum(1 for w in pos if w in text)
    n = sum(1 for w in neg if w in text)
    if p > n:
        return HeadlineSentiment(
            headline=headline,
            sentiment="positive",
            confidence=0.55,
            brief_reason="Offline lexicon matched bullish cue words.",
        )
    if n > p:
        return HeadlineSentiment(
            headline=headline,
            sentiment="negative",
            confidence=0.55,
            brief_reason="Offline lexicon matched bearish cue words.",
        )
    return HeadlineSentiment(
        headline=headline,
        sentiment="neutral",
        confidence=0.4,
        brief_reason="Offline lexicon found no strong polarity cues.",
    )


def _offline_trade_signal(summary: dict[str, Any], sentiment_score: float) -> TradeSignal:
    mom = (summary.get("momentum_signal") or "neutral").lower()
    rsi = (summary.get("indicators") or {}).get("rsi_14")
    if mom == "bullish" and sentiment_score >= 0:
        signal = "Buy"
    elif mom == "bearish" and sentiment_score <= 0:
        signal = "Sell"
    else:
        signal = "Hold"
    return TradeSignal(
        signal=signal,  # type: ignore[arg-type]
        justification=(
            f"Price versus SMA50/SMA200 alignment currently reads {mom}, while RSI "
            f"is {rsi}. Combining trend structure with MACD histogram direction and "
            f"Bollinger location suggests balancing momentum against mean-reversion "
            f"risk. Aggregated news sentiment score is {sentiment_score:.2f}, which "
            f"moderates conviction. Overall the blended technical and sentiment view "
            f"supports a {signal} stance rather than chasing a single indicator."
        ),
        key_factors=["sma_stack", "rsi_regime", "macd_hist", "news_sentiment"],
    )


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


@retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
def _chat_json(system: str, user: str, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    client = _client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or ""
    return _extract_json(content)


def analyze_headline(headline: str, *, allow_offline: bool = True) -> HeadlineSentiment | None:
    """Classify one headline; return None on persistent validation failure."""
    if not has_groq_key():
        if allow_offline:
            logger.warning("No GROQ_API_KEY  using offline sentiment fallback")
            return _offline_headline_sentiment(headline)
        return None
    user = SENTIMENT_USER.format(headline=headline)
    try:
        raw = _chat_json(SENTIMENT_SYSTEM, user)
        raw.setdefault("headline", headline)
        return HeadlineSentiment.model_validate(raw)
    except (ValidationError, json.JSONDecodeError, Exception) as exc:  # noqa: BLE001
        logger.error("Sentiment validation/parse failed for %r: %s", headline[:80], exc)
        return None


def aggregate_sentiment(items: list[HeadlineSentiment]) -> SentimentBatchResult:
    if not items:
        return SentimentBatchResult(
            items=[], overall_sentiment_score=0.0, overall_label="neutral"
        )
    weighted = [
        SENTIMENT_WEIGHT[i.sentiment] * float(i.confidence) for i in items
    ]
    confidences = [float(i.confidence) for i in items]
    denom = sum(confidences) or 1.0
    score = sum(weighted) / denom
    if score > 0.15:
        label = "positive"
    elif score < -0.15:
        label = "negative"
    else:
        label = "neutral"
    return SentimentBatchResult(
        items=items, overall_sentiment_score=float(score), overall_label=label
    )


def analyze_headlines(headlines: list[str]) -> SentimentBatchResult:
    parsed: list[HeadlineSentiment] = []
    for h in headlines:
        item = analyze_headline(h)
        if item is None:
            # Graceful fallback: neutral low-confidence stub so pipeline continues
            parsed.append(
                HeadlineSentiment(
                    headline=h,
                    sentiment="neutral",
                    confidence=0.1,
                    brief_reason="Validation failed; treated as neutral.",
                )
            )
        else:
            parsed.append(item)
    return aggregate_sentiment(parsed)


def generate_trade_signal(
    ticker: str,
    summary: dict[str, Any],
    sentiment_score: float,
    *,
    allow_offline: bool = True,
) -> TradeSignal | None:
    if not has_groq_key():
        if allow_offline:
            logger.warning("No GROQ_API_KEY - using offline trade-signal fallback")
            return _offline_trade_signal(summary, sentiment_score)
        return None
    user = SIGNAL_USER.format(
        ticker=ticker,
        summary_json=json.dumps(
            {k: v for k, v in summary.items() if k != "indicators"},
            default=str,
            indent=2,
        ),
        indicators_json=json.dumps(summary.get("indicators", {}), default=str, indent=2),
        sentiment_score=sentiment_score,
    )
    try:
        raw = _chat_json(SIGNAL_SYSTEM, user)
        return TradeSignal.model_validate(raw)
    except (ValidationError, json.JSONDecodeError, Exception) as exc:  # noqa: BLE001
        logger.error("Trade signal validation failed: %s", exc)
        return None


def run_llm_reasoning(
    ticker: str,
    summary: dict[str, Any],
    headlines: list[str],
) -> dict[str, Any]:
    sentiment = analyze_headlines(headlines)
    signal = generate_trade_signal(ticker, summary, sentiment.overall_sentiment_score)
    return {
        "mode": "groq" if has_groq_key() else "offline_fallback",
        "sentiment": sentiment.model_dump(),
        "signal": signal.model_dump() if signal else None,
    }
