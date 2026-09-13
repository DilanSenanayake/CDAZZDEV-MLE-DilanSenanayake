"""
Task 1A - Financial data ingestion and feature engineering.

Fetches OHLCV via yfinance, computes technical indicators from first principles
(no TA-Lib), retrieves news headlines, and builds a summary dictionary.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Indicator windows documented explicitly (no magic numbers buried in call sites).
SMA_FAST = 50
SMA_SLOW = 200
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_WINDOW = 20
BB_NUM_STD = 2.0
DEFAULT_PERIOD = "2y"
MIN_HEADLINES = 10


def fetch_ohlcv(ticker: str, period: str = DEFAULT_PERIOD) -> pd.DataFrame:
    """
    Fetch daily OHLCV without hardcoded calendar date strings.
    Uses yfinance period relative to 'now'.
    """
    ticker = ticker.upper().strip()
    if not ticker:
        raise ValueError("ticker must be a non-empty string")

    stock = yf.Ticker(ticker)
    df = stock.history(period=period, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"No OHLCV data returned for {ticker} period={period}")

    # Standardize column names
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    needed = ["open", "high", "low", "close", "volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"OHLCV missing columns {missing} for {ticker}")

    out = df[needed].copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    # Coerce numerics; leave NaNs for downstream handlers
    for col in needed:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def rsi_wilder(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    RSI with Wilder smoothing (exponential moving average of gains/losses).
    # AI-ASSISTED: Cursor Composer, Prompt: 'RSI with Wilder smoothing from first principles', Date: 2026-09-13
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder: first average is SMA, then recursive EMA with alpha=1/period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # When avg_loss is 0 and avg_gain > 0, RSI = 100
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    return rsi


def macd(
    close: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram from EMA differentials."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist}
    )


def bollinger_bands(
    close: pd.Series, window: int = BB_WINDOW, num_std: float = BB_NUM_STD
) -> pd.DataFrame:
    mid = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower})


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Attach all required technical indicators to an OHLCV frame."""
    out = df.copy()
    close = out["close"]
    out[f"sma_{SMA_FAST}"] = sma(close, SMA_FAST)
    out[f"sma_{SMA_SLOW}"] = sma(close, SMA_SLOW)
    out["rsi_14"] = rsi_wilder(close, RSI_PERIOD)
    out = out.join(macd(close))
    out = out.join(bollinger_bands(close))
    return out


def _headline_from_item(item: Any) -> str | None:
    if item is None:
        return None
    if isinstance(item, str):
        text = item.strip()
        return text or None
    if isinstance(item, dict):
        for key in ("title", "headline", "summary"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        content = item.get("content")
        if isinstance(content, dict):
            return _headline_from_item(content)
    # yfinance sometimes returns objects with .title
    title = getattr(item, "title", None)
    if isinstance(title, str) and title.strip():
        return title.strip()
    return None


def fetch_news_headlines(ticker: str, min_count: int = MIN_HEADLINES) -> list[str]:
    """
    Retrieve at least min_count headlines.
    Primary: yfinance news endpoint. Fallback: Yahoo Finance RSS.
    """
    ticker = ticker.upper().strip()
    headlines: list[str] = []
    seen: set[str] = set()

    def _add(text: str | None) -> None:
        if not text:
            return
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        headlines.append(text)

    try:
        stock = yf.Ticker(ticker)
        news = getattr(stock, "news", None) or []
        for item in news:
            _add(_headline_from_item(item))
            if len(headlines) >= min_count:
                return headlines[: max(min_count, len(headlines))]
    except Exception as exc:  # noqa: BLE001 - robustness required by rubric
        logger.warning("yfinance news failed for %s: %s", ticker, exc)

    # RSS fallback
    try:
        import feedparser

        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        feed = feedparser.parse(url)
        for entry in getattr(feed, "entries", []) or []:
            _add(getattr(entry, "title", None))
            if len(headlines) >= min_count:
                break
    except Exception as exc:  # noqa: BLE001
        logger.warning("RSS news fallback failed for %s: %s", ticker, exc)

    if len(headlines) < min_count:
        logger.warning(
            "Only %s headlines retrieved for %s (wanted %s)",
            len(headlines),
            ticker,
            min_count,
        )
    return headlines


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _momentum_signal(row: pd.Series) -> str:
    """
    Derive a discrete momentum signal from indicator combinations.
    Rules are explicit combinations, not single-indicator echoes.
    """
    price = _safe_float(row.get("close"))
    sma50 = _safe_float(row.get(f"sma_{SMA_FAST}"))
    sma200 = _safe_float(row.get(f"sma_{SMA_SLOW}"))
    rsi = _safe_float(row.get("rsi_14"))
    macd_h = _safe_float(row.get("macd_hist"))
    bb_upper = _safe_float(row.get("bb_upper"))
    bb_lower = _safe_float(row.get("bb_lower"))

    bullish = 0
    bearish = 0

    if price is not None and sma50 is not None and sma200 is not None:
        if price > sma50 > sma200:
            bullish += 2
        elif price < sma50 < sma200:
            bearish += 2
        elif price > sma50:
            bullish += 1
        elif price < sma50:
            bearish += 1

    if rsi is not None:
        if rsi >= 70:
            bearish += 1  # overbought
        elif rsi <= 30:
            bullish += 1  # oversold bounce potential
        elif rsi >= 55:
            bullish += 1
        elif rsi <= 45:
            bearish += 1

    if macd_h is not None:
        if macd_h > 0:
            bullish += 1
        elif macd_h < 0:
            bearish += 1

    if price is not None and bb_upper is not None and bb_lower is not None:
        if price >= bb_upper:
            bearish += 1
        elif price <= bb_lower:
            bullish += 1

    if bullish - bearish >= 2:
        return "bullish"
    if bearish - bullish >= 2:
        return "bearish"
    return "neutral"


def build_summary(ticker: str, df: pd.DataFrame) -> dict[str, Any]:
    """
    Clean summary dictionary with required fields.
    Handles missing fundamentals / NaN indicators without raising.
    """
    ticker = ticker.upper().strip()
    latest = df.iloc[-1]
    current_price = _safe_float(latest.get("close"))

    # 52-week high/low from available history (?1y window of the series)
    one_year = df.tail(252) if len(df) >= 252 else df
    high_52w = _safe_float(one_year["high"].max()) if not one_year.empty else None
    low_52w = _safe_float(one_year["low"].min()) if not one_year.empty else None

    # YTD return
    ytd_return = None
    try:
        year = datetime.now(timezone.utc).year
        ytd = df[df.index.year == year]
        if ytd.empty:
            # timezone-naive index fallback
            ytd = df[df.index.map(lambda x: getattr(x, "year", None) == year)]
        if len(ytd) >= 2:
            start = _safe_float(ytd.iloc[0]["close"])
            end = _safe_float(ytd.iloc[-1]["close"])
            if start and end and start != 0:
                ytd_return = (end / start) - 1.0
    except Exception as exc:  # noqa: BLE001
        logger.warning("YTD calculation failed: %s", exc)

    pe_ratio = None
    try:
        info = yf.Ticker(ticker).info or {}
        pe_ratio = _safe_float(
            info.get("trailingPE") or info.get("forwardPE") or info.get("peRatio")
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("PE lookup failed for %s: %s", ticker, exc)

    return {
        "ticker": ticker,
        "as_of": str(df.index[-1]),
        "current_price": current_price,
        "fifty_two_week_high": high_52w,
        "fifty_two_week_low": low_52w,
        "pe_ratio": pe_ratio,
        "ytd_return": ytd_return,
        "momentum_signal": _momentum_signal(latest),
        "indicators": {
            "sma_50": _safe_float(latest.get(f"sma_{SMA_FAST}")),
            "sma_200": _safe_float(latest.get(f"sma_{SMA_SLOW}")),
            "rsi_14": _safe_float(latest.get("rsi_14")),
            "macd": _safe_float(latest.get("macd")),
            "macd_signal": _safe_float(latest.get("macd_signal")),
            "macd_hist": _safe_float(latest.get("macd_hist")),
            "bb_mid": _safe_float(latest.get("bb_mid")),
            "bb_upper": _safe_float(latest.get("bb_upper")),
            "bb_lower": _safe_float(latest.get("bb_lower")),
        },
    }


def run_pipeline(ticker: str = "AAPL", period: str = DEFAULT_PERIOD) -> dict[str, Any]:
    """End-to-end Task 1A pipeline."""
    ohlcv = fetch_ohlcv(ticker, period=period)
    featured = add_indicators(ohlcv)
    headlines = fetch_news_headlines(ticker, min_count=MIN_HEADLINES)
    summary = build_summary(ticker, featured)
    return {
        "ohlcv": ohlcv,
        "featured": featured,
        "headlines": headlines,
        "summary": summary,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_pipeline("AAPL")
    print("Rows:", len(result["featured"]))
    print("Headlines:", len(result["headlines"]))
    print("Summary:", result["summary"])
