"""
Bonus  Render a one-page equity research brief (Markdown + HTML) with a chart.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import markdown as md_lib
import matplotlib.pyplot as plt
import pandas as pd


RISK_DISCLAIMER = (
    "Risk disclaimer: This brief is for educational/assessment purposes only and "
    "does not constitute investment advice, an offer, or a solicitation to buy or "
    "sell any security. Past performance and model-generated signals are not "
    "indicative of future results. Always conduct independent due diligence."
)


def _price_chart_base64(featured: pd.DataFrame, ticker: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(featured.index, featured["close"], label="Close", linewidth=1.2)
    if "sma_50" in featured.columns:
        ax.plot(featured.index, featured["sma_50"], label="SMA50", linewidth=1.0)
    if "sma_200" in featured.columns:
        ax.plot(featured.index, featured["sma_200"], label="SMA200", linewidth=1.0)
    ax.set_title(f"{ticker} Price with SMA50 / SMA200")
    ax.set_ylabel("Price")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def build_markdown_brief(
    ticker: str,
    summary: dict[str, Any],
    sentiment: dict[str, Any],
    signal: dict[str, Any] | None,
    headlines: list[str],
) -> str:
    ind = summary.get("indicators") or {}
    ytd = summary.get("ytd_return")
    ytd_s = f"{ytd * 100:.2f}%" if isinstance(ytd, (int, float)) else "N/A"
    pe = summary.get("pe_ratio")
    pe_s = f"{pe:.2f}" if isinstance(pe, (int, float)) else "N/A"

    top = (sentiment.get("items") or [])[:3]
    if not top:
        top_lines = [f"- {h}" for h in headlines[:3]]
    else:
        top_lines = [
            f"- [{i.get('sentiment')}] {i.get('headline')} "
            f"(conf={i.get('confidence'):.2f})"
            for i in top
        ]

    signal_block = "Signal unavailable (validation failed)."
    if signal:
        signal_block = (
            f"**Recommendation: {signal.get('signal')}**\n\n"
            f"{signal.get('justification')}\n\n"
            f"Key factors: {', '.join(signal.get('key_factors') or [])}"
        )

    return f"""# Equity Research Brief  {ticker}

## Company Snapshot
- **As of:** {summary.get('as_of')}
- **Current price:** {summary.get('current_price')}
- **52-week high / low:** {summary.get('fifty_two_week_high')} / {summary.get('fifty_two_week_low')}
- **P/E:** {pe_s}
- **YTD return:** {ytd_s}
- **Momentum signal (rules-based):** {summary.get('momentum_signal')}

## Technical Outlook
- SMA50={ind.get('sma_50')}, SMA200={ind.get('sma_200')}
- RSI(14)={ind.get('rsi_14')}
- MACD={ind.get('macd')}, Signal={ind.get('macd_signal')}, Hist={ind.get('macd_hist')}
- Bollinger mid/upper/lower={ind.get('bb_mid')} / {ind.get('bb_upper')} / {ind.get('bb_lower')}

## News Sentiment Summary
- **Overall label:** {sentiment.get('overall_label')}
- **Overall score:** {sentiment.get('overall_sentiment_score')}
- **Top headlines:**
{chr(10).join(top_lines)}

## LLM Recommendation
{signal_block}

## Risk Disclaimer
{RISK_DISCLAIMER}
"""


def render_html_report(
    ticker: str,
    summary: dict[str, Any],
    sentiment: dict[str, Any],
    signal: dict[str, Any] | None,
    headlines: list[str],
    featured: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    md = build_markdown_brief(ticker, summary, sentiment, signal, headlines)
    md_path = output_dir / f"{ticker}_research_brief.md"
    md_path.write_text(md, encoding="utf-8")

    chart_b64 = _price_chart_base64(featured.tail(252), ticker)
    body = md_lib.markdown(md)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Equity Research Brief  {ticker}</title>
  <style>
    body {{ font-family: Georgia, 'Times New Roman', serif; max-width: 820px;
           margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; line-height: 1.45; }}
    h1, h2 {{ font-family: 'Segoe UI', Helvetica, sans-serif; }}
    img {{ max-width: 100%; border: 1px solid #ddd; }}
    .chart {{ margin: 1.25rem 0; }}
  </style>
</head>
<body>
{body}
<div class="chart">
  <h2>Price Chart</h2>
  <img alt="price chart" src="data:image/png;base64,{chart_b64}"/>
</div>
</body>
</html>
"""
    html_path = output_dir / f"{ticker}_research_brief.html"
    html_path.write_text(html, encoding="utf-8")

    # Also save standalone PNG
    png_path = output_dir / f"{ticker}_price_chart.png"
    raw = base64.b64decode(chart_b64)
    png_path.write_bytes(raw)

    return {"markdown": md_path, "html": html_path, "chart": png_path}
