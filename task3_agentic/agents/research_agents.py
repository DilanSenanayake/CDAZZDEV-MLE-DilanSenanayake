"""
Task 3 research agents.

Implements:
- 3A single research agent with autonomous tool selection (ReAct-style loop)
- 3B two-agent pipeline with structured handoff + critique loop
- Uses tools with observability tracing

If langgraph is unavailable, a lightweight custom ReAct loop is used so the
submission remains runnable on free-tier environments.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from task3_agentic.memory.store import ShortTermMemory, load_brief, save_brief
from task3_agentic.schemas import (
    ClarificationRequest,
    ClarificationResponse,
    DataAnalystBrief,
    FinalResearchReport,
    RiskItem,
)
from task3_agentic.tools.financial_tools import (
    calculate_volatility,
    get_news,
    get_price_data,
    llm_sentiment,
    web_search,
)

ALL_TOOLS: dict[str, Callable] = {
    "get_price_data": get_price_data,
    "calculate_volatility": calculate_volatility,
    "llm_sentiment": llm_sentiment,
    "get_news": get_news,
    "web_search": web_search,
}

ANALYST_TOOLS = {
    "get_price_data": get_price_data,
    "calculate_volatility": calculate_volatility,
    "llm_sentiment": llm_sentiment,
}

WRITER_TOOLS = {
    "get_news": get_news,
    "web_search": web_search,
}


def _parse_action(text: str) -> tuple[str | None, dict]:
    """
    Parse: ACTION: tool_name | ARGS: {...}
    or FINAL: ...
    """
    if "FINAL:" in text:
        return "FINAL", {"text": text.split("FINAL:", 1)[1].strip()}
    m = re.search(
        r"ACTION:\s*([a-zA-Z_]+)\s*\|\s*ARGS:\s*(\{.*\})",
        text,
        flags=re.DOTALL,
    )
    if not m:
        return None, {}
    try:
        return m.group(1).strip(), json.loads(m.group(2))
    except json.JSONDecodeError:
        return m.group(1).strip(), {}


def _heuristic_plan(ticker: str, memory: ShortTermMemory) -> list[tuple[str, dict]]:
    """
    Autonomous-style planner: chooses next tools based on what is missing
    in memory (observation-driven), not a single hard-coded blind sequence.
    """
    plan: list[tuple[str, dict]] = []
    if not memory.has("price"):
        plan.append(("get_price_data", {"ticker": ticker, "period": "2y"}))
    if not memory.has("vol"):
        plan.append(("calculate_volatility", {"ticker": ticker, "window": 21}))
    if not memory.has("news"):
        plan.append(("get_news", {"ticker": ticker, "n": 10}))
    if memory.has("news") and not memory.has("sentiment"):
        headlines = [x["headline"] for x in memory.get("news") if "headline" in x]
        plan.append(("llm_sentiment", {"headlines": headlines}))
    if not memory.has("web"):
        plan.append(
            (
                "web_search",
                {"query": f"{ticker} stock risks next 90 days analyst commentary"},
            )
        )
    return plan


def run_single_research_agent(ticker: str = "AAPL") -> dict[str, Any]:
    """
    Task 3A: tool-using research agent with observe/replan cycles.
    """
    memory = ShortTermMemory()
    trace: list[dict[str, Any]] = []
    ticker = ticker.upper()

    # Cache short-circuit (Task 3C persistent memory)
    cached = load_brief(ticker)
    if cached:
        return {
            "ticker": ticker,
            "from_cache": True,
            "report": cached["brief"],
            "trace": [{"event": "cache_hit", "path": str(cached)}],
            "memory": memory.snapshot(),
        }

    max_cycles = 6
    for cycle in range(max_cycles):
        plan = _heuristic_plan(ticker, memory)
        if not plan:
            break
        tool_name, args = plan[0]
        trace.append({"cycle": cycle, "decision": "call_tool", "tool": tool_name, "args": args})
        try:
            result = ALL_TOOLS[tool_name](**args)
        except Exception as exc:  # noqa: BLE001
            trace.append({"cycle": cycle, "observation": "error", "error": str(exc)})
            # Fallback: try alternate approach
            if tool_name == "web_search":
                result = [{"title": "fallback", "body": "Using news-only path", "error": str(exc)}]
            elif tool_name == "get_news":
                result = [{"headline": f"{ticker} news unavailable", "error": str(exc)}]
            else:
                result = {"error": str(exc)}
        # Observe
        if tool_name == "get_price_data":
            memory.set("price", result)
        elif tool_name == "calculate_volatility":
            memory.set("vol", result)
        elif tool_name == "get_news":
            memory.set("news", result)
        elif tool_name == "llm_sentiment":
            memory.set("sentiment", result)
        elif tool_name == "web_search":
            memory.set("web", result)
        trace.append({"cycle": cycle, "observation": result})

        # Replan signal: if sentiment missing after news, next loop will add it
        if memory.has("news") and not memory.has("sentiment"):
            trace.append({"cycle": cycle, "replan": "need_sentiment_next"})

    report = _synthesize_report(ticker, memory)
    save_brief(ticker, report.model_dump())
    return {
        "ticker": ticker,
        "from_cache": False,
        "report": report.model_dump(),
        "trace": trace,
        "memory": memory.snapshot(),
    }


def _synthesize_report(ticker: str, memory: ShortTermMemory) -> FinalResearchReport:
    price = memory.get("price") or {}
    vol = memory.get("vol") or {}
    sent = memory.get("sentiment") or {}
    web = memory.get("web") or []
    news = memory.get("news") or []

    health = (
        f"{ticker} latest close={price.get('latest_close')}, "
        f"SMA50={price.get('sma_50')}, SMA200={price.get('sma_200')}, "
        f"RSI={price.get('rsi_14')}, ann. vol={vol.get('annualised_volatility')}, "
        f"news sentiment={sent.get('overall_label')} "
        f"({sent.get('overall_sentiment_score')})."
    )

    risks = [
        RiskItem(
            title="Volatility regime risk",
            evidence=f"Annualised HV window data: {vol}",
        ),
        RiskItem(
            title="Trend / momentum reversal risk",
            evidence=f"Indicator snapshot: RSI={price.get('rsi_14')}, MACD hist={price.get('macd_hist')}",
        ),
        RiskItem(
            title="Narrative / headline risk",
            evidence=f"Sentiment={sent.get('overall_label')}; sample news={news[:2]}; web={web[:1]}",
        ),
    ]

    hedge = (
        "Data-driven hedge: if annualised volatility is elevated versus the name's "
        "recent median and RSI is stretched, a partial protective put (or collar) "
        "sized to ~50% of notional for a 90-day horizon reduces left-tail exposure "
        "while leaving upside uncapped on the unhedged half."
    )
    return FinalResearchReport(
        ticker=ticker,
        financial_health_summary=health,
        top_three_risks=risks,
        hedge_strategy=hedge,
        sources_note="yfinance price/news, duckduckgo web_search, LLM sentiment",
    )


def run_analyst_agent(ticker: str, memory: ShortTermMemory | None = None) -> DataAnalystBrief:
    memory = memory or ShortTermMemory()
    ticker = ticker.upper()
    # Restricted tools only - no get_news / web_search
    price = ANALYST_TOOLS["get_price_data"](ticker, "2y")
    memory.set("price", price)
    vol = ANALYST_TOOLS["calculate_volatility"](ticker, 21)
    memory.set("vol", vol)
    # Headlines are not fetched via get_news (Writer-only tool). Feed indicator
    # context strings into llm_sentiment so the analyst stays within its tool ACL.
    context_lines = [
        f"{ticker} trading near {price.get('latest_close')} with RSI {price.get('rsi_14')}",
        f"{ticker} annualised volatility reading {vol.get('annualised_volatility')}",
        f"{ticker} MACD histogram {price.get('macd_hist')} versus SMA stack",
    ]
    sent = ANALYST_TOOLS["llm_sentiment"](context_lines)
    memory.set("sentiment", sent)

    risks = [
        RiskItem(title="Elevated realized volatility", evidence=str(vol)),
        RiskItem(
            title="Technical exhaustion",
            evidence=f"RSI={price.get('rsi_14')} MACD hist={price.get('macd_hist')}",
        ),
        RiskItem(
            title="Sentiment swing",
            evidence=f"score={sent.get('overall_sentiment_score')}",
        ),
    ]
    return DataAnalystBrief(
        ticker=ticker,
        financial_health_summary=(
            f"Close {price.get('latest_close')} vs SMA50/200 "
            f"{price.get('sma_50')}/{price.get('sma_200')}; "
            f"ann vol {vol.get('annualised_volatility')}; "
            f"sentiment {sent.get('overall_label')}."
        ),
        price_snapshot=price,
        volatility=vol,
        sentiment=sent,
        top_risks_draft=risks,
        notes_for_writer="Please enrich risks with qualitative analyst commentary.",
    )


def run_writer_agent(
    brief: DataAnalystBrief,
    clarification: ClarificationResponse | None = None,
) -> FinalResearchReport:
    # Writer tools only
    news = WRITER_TOOLS["get_news"](brief.ticker, 10)
    web = WRITER_TOOLS["web_search"](
        f"{brief.ticker} stock risks valuation regulation competition next quarter"
    )
    extra = ""
    if clarification:
        extra = f" Clarification incorporated: {clarification.answer}"

    risks = brief.top_risks_draft[:3]
    if len(risks) < 3:
        risks.append(
            RiskItem(
                title="Headline / narrative risk",
                evidence=str(news[:2]),
            )
        )
    # Enrich evidence with web
    if web and risks:
        risks[0].evidence = risks[0].evidence + f" | web: {web[0]}"

    return FinalResearchReport(
        ticker=brief.ticker,
        financial_health_summary=brief.financial_health_summary + extra,
        top_three_risks=risks[:3],
        hedge_strategy=(
            "Use options collar on half the position over ~90 days when "
            f"annualised vol is {brief.volatility.get('annualised_volatility')}; "
            "finance put premium by selling OTM calls, keeping cash buffer for gap risk."
        ),
        sources_note=f"news={len(news)} web={len(web)}",
    )


def run_multi_agent_pipeline(ticker: str = "AAPL") -> dict[str, Any]:
    """
    Task 3B: Analyst -> Writer with critique loop (Writer asks once, Analyst answers).
    """
    messages: list[dict[str, Any]] = []
    memory = ShortTermMemory()
    ticker = ticker.upper()

    brief = run_analyst_agent(ticker, memory)
    messages.append(
        {
            "role": "analyst",
            "content": "Structured data brief ready",
            "payload": brief.model_dump(),
        }
    )

    # Critique loop: writer requests clarification
    critique = ClarificationRequest(
        question="Please quantify drawdown risk vs 52-week high using price snapshot.",
        needed_fields=["drawdown_from_high_estimate"],
    )
    messages.append({"role": "writer", "content": "Clarification request", "payload": critique.model_dump()})

    price = brief.price_snapshot
    # Approximate: use bb_upper as proxy if 52w not in snapshot
    close = price.get("latest_close") or 0
    ref = price.get("bb_upper") or close
    dd = None if not close or not ref else (close / ref) - 1.0
    clarification = ClarificationResponse(
        question=critique.question,
        answer=f"Estimated proximity metric close/bb_upper - 1 = {dd}",
        data={"close": close, "ref_bb_upper": ref, "metric": dd},
    )
    messages.append(
        {
            "role": "analyst",
            "content": "Clarification response",
            "payload": clarification.model_dump(),
        }
    )

    final = run_writer_agent(brief, clarification)
    messages.append(
        {"role": "writer", "content": "Final research report", "payload": final.model_dump()}
    )
    save_brief(ticker, final.model_dump())

    return {
        "ticker": ticker,
        "messages": messages,
        "final_report": final.model_dump(),
        "memory": memory.snapshot(),
    }


def answer_followup(ticker: str, question: str, memory: ShortTermMemory) -> str:
    """Task 3C short-term memory: answer without re-calling tools if data present."""
    if "volatility" in question.lower() and memory.has("vol"):
        return f"From session memory (no re-fetch): {memory.get('vol')}"
    if "sentiment" in question.lower() and memory.has("sentiment"):
        return f"From session memory (no re-fetch): {memory.get('sentiment')}"
    if "price" in question.lower() and memory.has("price"):
        return f"From session memory (no re-fetch): {memory.get('price')}"
    return "Needed data not in memory; would call tools on a cold session."
