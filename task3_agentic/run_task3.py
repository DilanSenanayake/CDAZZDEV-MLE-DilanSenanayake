"""Run Task 3 end-to-end demos and write artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from task3_agentic.agents.research_agents import (
    answer_followup,
    run_multi_agent_pipeline,
    run_single_research_agent,
)
from task3_agentic.memory.store import ShortTermMemory, load_brief
from task3_agentic.observability import LOG_PATH


def main(ticker: str = "AAPL") -> None:
    # Clear prior cache for a fresh first run demo
    from task3_agentic.memory.store import cache_path

    cp = cache_path(ticker)
    if cp.exists():
        cp.unlink()
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    print("=== Task 3A single agent ===")
    single = run_single_research_agent(ticker)
    print(json.dumps({"from_cache": single["from_cache"], "report_keys": list(single["report"].keys()), "trace_len": len(single["trace"])}, indent=2))

    print("=== Task 3C follow-up from short-term memory ===")
    mem = ShortTermMemory()
    for k, v in (single.get("memory") or {}).items():
        mem.set(k, v)
    follow = answer_followup(ticker, "What was the volatility?", mem)
    print(follow)

    print("=== Task 3C second run should hit cache ===")
    cached_run = run_single_research_agent(ticker)
    print("from_cache:", cached_run["from_cache"])

    print("=== Task 3B multi-agent ===")
    # Use a different cache key path by saving final after multi
    multi = run_multi_agent_pipeline(ticker)
    print("messages:", len(multi["messages"]))
    print("final risks:", len(multi["final_report"]["top_three_risks"]))

    out = ROOT / "task3_agentic" / "logs" / f"{ticker}_run_summary.json"
    out.write_text(
        json.dumps(
            {
                "single": {
                    "from_cache_first": single["from_cache"],
                    "report": single["report"],
                    "trace": single["trace"],
                },
                "followup": follow,
                "second_run_cache": cached_run["from_cache"],
                "multi": multi,
                "trace_log": str(LOG_PATH),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print("wrote", out)
    print("agent_trace exists:", LOG_PATH.exists(), "lines:", len(LOG_PATH.read_text(encoding='utf-8').splitlines()) if LOG_PATH.exists() else 0)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "AAPL")
