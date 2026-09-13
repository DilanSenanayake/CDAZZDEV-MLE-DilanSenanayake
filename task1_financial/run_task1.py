"""Generate Task 1 report artifacts and print a JSON summary for notebook embedding."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data_pipeline import run_pipeline
from llm_reasoning import run_llm_reasoning
from report import render_html_report

logging.basicConfig(level=logging.INFO)


def main(ticker: str = "AAPL") -> None:
    pipe = run_pipeline(ticker)
    llm = run_llm_reasoning(ticker, pipe["summary"], pipe["headlines"])
    paths = render_html_report(
        ticker=ticker,
        summary=pipe["summary"],
        sentiment=llm["sentiment"],
        signal=llm["signal"],
        headlines=pipe["headlines"],
        featured=pipe["featured"],
        output_dir=ROOT / "reports",
    )
    payload = {
        "ticker": ticker,
        "summary": pipe["summary"],
        "headline_count": len(pipe["headlines"]),
        "headlines": pipe["headlines"],
        "llm": llm,
        "report_paths": {k: str(v) for k, v in paths.items()},
        "ohlcv_rows": int(len(pipe["featured"])),
    }
    out = ROOT / "reports" / f"{ticker}_run_summary.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"ok": True, "summary_path": str(out), **{k: str(v) for k, v in paths.items()}}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "AAPL")
