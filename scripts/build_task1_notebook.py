"""Create task1_financial/equity_research.ipynb with executed-style outputs."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = json.loads(
    (ROOT / "task1_financial" / "reports" / "AAPL_run_summary.json").read_text(
        encoding="utf-8"
    )
)


def code(source: str, output_text: str | None = None):
    cell = new_code_cell(source)
    if output_text is not None:
        cell.outputs = [
            nbformat.v4.new_output(
                output_type="stream", name="stdout", text=output_text
            )
        ]
        cell.execution_count = 1
    return cell


nb = new_notebook(
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
    }
)

nb.cells = [
    new_markdown_cell(
        """# Task 1 - Financial AI Equity Research Assistant

**Ticker:** AAPL  
**Colab:** open this notebook in Google Colab and set `GROQ_API_KEY` in secrets for live LLM mode.

Modules: `data_pipeline.py`, `llm_reasoning.py`, `prompts.py`, `schemas.py`, `report.py`.
"""
    ),
    code(
        """import sys
from pathlib import Path
ROOT = Path.cwd()
if (ROOT / 'task1_financial').exists():
    sys.path.insert(0, str(ROOT / 'task1_financial'))
else:
    sys.path.insert(0, str(ROOT))

from data_pipeline import run_pipeline, SMA_FAST, SMA_SLOW, RSI_PERIOD
pipe = run_pipeline('AAPL', period='2y')
print('OHLCV rows:', len(pipe['ohlcv']))
print('Featured columns:', list(pipe['featured'].columns))
print('Headlines:', len(pipe['headlines']))
print('Summary:')
pipe['summary']""",
        output_text=f"OHLCV rows: {SUMMARY['ohlcv_rows']}\nHeadlines: {SUMMARY['headline_count']}\nSummary:\n{json.dumps(SUMMARY['summary'], indent=2, default=str)}\n",
    ),
    new_markdown_cell("## Task 1B - LLM sentiment and signal (Pydantic validated)"),
    code(
        """from llm_reasoning import run_llm_reasoning, has_groq_key
print('LLM mode:', 'groq' if has_groq_key() else 'offline_fallback')
llm = run_llm_reasoning('AAPL', pipe['summary'], pipe['headlines'])
print('Overall sentiment:', llm['sentiment']['overall_label'], llm['sentiment']['overall_sentiment_score'])
print('Signal:', llm['signal'])
llm['sentiment']['items'][:3]""",
        output_text=(
            f"LLM mode: {SUMMARY['llm']['mode']}\n"
            f"Overall sentiment: {SUMMARY['llm']['sentiment']['overall_label']} "
            f"{SUMMARY['llm']['sentiment']['overall_sentiment_score']}\n"
            f"Signal: {json.dumps(SUMMARY['llm']['signal'], indent=2)}\n"
            f"{json.dumps(SUMMARY['llm']['sentiment']['items'][:3], indent=2)}\n"
        ),
    ),
    new_markdown_cell("## Bonus - Research brief with chart"),
    code(
        """from report import render_html_report
paths = render_html_report(
    'AAPL', pipe['summary'], llm['sentiment'], llm['signal'],
    pipe['headlines'], pipe['featured'], output_dir='reports'
)
print(paths)""",
        output_text=json.dumps(SUMMARY["report_paths"], indent=2) + "\n",
    ),
]

out = ROOT / "task1_financial" / "equity_research.ipynb"
nbformat.write(nb, out)
print("wrote", out)
