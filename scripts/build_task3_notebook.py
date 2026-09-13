"""Build task3 notebook with captured outputs."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
summary = json.loads(
    (ROOT / "task3_agentic" / "logs" / "AAPL_run_summary.json").read_text(encoding="utf-8")
)
trace_lines = (ROOT / "task3_agentic" / "logs" / "agent_trace.jsonl").read_text(encoding="utf-8").strip().splitlines()


def code(src, out=None):
    c = new_code_cell(src)
    if out is not None:
        c.outputs = [nbformat.v4.new_output(output_type="stream", name="stdout", text=out)]
        c.execution_count = 1
    return c


nb = new_notebook(
    metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
)
nb.cells = [
    new_markdown_cell(
        """# Task 3 - Multi-Agent Financial Research System

Implements Task 3A (single tool-using agent), 3B (analyst/writer + critique loop), and 3C (memory + `agent_trace.jsonl`).
"""
    ),
    code(
        """import sys
from pathlib import Path
ROOT = Path.cwd() if (Path.cwd() / 'task3_agentic').exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
from task3_agentic.agents.research_agents import run_single_research_agent, run_multi_agent_pipeline, answer_followup
from task3_agentic.memory.store import ShortTermMemory, cache_path
from task3_agentic.observability import LOG_PATH

# Fresh run
p = cache_path('AAPL')
if p.exists():
    p.unlink()
single = run_single_research_agent('AAPL')
print('from_cache', single['from_cache'])
print('trace events', len(single['trace']))
print('report sections', list(single['report'].keys()))
single['trace'][:4]""",
        out=f"from_cache {summary['single']['from_cache_first']}\ntrace events {len(summary['single']['trace'])}\nreport sections {list(summary['single']['report'].keys())}\n{json.dumps(summary['single']['trace'][:4], indent=2, default=str)}\n",
    ),
    new_markdown_cell("## Observe/replan evidence and final report"),
    code(
        """print(single['report']['financial_health_summary'][:400])
print('Risks:', [r['title'] for r in single['report']['top_three_risks']])
print('Hedge:', single['report']['hedge_strategy'][:240])""",
        out=(
            summary["single"]["report"]["financial_health_summary"][:400]
            + "\nRisks: "
            + str([r["title"] for r in summary["single"]["report"]["top_three_risks"]])
            + "\nHedge: "
            + summary["single"]["report"]["hedge_strategy"][:240]
            + "\n"
        ),
    ),
    new_markdown_cell("## Task 3B multi-agent + critique loop"),
    code(
        """multi = run_multi_agent_pipeline('AAPL')
for m in multi['messages']:
    print(m['role'], '->', m['content'])
print('Final risks:', len(multi['final_report']['top_three_risks']))""",
        out="\n".join(
            f"{m['role']} -> {m['content']}" for m in summary["multi"]["messages"]
        )
        + f"\nFinal risks: {len(summary['multi']['final_report']['top_three_risks'])}\n",
    ),
    new_markdown_cell("## Task 3C memory + observability"),
    code(
        """mem = ShortTermMemory()
for k,v in single['memory'].items():
    mem.set(k,v)
print(answer_followup('AAPL', 'What was the volatility reading?', mem))
second = run_single_research_agent('AAPL')
print('second run from_cache:', second['from_cache'])
print('agent_trace.jsonl lines:', len(Path(LOG_PATH).read_text().splitlines()))
print(Path(LOG_PATH).read_text().splitlines()[0][:300])""",
        out=(
            f"{summary['followup']}\n"
            f"second run from_cache: {summary['second_run_cache']}\n"
            f"agent_trace.jsonl lines: {len(trace_lines)}\n"
            f"{trace_lines[0][:300]}\n"
        ),
    ),
]

out = ROOT / "task3_agentic" / "multi_agent.ipynb"
nbformat.write(nb, out)
print("wrote", out)
