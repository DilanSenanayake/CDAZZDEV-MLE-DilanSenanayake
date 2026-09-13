"""Simple Streamlit dashboard over agent_trace.jsonl (Task 3 bonus)."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

LOG = Path(__file__).resolve().parent / "logs" / "agent_trace.jsonl"

st.set_page_config(page_title="Agent Trace", layout="wide")
st.title("Task 3 Agent Trace Dashboard")

if not LOG.exists():
    st.warning(f"No log at {LOG}. Run `python task3_agentic/run_task3.py` first.")
else:
    rows = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    st.metric("Tool calls", len(rows))
    st.dataframe(rows, use_container_width=True)
    tools = {}
    for r in rows:
        tools[r["tool"]] = tools.get(r["tool"], 0) + 1
    st.bar_chart({"count": tools})
