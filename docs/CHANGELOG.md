# Changelog

All notable planning decisions, documentation updates, and implementation changes for the CDAZZDEV Senior MLE Assessment are recorded here.

Format: date (local) - category - summary - detail.

Categories: `plan` - `docs` - `task1` - `task2` - `task3` - `infra` - `submit` - `decision`

---

## 2026-09-13 (implementation)

### infra - Repository scaffold completed

- Initialized git repo; added `.gitignore`, `.env.example`, `requirements.txt`.
- Created `task1_financial/`, `task2_genai/`, `task3_agentic/` layout matching assessment naming.

### task1 - Equity research pipeline implemented

- `data_pipeline.py`: 2y OHLCV, SMA/RSI/MACD/BB from first principles, news, summary dict.
- `llm_reasoning.py` + Pydantic schemas + separated prompts; offline fallback without Groq key.
- Bonus HTML/Markdown brief with matplotlib chart under `task1_financial/reports/`.
- Notebook `equity_research.ipynb` with visible outputs.

### task3 - Multi-agent system implemented

- Five traced tools; observation-driven single agent; analyst/writer critique loop.
- Short-term + persistent memory; `logs/agent_trace.jsonl`; Streamlit dashboard.
- Notebook `multi_agent.ipynb` with visible outputs.

### task2 - Fine-tuning pipeline artifacts

- Use case + teacher prompt; 120-example JSONL with 80/10/10 split and diversity report.
- Colab QLoRA notebook with justified hyperparameters.
- Offline eval: ROUGE-L table, secondary metric, hallucination labels, qualitative analysis.

### submit - Mandatory docs

- Root `README.md`, `CITATIONS.md`, `REFLECTION.md` (<=600 words), `docs/VIDEO_SCRIPT.md`.
- Remaining candidate actions: add Groq/HF keys, run Colab QLoRA + HF push, record video, publish GitHub.

---

## 2026-09-13

### plan / decision - Initial assessment planning session

- Reviewed invitation email and `CDAZZDEV_Senior_MLE_Assessment_2026.pdf`.
- Locked strategy: **complete all three tasks**; fallback priority **Task 1 ? Task 3 ? Task 2**.
- Locked tooling preferences: Groq + OpenRouter backup; LangGraph; QLoRA on Colab; W&B for Task 2.
- Preferred Task 2 domain: **financial compliance / risk-disclosure Q&A** (domain-specific; avoids generic chatbot penalty).
- Preferred agent framework: **LangGraph** for autonomous tool selection and visible traces.
- Default tickers under consideration: `AAPL` / `MSFT` (final pick TBD).

### decision - Email overrides PDF on video

- PDF marks video walkthrough as optional (+5 bonus).
- Email states video is **mandatory** and submissions without it will not be reviewed.
- **Resolution:** treat video as mandatory; record ?5 minutes, personally narrated.

### docs - Created planning documentation set

| File | Purpose |
|------|---------|
| `docs/ASSESSMENT_PLAN.md` | Master plan: rubrics, structure, schedule, checklist |
| `docs/STEPS.md` | Ordered execution steps with status tracker |
| `docs/CHANGELOG.md` | This file - dated record of steps and changes |

### infra - Workspace state

- Workspace previously contained only the assessment PDF.
- No implementation code yet (`task1_financial/`, `task2_genai/`, `task3_agentic/` not created).
- GitHub public repo name and candidate display name still TBD.

---

## Template for future entries

```markdown
## YYYY-MM-DD

### category - Short title

- What changed
- Why it changed
- Rubric / submission impact (if any)
- Related files / PRs
```

### Examples to add as work proceeds

- `task1` - Implemented RSI/MACD from first principles; validated against known values  
- `task2` - Reduced batch size after Colab OOM; documented in notebook  
- `task3` - Switched from fixed tool sequence to LangGraph ReAct for autonomous selection  
- `submit` - Pushed merged model to Hugging Face; link recorded in README  
- `decision` - Chose ticker `MSFT` for Tasks 1 and 3  

---

## Change index (quick scan)

| Date | Category | Change |
|------|----------|--------|
| 2026-09-13 | plan / decision | All-three strategy; tooling; Task 2 domain preference |
| 2026-09-13 | decision | Video mandatory per email |
| 2026-09-13 | docs | Added ASSESSMENT_PLAN, STEPS, CHANGELOG |
| 2026-09-13 | infra | Docs-only workspace; no code yet |
