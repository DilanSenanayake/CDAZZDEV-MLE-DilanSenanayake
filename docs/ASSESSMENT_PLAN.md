# CDAZZDEV Senior MLE Assessment - Master Plan

**Role:** Senior Machine Learning Engineer  
**Company:** Ceylon Dazzling Dev Holding (Pvt.) Ltd. (CDAZZDEV)  
**Document source:** `CDAZZDEV_Senior_MLE_Assessment_2026.pdf` + invitation email  
**Deadline:** Within 2 working days of receiving the assessment email  
**Cost policy:** Free-tier tools only - no personal expenditure required  

---

## 1. Assessment overview

| Task | Domain | Core deliverable | Marks |
|------|--------|------------------|-------|
| Task 1 | Financial AI | LLM-powered equity research pipeline with structured outputs | 100 |
| Task 2 | Generative AI | Domain-specific fine-tuning pipeline with rigorous evaluation | 100 |
| Task 3 | Agentic Workflows | Multi-agent financial research system with memory and observability | 100 |

- Minimum: **at least one** completed task  
- Preferred: **all three** (priority consideration)  
- AI tools: fully permitted if cited in `CITATIONS.md`  
- Follow-up interview: must defend architecture and code  

---

## 2. Strategy decisions (locked)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scope | Complete **all three** tasks | Priority for candidates who submit all domains |
| If time slips | Ship Task 1 + Task 3 polished; Task 2 with core eval (skip RAG bonus) | Highest score density first |
| Video | Treat as **mandatory** (email), not optional (PDF) | Email is the submission channel - missing video = not reviewed |
| Task 1 ticker | TBD (default candidates: `AAPL` or `MSFT`) | Liquid, strong news coverage via yfinance |
| Task 2 use case | Financial compliance / risk-disclosure Q&A (preferred) | Domain-specific; aligns with Financial AI theme; avoids generic-chatbot cap (max 5/30) |
| Task 2 student model | Phi-3-mini or Mistral-7B | Fits Colab free T4 + QLoRA |
| Task 2 teacher model | Groq Llama-3-70B (or OpenRouter free GPT-4o) | Must **?** student model |
| Agent framework | LangGraph | Clearer observe/replan traces for rubric vs fixed CrewAI crews |
| LLM inference | Groq primary; OpenRouter backup | Free tier |
| Experiment tracking | Weights & Biases free tier | Loss curves for Task 2 |

---

## 3. Email vs PDF - conflict log

| Topic | PDF | Email | Resolution |
|-------|-----|-------|------------|
| Video walkthrough | Optional (+5 bonus per task) | **Mandatory** - submissions without video not considered | **Follow email:** record ?5 min personal narrated walkthrough |
| HF model access | Public preferred; private OK with reviewer access | Same + Drive alternative | Prefer public HF; Drive fallback for large merges |

---

## 4. Target repository structure

```
CDAZZDEV-MLE-[YourName]/
??? README.md
??? CITATIONS.md
??? REFLECTION.md                    # ?600 words, all tasks
??? .env.example
??? .gitignore
??? requirements.txt
??? docs/
?   ??? ASSESSMENT_PLAN.md           # this file
?   ??? STEPS.md                     # execution checklist
?   ??? CHANGELOG.md                 # dated changes
??? task1_financial/
?   ??? README.md
?   ??? equity_research.ipynb        # keep cell outputs
?   ??? data_pipeline.py
?   ??? llm_reasoning.py
?   ??? prompts.py
?   ??? schemas.py
?   ??? reports/
??? task2_genai/
?   ??? README.md
?   ??? finetune.ipynb
?   ??? data/                        # train/val/test JSONL
?   ??? prompts/                     # teacher system prompt
?   ??? eval/
??? task3_agentic/
    ??? README.md
    ??? multi_agent.ipynb
    ??? tools/
    ??? agents/
    ??? memory/
    ??? logs/
        ??? agent_trace.jsonl
```

**Repo naming rule:** `CDAZZDEV-MLE-[YourName]` - must be **public**.

---

## 5. Shared free tooling stack

| Purpose | Platform / library |
|---------|-------------------|
| GPU | Google Colab (free T4 / L4) |
| LLM API | Groq, OpenRouter |
| Model hosting | Hugging Face Hub |
| Fine-tuning | PEFT, TRL, BitsAndBytes, Transformers |
| Financial data | yfinance (Alpha Vantage free tier optional) |
| Vector DB | ChromaDB (local) - Task 2 bonus |
| Agents | LangChain / LangGraph |
| Tracking | wandb |
| Web search | duckduckgo-search |
| VCS | GitHub public repo |

**Secrets:** Colab secrets / environment variables only. Never commit API keys or HF tokens.

---

## 6. Task 1 - Financial AI (100 pts)

### 6.1 Task 1A - Data pipeline (60 pts)

| Criterion | Marks | Requirement |
|-----------|-------|-------------|
| OHLCV fetch | 10 | ?2 years, correct ticker, **no hardcoded date strings** |
| Indicator accuracy | 25 | SMA50, SMA200, RSI14, MACD(12,26,9), BB(20,2) - **no TA-Lib** |
| News retrieval | 10 | ?10 headlines from free source |
| Summary dictionary | 10 | price, 52w H/L, PE, YTD return, momentum signal |
| Robustness | 5 | Missing data handled; readable; commented |

### 6.2 Task 1B - LLM sentiment & signal (40 pts)

| Criterion | Marks | Requirement |
|-----------|-------|-------------|
| Per-headline JSON | 10 | headline, sentiment, confidence, brief_reason + aggregate score |
| Signal reasoning | 15 | Buy/Hold/Sell from **indicator combinations**, not echo |
| Structured validation | 10 | Pydantic/JSON schema; failures logged & handled |
| Prompt engineering | 5 | Prompts as constants/templates - not inline |

### 6.3 Bonus (up to +5)

One-page equity brief (MD ? HTML or PDF): company snapshot, technical outlook, news sentiment (top 3 headlines), LLM recommendation, **mandatory risk disclaimer**, ?1 matplotlib chart.

### 6.4 Threshold

- ?70: production-level  
- 50-69: adequate  
- <50: below standard  

---

## 7. Task 2 - Generative AI fine-tuning (100 pts)

### 7.1 Task 2A - Use case & dataset (30 pts)

| Criterion | Marks | Requirement |
|-----------|-------|-------------|
| Use case quality | 10 | Domain-specific I/O + success criteria (not generic chatbot) |
| Dataset size | 5 | ?100 examples; teacher prompt included |
| Diversity | 10 | Length distribution + keyword/topic analysis |
| Format & split | 5 | Chat JSONL; 80/10/10 with sizes stated |

### 7.2 Task 2B - QLoRA execution (40 pts)

| Criterion | Marks | Requirement |
|-----------|-------|-------------|
| QLoRA | 10 | 4-bit NF4 + PEFT |
| Hyperparameter justification | 15 | Document r, alpha, targets, LR, scheduler, epochs, batch, grad accum, max seq - **no unexplained defaults** |
| Loss monitoring | 10 | Train + val loss per epoch; val loss decreases |
| Model saved | 5 | `merge_and_unload()` + HF or Drive link |

### 7.3 Task 2C - Evaluation (30 pts)

| Criterion | Marks | Requirement |
|-----------|-------|-------------|
| ROUGE-L | 8 | Base vs fine-tuned table on same test set |
| Additional metric | 7 | BERTScore F1 **or** LLM-as-judge (structured JSON) |
| Hallucination rate | 7 | ?10 manual labels (correct / partial / hallucinated) |
| Qualitative analysis | 8 | Two evidence-backed paragraphs + next steps |

### 7.4 Bonus (up to +5)

RAG fallback via ChromaDB when confidence (perplexity / self-rating) is low - before/after example in notebook.

### 7.5 Common mark losses

- <30 training examples claimed as fine-tuned  
- Same model as teacher and student  
- Cleared notebook outputs  
- Hardcoded API keys / tokens  

---

## 8. Task 3 - Agentic workflows (100 pts)

### 8.1 Task 3A - Tool-using research agent (50 pts)

**Query:** Analyse financial health and market sentiment of `[TICKER]`. Identify top three 90-day share-price risks and suggest one data-driven hedge.

**Tools (all five required):**

1. `get_price_data(ticker, period)`  
2. `get_news(ticker, n)`  
3. `calculate_volatility(ticker, window)`  
4. `llm_sentiment(headlines)`  
5. `web_search(query)`  

| Criterion | Marks |
|-----------|-------|
| All five tools | 15 |
| Autonomous tool selection | 10 |
| Observe & replan cycle | 8 |
| Final report quality (3 sections) | 10 |
| Error handling | 7 |

### 8.2 Task 3B - Multi-agent coordination (35 pts)

| Agent | Role | Tools | Output |
|-------|------|-------|--------|
| A | Data Analyst | price, volatility, sentiment | Structured JSON brief |
| B | Research Writer | web_search, get_news | Final research report |

Must include: Pydantic handoff, visible message trace, **critique loop once**, end-to-end automation.

### 8.3 Task 3C - Memory & observability (15 pts)

| Criterion | Marks | Requirement |
|-----------|-------|-------------|
| Short-term memory | 5 | Follow-up from context without re-fetch |
| Persistent cache | 5 | JSON by ticker+date; second run loads cache |
| `agent_trace.jsonl` | 5 | tool, inputs, truncated output (200 chars), duration |

### 8.4 Bonus (up to +5)

LangSmith screenshot **or** Streamlit dashboard over `agent_trace.jsonl`.

---

## 9. Mandatory submission package

| Deliverable | Required | Notes |
|-------------|----------|-------|
| Public GitHub repo | Yes | Named `CDAZZDEV-MLE-[YourName]`; folders per task |
| Colab notebooks with **visible outputs** | Yes | One per task attempted |
| Hugging Face model link | Task 2 | Public preferred |
| Google Drive (large files) | If needed | Anyone with link can view |
| `agent_trace.jsonl` | Task 3 | Under `task3_agentic/logs/` |
| `CITATIONS.md` | Yes | AI + OSS in required format |
| `REFLECTION.md` | Yes | ?600 words |
| Video walkthrough ?5 min | **Yes (email)** | You narrate; YouTube unlisted or Drive |

### Citation format (required)

```text
# AI-ASSISTED: Claude (claude.sonnet-4), Prompt: '...', Date: YYYY-MM-DD
# SOURCE: Adapted from https://github.com/example/repo, file: trainer.py, Lines 45-82
```

Teacher-model data generation: include full system prompt in notebook/README appendix.

### Disqualifiers

- Hardcoded credentials in repo  
- Cleared notebook outputs  
- Design/slides instead of executable code  
- Private / inaccessible repo at review time  
- Plagiarism without citation  

---

## 10. Two-day schedule

### Day 1

| Block | Work |
|-------|------|
| Morning | Repo scaffold, secrets, Task 1A + 1B core |
| Afternoon | Task 1 bonus report; extract shared tools for Task 3 |
| Evening | Task 3A single agent + visible traces |

### Day 2

| Block | Work |
|-------|------|
| Morning | Task 3B critique loop + Task 3C memory / `agent_trace.jsonl` |
| Midday | Task 2A synthetic data + diversity metrics |
| Afternoon | Task 2B Colab QLoRA, merge, HF push |
| Evening | Task 2C eval; CITATIONS + REFLECTION; video; public-link verification |

---

## 11. Scoring focus (what to optimize)

- **Task 1:** Correct indicators + Pydantic validation + separated prompts > fancy UI  
- **Task 2:** Strong use case + diversity + ROUGE table + hallucination labels > extra epochs  
- **Task 3:** Visible autonomy + critique loop + `agent_trace.jsonl` > more agents  

---

## 12. Pre-send checklist

- [ ] GitHub public (verify in incognito)  
- [ ] Notebook outputs visible  
- [ ] No API keys / tokens in repo  
- [ ] HF / Drive links open without login (or reviewer access granted)  
- [ ] `CITATIONS.md` complete  
- [ ] `REFLECTION.md` ?600 words  
- [ ] `agent_trace.jsonl` present (if Task 3)  
- [ ] Video link works (anyone with link / unlisted YouTube)  
- [ ] Email reply includes: GitHub, HF (Task 2), Drive (if any), video  

---

## 13. Related documents

- [STEPS.md](./STEPS.md) - ordered execution steps and status  
- [CHANGELOG.md](./CHANGELOG.md) - dated record of plan/code changes  
