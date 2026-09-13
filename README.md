# CDAZZDEV Senior MLE Assessment

**Candidate:** Dilan Senanayake  
**Repository:** [CDAZZDEV-MLE-DilanSenanayake](https://github.com/DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake)  
**Domains:** Financial AI | Generative AI | Agentic Workflows

Public repository with all three tasks, notebooks (outputs preserved), `CITATIONS.md`, and `REFLECTION.md`.

## Submission links

| Deliverable | Link |
|-------------|------|
| GitHub (public) | https://github.com/DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake |
| Hugging Face model (Task 2) | https://huggingface.co/DilanSenanayake/phi3-compliance-qlora-merged |
| Video walkthrough (≤5 min) | https://drive.google.com/file/d/1v8D2KpuHk_Z6bSf4kBqESjMFVPcZ6wlv/view?usp=sharing |
| Task 2 Colab (executed) | https://colab.research.google.com/drive/1FTUT9PINjueY3RslS-p3-IdZK56ZRjqN?usp=sharing |

## Tasks

| Task | Folder | Notebook | Colab |
|------|--------|----------|-------|
| 1 Financial AI | [`task1_financial/`](task1_financial/) | [`equity_research.ipynb`](task1_financial/equity_research.ipynb) | [Open](https://colab.research.google.com/github/DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake/blob/master/task1_financial/equity_research.ipynb) |
| 2 Generative AI | [`task2_genai/`](task2_genai/) | [`finetune.ipynb`](task2_genai/finetune.ipynb) | [Open (T4)](https://colab.research.google.com/github/DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake/blob/master/task2_genai/finetune.ipynb) |
| 3 Agentic Workflows | [`task3_agentic/`](task3_agentic/) | [`multi_agent.ipynb`](task3_agentic/multi_agent.ipynb) | [Open](https://colab.research.google.com/github/DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake/blob/master/task3_agentic/multi_agent.ipynb) |

## Mandatory docs

- [`CITATIONS.md`](CITATIONS.md) — AI and open-source citations  
- [`REFLECTION.md`](REFLECTION.md) — architecture, limits, next steps (≤600 words)  
- Task 3 trace: [`task3_agentic/logs/agent_trace.jsonl`](task3_agentic/logs/agent_trace.jsonl)

## Setup

```bash
git clone https://github.com/DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake.git
cd CDAZZDEV-MLE-DilanSenanayake
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # add GROQ_API_KEY (optional for live LLM)
```

## Run locally

```bash
python task1_financial/run_task1.py AAPL
python task3_agentic/run_task3.py AAPL
python task2_genai/generate_dataset.py
python task2_genai/eval_offline.py
streamlit run task3_agentic/dashboard.py
```

Task 2 QLoRA training requires **Google Colab T4 GPU** (see Colab links above).

## Security

No API keys are committed. Use `.env` (gitignored) or Colab secrets.

---

## Email reply (copy/paste)

Subject: CDAZZDEV Senior MLE Assessment Submission — Dilan Senanayake

```text
Dear CDAZZDEV Hiring Team,

Please find my completed Senior MLE technical assessment submission below.
I have completed all three tasks (Financial AI, Generative AI, and Agentic Workflows).

1. GitHub Repository (public):
https://github.com/DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake

2. Hugging Face Model (Task 2):
https://huggingface.co/DilanSenanayake/phi3-compliance-qlora-merged

3. Video Walkthrough (≤5 minutes, Google Drive — anyone with the link):
https://drive.google.com/file/d/1v8D2KpuHk_Z6bSf4kBqESjMFVPcZ6wlv/view?usp=sharing

4. Task 2 Colab notebook with executed training outputs (optional mirror):
https://colab.research.google.com/drive/1FTUT9PINjueY3RslS-p3-IdZK56ZRjqN?usp=sharing

The repository includes notebooks with visible cell outputs, CITATIONS.md, REFLECTION.md,
and task3_agentic/logs/agent_trace.jsonl.

Thank you for your consideration.

Best regards,
Dilan Senanayake
```
