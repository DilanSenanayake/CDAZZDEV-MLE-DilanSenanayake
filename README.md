# CDAZZDEV Senior MLE Assessment

**Candidate repo name:** `CDAZZDEV-MLE-DilanSenanayake`  
**Domains:** Financial AI | Generative AI | Agentic Workflows

Public repository containing all three tasks, notebooks with visible outputs, citations, and reflection.

## Tasks

| Task | Folder | Notebook | Colab |
|------|--------|----------|-------|
| 1 Financial AI | [`task1_financial/`](task1_financial/) | [`equity_research.ipynb`](task1_financial/equity_research.ipynb) | [Open](https://colab.research.google.com/github/DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake/blob/master/task1_financial/equity_research.ipynb) |
| 2 Generative AI | [`task2_genai/`](task2_genai/) | [`finetune.ipynb`](task2_genai/finetune.ipynb) | [Open (T4 GPU)](https://colab.research.google.com/github/DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake/blob/master/task2_genai/finetune.ipynb) |
| 3 Agentic Workflows | [`task3_agentic/`](task3_agentic/) | [`multi_agent.ipynb`](task3_agentic/multi_agent.ipynb) | [Open](https://colab.research.google.com/github/DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake/blob/master/task3_agentic/multi_agent.ipynb) |

Full Colab path list: [`docs/COLAB.md`](docs/COLAB.md)

## Setup

```bash
git clone <this-repo>
cd CDAZZDEV-MLE-Dilan
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # add GROQ_API_KEY, HF_TOKEN, WANDB_API_KEY
```

## Run demos locally

```bash
python task1_financial/run_task1.py AAPL
python task3_agentic/run_task3.py AAPL
python task2_genai/generate_dataset.py
python task2_genai/eval_offline.py
streamlit run task3_agentic/dashboard.py
```

Task 2 QLoRA training: open in **Google Colab (T4 GPU)**  
? https://colab.research.google.com/github/DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake/blob/master/task2_genai/finetune.ipynb

See [`docs/COLAB.md`](docs/COLAB.md) for all notebook Colab links.

## Submission links (fill before email)

| Deliverable | Link |
|-------------|------|
| GitHub (public) | https://github.com/DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake |
| Hugging Face model (Task 2) | https://huggingface.co/DilanSenanayake/phi3-compliance-qlora-merged |
| Task 2 Colab (GitHub) | https://colab.research.google.com/github/DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake/blob/master/task2_genai/finetune.ipynb |
| Task 2 Colab (Drive, executed) | https://colab.research.google.com/drive/1FTUT9PINjueY3RslS-p3-IdZK56ZRjqN?usp=sharing |
| Google Drive (large weights, optional) | _if needed_ |
| Video walkthrough (<=5 min, mandatory per email) | _YouTube unlisted or Drive_ |

## Mandatory docs

- [`CITATIONS.md`](CITATIONS.md)
- [`REFLECTION.md`](REFLECTION.md) (<=600 words)
- Planning docs: [`docs/ASSESSMENT_PLAN.md`](docs/ASSESSMENT_PLAN.md), [`docs/STEPS.md`](docs/STEPS.md), [`docs/CHANGELOG.md`](docs/CHANGELOG.md), [`docs/COLAB.md`](docs/COLAB.md)

## Video walkthrough checklist

Record yourself narrating (<=5 minutes):

1. Task 1 pipeline + HTML brief  
2. Task 3 agent trace + critique loop  
3. Task 2 use case, dataset diversity, fine-tune notebook hyperparameters + eval table  

See [`docs/VIDEO_SCRIPT.md`](docs/VIDEO_SCRIPT.md).

## Security

Never commit API keys. Use `.env` (gitignored) or Colab secrets.
