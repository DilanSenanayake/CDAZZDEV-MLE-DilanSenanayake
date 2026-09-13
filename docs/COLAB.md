# Google Colab notebooks

Repo: [DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake](https://github.com/DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake)

## Open-in-Colab links (from GitHub)

| Task | Notebook path in repo | Open in Colab |
|------|----------------------|---------------|
| Task 1 | `task1_financial/equity_research.ipynb` | [Open](https://colab.research.google.com/github/DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake/blob/master/task1_financial/equity_research.ipynb) |
| Task 2 | `task2_genai/finetune.ipynb` | [Open](https://colab.research.google.com/github/DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake/blob/master/task2_genai/finetune.ipynb) |
| Task 3 | `task3_agentic/multi_agent.ipynb` | [Open](https://colab.research.google.com/github/DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake/blob/master/task3_agentic/multi_agent.ipynb) |

## Direct Colab URLs (copy/paste)

```text
Task 1:
https://colab.research.google.com/github/DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake/blob/master/task1_financial/equity_research.ipynb

Task 2 (GPU / QLoRA — use T4):
https://colab.research.google.com/github/DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake/blob/master/task2_genai/finetune.ipynb

Task 3:
https://colab.research.google.com/github/DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake/blob/master/task3_agentic/multi_agent.ipynb
```

## Task 2 executed Colab (Drive copy with training outputs)

If you saved a working Colab with executed cells:

https://colab.research.google.com/drive/1FTUT9PINjueY3RslS-p3-IdZK56ZRjqN?usp=sharing

Prefer committing the notebook **with outputs** into `task2_genai/finetune.ipynb` on GitHub so reviewers do not need Drive access.

## Task 2 run order (Colab T4)

1. Runtime → Change runtime type → **T4 GPU**
2. Open the Task 2 Colab link above (or clone repo in Colab and open `task2_genai/finetune.ipynb`)
3. Run install cell → **Restart session**
4. Run After-restart cell → training cell
5. Push merged model to Hugging Face

**HF model:** https://huggingface.co/DilanSenanayake/phi3-compliance-qlora-merged
