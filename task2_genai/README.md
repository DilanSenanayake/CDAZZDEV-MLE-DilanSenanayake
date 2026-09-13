# Task 2 - Domain-Specific Fine-Tuning Pipeline

Financial **risk-disclosure / compliance Q&A** fine-tuning with QLoRA and rigorous evaluation.

## Problem statement

See [`USE_CASE.md`](USE_CASE.md).

## Dataset

```bash
python task2_genai/generate_dataset.py
```

- >=100 examples (120 generated)
- Chat JSONL with system/user/assistant turns
- Split 80/10/10 recorded in `data/split_meta.json`
- Diversity: topic histogram + prompt-length stats + keyword frequencies
- Teacher prompt: [`prompts/teacher_system.txt`](prompts/teacher_system.txt)

## Fine-tuning (Colab T4)

**Repo path:** `task2_genai/finetune.ipynb`

Open in Colab (GPU required):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake/blob/master/task2_genai/finetune.ipynb)

- GitHub to Colab: https://colab.research.google.com/github/DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake/blob/master/task2_genai/finetune.ipynb
- Drive copy (executed run): https://colab.research.google.com/drive/1FTUT9PINjueY3RslS-p3-IdZK56ZRjqN?usp=sharing

**Run order:**
1. Runtime -> **T4 GPU**
2. Clone/pull + install cells
3. **Restart runtime** after install
4. Run **After restart** cell, then training cell
5. Confirm val loss decreases; merge + push to Hugging Face Hub

**HF model link:** https://huggingface.co/DilanSenanayake/phi3-compliance-qlora-merged

## Evaluation

```bash
python task2_genai/eval_offline.py
```

Produces:

- ROUGE-L base vs specialized comparison (`eval/metrics.json`)
- Secondary compliance-keyword F1 proxy (BERTScore recommended on Colab)
- Manual review of 10 responses + hallucination rate
- Qualitative analysis in `eval/qualitative_analysis.md`

## Hyperparameters

Fully justified inside `finetune.ipynb` (r, alpha, targets, LR, scheduler, epochs, batch, grad accum, max length, NF4).
