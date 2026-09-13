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

Open [`finetune.ipynb`](finetune.ipynb) in Colab with GPU:

1. Upload/clone repo folder `task2_genai`
2. Run install + QLoRA cells
3. Confirm train/val loss per epoch (val loss should decrease)
4. `merge_and_unload()` -> push merged model to Hugging Face Hub

**HF model link (replace after push):** `https://huggingface.co/<YOUR_USER>/phi3-compliance-qlora-merged`

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
