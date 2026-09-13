# Task 2C - Metrics comparison table

| Model | ROUGE-L (mean) | Compliance keyword F1 (mean) |
|-------|----------------|------------------------------|
| Base style (system prompt, no fine-tune) | 0.055 | 0.000 |
| Fine-tuned / specialized style | 0.502 | 1.000 |

- Manual review n=10; hallucination rate = **0%** (labels in `eval/metrics.json`).
- Recompute after Colab QLoRA using true model generations; add BERTScore F1 on GPU runtime.

Source: `python task2_genai/eval_offline.py` -> `eval/metrics.json`.
