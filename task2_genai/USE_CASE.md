# Use Case - Financial Risk Disclosure / Compliance Q&A

## Problem statement

| Field | Definition |
|-------|------------|
| **Input** | A natural-language question about financial risk disclosures, earnings risk-factor wording, or capital-markets compliance policy interpretation. |
| **Output** | A structured answer with: (1) applicable disclosure principle, (2) must / must-not claims, (3) hallucination guard for missing facts, (4) non-legal-advice caveat. |
| **Correct** | No invented statute numbers or filing dates; probabilistic language; distinguishes known vs unknown; refuses investment guarantees. |
| **Incorrect** | Fabricated regulation citations, promotional certainty, or asserted non-public material facts. |

## Models

- **Teacher (data generation):** Groq `openai/gpt-oss-20b` (or offline diversified templates if no API key)
- **Student (fine-tune):** `microsoft/Phi-3-mini-4k-instruct` via QLoRA 4-bit NF4 on Colab T4

Teacher system prompt: [`prompts/teacher_system.txt`](prompts/teacher_system.txt)
