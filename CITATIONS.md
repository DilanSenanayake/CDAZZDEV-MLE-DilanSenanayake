# CITATIONS.md

AI assistance and open-source adaptations for the CDAZZDEV Senior MLE Assessment.

## AI-assisted code

```
# AI-ASSISTED: Cursor Composer, Prompt: 'Plan and implement CDAZZDEV Senior MLE assessment Tasks 1-3 with yfinance indicators, Groq Pydantic LLM, LangGraph-style agents, QLoRA fine-tune pipeline', Date: 2026-09-13
```

```
# AI-ASSISTED: Cursor Composer, Prompt: 'RSI with Wilder smoothing from first principles without TA-Lib', Date: 2026-09-13
```

```
# AI-ASSISTED: Cursor Composer, Prompt: 'Build multi-agent financial research system with critique loop and agent_trace.jsonl observability', Date: 2026-09-13
```

```
# AI-ASSISTED: Cursor Composer, Prompt: 'Generate diversified financial compliance Q&A JSONL dataset and QLoRA Colab notebook with justified hyperparameters', Date: 2026-09-13
```

## Adapted open-source / libraries

```
# SOURCE: Adapted from Hugging Face PEFT/TRL QLoRA examples (https://github.com/huggingface/peft, https://github.com/huggingface/trl) for BitsAndBytes NF4 + LoRA training cell in task2_genai/finetune.ipynb
```

```
# SOURCE: yfinance public API usage patterns (https://github.com/ranaroussi/yfinance) for OHLCV and news retrieval in task1_financial/data_pipeline.py
```

```
# SOURCE: duckduckgo-search / DDGS text search usage for task3_agentic/tools/financial_tools.py web_search tool
```

## Teacher-model data generation

Full teacher system prompt: [`task2_genai/prompts/teacher_system.txt`](task2_genai/prompts/teacher_system.txt)

When `GROQ_API_KEY` is present, examples are generated with Groq `openai/gpt-oss-20b`.
When absent, `generate_dataset.py` expands a diversified offline template set across 20 topics (still >=100 examples) so the pipeline remains reproducible on free tier.
