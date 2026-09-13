"""Rebuild task2_genai/finetune.ipynb as a self-contained Colab notebook."""

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

cells = []

cells.append(new_markdown_cell("""# Task 2 - Domain Fine-Tuning (QLoRA)

**Use case:** Financial risk-disclosure / compliance Q&A  
**Student:** `microsoft/Phi-3-mini-4k-instruct`  
**Teacher:** Groq `openai/gpt-oss-20b`

## Before you start
1. **Runtime -> Change runtime type -> T4 GPU -> Save**
2. Run cells top-to-bottom
3. After the **Install** cell you **must** restart the runtime, then run the **After restart** cell
"""))

cells.append(new_code_cell("""# Clone or update repo (Colab)
from pathlib import Path

if not Path("CDAZZDEV-MLE-DilanSenanayake").exists():
    !git clone https://github.com/DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake.git
else:
    !git -C CDAZZDEV-MLE-DilanSenanayake pull

%cd CDAZZDEV-MLE-DilanSenanayake/task2_genai
!pwd
!ls data eval finetune.ipynb
"""))

cells.append(new_code_cell("""# GPU check — QLoRA will NOT work on CPU runtime
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
else:
    raise RuntimeError("Enable GPU: Runtime -> Change runtime type -> T4 GPU -> Save, then re-run from top")
"""))

cells.append(new_markdown_cell("""## Install dependencies

Run the cell below, then **Runtime -> Restart session** (required).
"""))

cells.append(new_code_cell("""# Pin versions tested on Colab T4; upgrade bitsandbytes for current CUDA wheels
import sys
!{sys.executable} -m pip install -q -U pip
!{sys.executable} -m pip install -q \\
    "transformers==4.46.3" "datasets==3.1.0" "peft==0.13.2" "trl==0.11.4" \\
    "accelerate==1.1.1" "evaluate" "rouge-score" "huggingface_hub"
# Install latest bitsandbytes (Colab Python 3.13 needs recent CUDA wheel)
!{sys.executable} -m pip install -q -U bitsandbytes
print("Done. NOW: Runtime -> Restart session, then run the NEXT cell only.")
"""))

cells.append(new_markdown_cell("""## After restart (run this cell first)

Do **not** skip this after restarting.
"""))

cells.append(new_code_cell("""# Post-restart setup
import os, sys, torch
from pathlib import Path

REPO = "/content/CDAZZDEV-MLE-DilanSenanayake/task2_genai"
if not Path(REPO).exists():
    REPO = str(Path.cwd() / "CDAZZDEV-MLE-DilanSenanayake" / "task2_genai")
os.chdir(REPO)
print("cwd:", os.getcwd())

assert torch.cuda.is_available(), "GPU not available — enable T4 GPU runtime"
print("GPU:", torch.cuda.get_device_name(0))

import bitsandbytes as bnb
print("bitsandbytes:", bnb.__version__)

import trl, transformers, peft
print("trl:", trl.__version__, "| transformers:", transformers.__version__, "| peft:", peft.__version__)
"""))

cells.append(new_markdown_cell("""## Hyperparameter justification

| Parameter | Value | Reason |
|-----------|-------|--------|
| LoRA r | 16 | Capacity for format learning without overfitting ~100 examples |
| LoRA alpha | 32 | alpha=2r standard PEFT scaling |
| Target modules | attn + MLP | Instruction-following on Phi-3 |
| Learning rate | 2e-4 | Standard QLoRA range |
| LR scheduler | cosine | Reduces late-epoch overfit |
| Epochs | 3 | Enough for scaffold learning on small set |
| Batch size | 1 | T4 VRAM with 4-bit + seq 1024 |
| Grad accumulation | 8 | Effective batch size 8 |
| Max seq length | 1024 | Fits compliance answers |
| Quantization | 4-bit NF4 | Rubric-required QLoRA |
| Warmup ratio | 0.03 | Stabilizes early steps |
"""))

cells.append(new_code_cell("""# Dataset split check
import json
from pathlib import Path
meta = json.loads(Path('data/split_meta.json').read_text())
print('train/val/test/total:', meta['train'], meta['val'], meta['test'], meta['total'])
print('topics:', len(meta['diversity']['topic_counts']))
"""))

cells.append(new_markdown_cell("## QLoRA training (Colab T4 GPU)"))

cells.append(new_code_cell("""# QLoRA fine-tune Phi-3-mini — Colab T4
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, SFTConfig

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
assert torch.cuda.is_available()

compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
if getattr(config, "rope_scaling", None):
    rs = dict(config.rope_scaling)
    if "type" not in rs and "rope_type" in rs:
        rs["type"] = rs["rope_type"]
    config.rope_scaling = rs

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    config=config,
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=compute_dtype,
    attn_implementation="eager",
)
print("Model loaded on GPU")

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
))
model.print_trainable_parameters()

train_ds = load_dataset("json", data_files="data/train.jsonl", split="train")
val_ds = load_dataset("json", data_files="data/val.jsonl", split="train")

def to_text(example):
    parts = [f"<|{m['role']}|>\\n{m['content']}<|end|>\\n" for m in example["messages"]]
    parts.append("<|assistant|>\\n")
    return {"text": "".join(parts)}

train_ds = train_ds.map(to_text)
val_ds = val_ds.map(to_text)

# trl 0.11.4: SFTConfig + tokenizer= (NOT processing_class)
_common = dict(
    output_dir="checkpoints/phi3-compliance-qlora",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=5,
    save_strategy="epoch",
    bf16=(compute_dtype == torch.bfloat16),
    fp16=(compute_dtype == torch.float16),
    report_to=["none"],
    max_grad_norm=0.3,
    max_seq_length=1024,
    dataset_text_field="text",
)
try:
    sft_args = SFTConfig(**_common, eval_strategy="epoch")
except TypeError:
    sft_args = SFTConfig(**_common, evaluation_strategy="epoch")

trainer = SFTTrainer(
    model=model,
    args=sft_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
)
print("SFTTrainer ready (trl", __import__("trl").__version__, ")")

train_result = trainer.train()
metrics_epoch = trainer.evaluate()
print("Train:", train_result)
print("Eval:", metrics_epoch)

merged = model.merge_and_unload()
merged.save_pretrained("merged_model")
tokenizer.save_pretrained("merged_model")
print("Saved merged_model/")
"""))

cells.append(new_markdown_cell("""## Push to Hugging Face (optional)

Set HF token in Colab Secrets as `HF_TOKEN`, then run:
"""))

cells.append(new_code_cell("""# from huggingface_hub import login
# from google.colab import userdata
# login(userdata.get('HF_TOKEN'))
# merged.push_to_hub('DilanSenanayake/phi3-compliance-qlora-merged')
# tokenizer.push_to_hub('DilanSenanayake/phi3-compliance-qlora-merged')
# print('Pushed to Hugging Face')
"""))

cells.append(new_markdown_cell("## Evaluation (Task 2C)"))

cells.append(new_code_cell("""import json
from pathlib import Path
m = json.loads(Path('eval/metrics.json').read_text())
print('ROUGE-L:', m['rougeL'])
print('Keyword F1:', m['keyword_compliance_f1'])
print('Hallucination %:', m['hallucination_rate_pct'])
"""))

cells.append(new_markdown_cell("""## Qualitative analysis

Fine-tuning improved structured compliance answers (principle, must/must-not, hallucination guard, caveat) versus generic base responses. Remaining gaps: industry-specific nuance and multi-part questions — more diverse teacher data and preference tuning would help.
"""))

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "colab": {"provenance": []},
})

out = __import__("pathlib").Path(__file__).resolve().parents[1] / "task2_genai" / "finetune.ipynb"
nbformat.write(nb, out)
print("Wrote", out, "cells:", len(cells))
