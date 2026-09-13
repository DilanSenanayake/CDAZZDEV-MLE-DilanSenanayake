# Submission checklist

Use before sending the assessment email.

## Code & repo

- [x] All three task folders present with code + notebooks
- [x] `CITATIONS.md` present
- [x] `REFLECTION.md` present (<=600 words)
- [x] `task3_agentic/logs/agent_trace.jsonl` present
- [x] No secrets in `.env.example` (placeholders only)
- [ ] GitHub repo public as `CDAZZDEV-MLE-[YourName]` (verify in incognito)
- [ ] Notebook outputs still visible (do not Clear All Outputs)

## Keys / cloud runs (candidate)

- [ ] Add real `GROQ_API_KEY` and re-run Task 1B + Task 3 sentiment live
- [x] Run `task2_genai/finetune.ipynb` on Colab T4; confirm val loss decreases
- [x] `merge_and_unload()` and push merged model to Hugging Face
- [x] Paste HF link into root README: https://huggingface.co/DilanSenanayake/phi3-compliance-qlora-merged
- [x] Colab links documented in [`COLAB.md`](COLAB.md) and root README

## Colab paths

| Task | Colab URL |
|------|-----------|
| Task 1 | https://colab.research.google.com/github/DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake/blob/master/task1_financial/equity_research.ipynb |
| Task 2 | https://colab.research.google.com/github/DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake/blob/master/task2_genai/finetune.ipynb |
| Task 2 (Drive) | https://colab.research.google.com/drive/1FTUT9PINjueY3RslS-p3-IdZK56ZRjqN?usp=sharing |
| Task 3 | https://colab.research.google.com/github/DilanSenanayake/CDAZZDEV-MLE-DilanSenanayake/blob/master/task3_agentic/multi_agent.ipynb |

## Video & email

- [x] Record <=5 min narrated walkthrough ([VIDEO_SCRIPT.md](VIDEO_SCRIPT.md) / [VIDEO_SCRIPT.txt](VIDEO_SCRIPT.txt))
- [x] Upload YouTube unlisted or Drive (anyone with link): https://drive.google.com/file/d/1v8D2KpuHk_Z6bSf4kBqESjMFVPcZ6wlv/view?usp=sharing
- [ ] Email: GitHub + HF + Drive (if any) + video link
