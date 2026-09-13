"""
Generate >=100 diverse financial risk-disclosure / compliance Q&A examples.

Primary path: Groq teacher model (Llama-3.70B) when GROQ_API_KEY is set.
Fallback: deterministic template expansion with topic diversity (offline).

Teacher system prompt is saved to task2_genai/prompts/teacher_system.txt
"""

from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "task2_genai" / "data"
PROMPTS = ROOT / "task2_genai" / "prompts"
DATA.mkdir(parents=True, exist_ok=True)
PROMPTS.mkdir(parents=True, exist_ok=True)

TEACHER_SYSTEM = """You are a senior financial compliance and risk-disclosure specialist.
Generate training pairs for a domain assistant.

USE CASE:
- Input: a user question about financial risk disclosures, SEC/company reporting language,
  earnings risk factors, or compliance policy interpretation in capital markets.
- Output: a precise answer that (1) states applicable disclosure principle,
  (2) lists what must / must not be claimed, (3) flags hallucination risks,
  (4) ends with a one-line confidence caveat.
- Correct: cites general public disclosure norms, no fabricated regulation numbers,
  no invented filing dates, distinguishes known vs unknown facts.
- Incorrect: invents statute citations, guarantees investment outcomes, or asserts
  non-public material facts.

Return ONLY JSON:
{"user": "...", "assistant": "...", "topic": "..."}
"""

PROMPTS.joinpath("teacher_system.txt").write_text(TEACHER_SYSTEM, encoding="utf-8")

TOPICS = [
    "forward_looking_statements",
    "material_adverse_change",
    "liquidity_risk",
    "credit_risk",
    "market_risk",
    "cybersecurity_disclosure",
    "related_party_transactions",
    "revenue_recognition_risk",
    "going_concern",
    "segment_reporting",
    "esg_claims_guardrails",
    "insider_trading_policy",
    "fair_disclosure_reg_fd",
    "earnings_guidance_language",
    "contingent_liabilities",
    "goodwill_impairment",
    "fx_translation_risk",
    "supply_chain_concentration",
    "litigation_contingency",
    "capital_adequacy",
]

SCENARIOS = [
    "How should a company describe {topic} in an annual report risk factor?",
    "A junior analyst wants to claim certainty about {topic}. What is compliant wording?",
    "Rewrite this unsafe claim about {topic}: 'Investors are guaranteed protection.'",
    "What evidence is required before asserting improvement in {topic}?",
    "List three disclosure mistakes teams make regarding {topic}.",
    "Draft a cautious FAQ answer for clients asking about {topic}.",
    "Contrast promotional marketing language vs compliant disclosure for {topic}.",
    "When is silence about {topic} itself a disclosure risk?",
    "Provide a checklist for reviewing {topic} statements before publication.",
    "Explain how {topic} interacts with forward-looking statement safe harbors.",
]


def _assistant_for(topic: str, question: str) -> str:
    return (
        f"Disclosure principle: statements about {topic.replace('_', ' ')} must be "
        f"balanced, non-promissory, and consistent with filed risk factors.\n"
        f"Must: use probabilistic language, cite only filed or clearly labeled illustrative "
        f"facts, separate known historical metrics from uncertain forward views.\n"
        f"Must not: invent regulation numbers, assert guaranteed outcomes, or present "
        f"non-public material information as fact.\n"
        f"Hallucination guard: if a specific filing date, statute section, or numeric "
        f"threshold was not provided in the question, state that it is unavailable rather "
        f"than fabricating it.\n"
        f"Response to the question: Address '{question[:120]}' by recommending cautious "
        f"wording, documenting assumptions, and escalating legal review for novel claims.\n"
        f"Caveat: This is educational guidance, not legal advice; confirm with counsel."
    )


def generate_offline(n: int = 120, seed: int = 42) -> list[dict]:
    random.seed(seed)
    rows: list[dict] = []
    i = 0
    while len(rows) < n:
        topic = TOPICS[i % len(TOPICS)]
        template = SCENARIOS[i % len(SCENARIOS)]
        # diversify entities / industries
        industry = ["bank", "semiconductor", "retailer", "airline", "saas", "biotech"][i % 6]
        region = ["US", "EU", "APAC"][i % 3]
        q = template.format(topic=topic.replace("_", " "))
        q = f"[{industry}/{region}] {q} (variant {i})"
        rows.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a financial risk-disclosure and compliance assistant. "
                            "Prefer cautious, non-hallucinated answers."
                        ),
                    },
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": _assistant_for(topic, q)},
                ],
                "topic": topic,
                "source": "offline_template_expansion",
            }
        )
        i += 1
    return rows


def generate_with_groq(n: int = 120) -> list[dict]:
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    rows: list[dict] = []
    for i in range(n):
        topic = TOPICS[i % len(TOPICS)]
        user_req = (
            f"Create training example #{i+1} on topic '{topic}'. "
            f"Vary industry and difficulty. JSON only."
        )
        resp = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            messages=[
                {"role": "system", "content": TEACHER_SYSTEM},
                {"role": "user", "content": user_req},
            ],
            temperature=0.8,
            response_format={"type": "json_object"},
        )
        raw = json.loads(resp.choices[0].message.content or "{}")
        rows.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a financial risk-disclosure and compliance assistant. "
                            "Prefer cautious, non-hallucinated answers."
                        ),
                    },
                    {"role": "user", "content": raw["user"]},
                    {"role": "assistant", "content": raw["assistant"]},
                ],
                "topic": raw.get("topic", topic),
                "source": "groq_teacher",
            }
        )
    return rows


def diversity_report(rows: list[dict]) -> dict:
    lengths = [len(r["messages"][1]["content"].split()) for r in rows]
    topics = Counter(r.get("topic", "unknown") for r in rows)
    # keyword frequency on user prompts
    bag: Counter[str] = Counter()
    for r in rows:
        for tok in re.findall(r"[a-zA-Z]{4,}", r["messages"][1]["content"].lower()):
            bag[tok] += 1
    return {
        "n_examples": len(rows),
        "prompt_length_words": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
            "p50": sorted(lengths)[len(lengths) // 2],
        },
        "topic_counts": dict(topics),
        "top_keywords": bag.most_common(25),
    }


def split_and_write(rows: list[dict]) -> dict:
    random.shuffle(rows)
    n = len(rows)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train, val, test = rows[:n_train], rows[n_train : n_train + n_val], rows[n_train + n_val :]

    def dump(name: str, data: list[dict]) -> None:
        path = DATA / name
        with path.open("w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")

    dump("train.jsonl", train)
    dump("val.jsonl", val)
    dump("test.jsonl", test)
    meta = {
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "total": n,
        "diversity": diversity_report(rows),
    }
    (DATA / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key and not key.startswith("your_"):
        print("Generating with Groq teacher...")
        rows = generate_with_groq(120)
    else:
        print("No GROQ_API_KEY - generating offline diverse template set...")
        rows = generate_offline(120)
    meta = split_and_write(rows)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
