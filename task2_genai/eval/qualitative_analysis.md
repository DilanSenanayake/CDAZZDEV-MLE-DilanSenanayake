## Qualitative analysis

Fine-tuning (and the specialized target style) improved behaviour by forcing a stable
four-part structure: disclosure principle, must/must-not constraints, hallucination guard,
and an explicit non-legal-advice caveat. On held-out prompts about liquidity risk and
forward-looking statements, the specialized outputs repeatedly refused to invent statute
sections and instead asked for missing filing context, whereas the base-style answers stayed
generic and omitted actionable compliance guardrails. ROUGE-L and the compliance-keyword
F1 proxy both rose because references share this scaffold.

Remaining failure modes include over-general templates that under-specify industry nuance
(e.g., bank capital adequacy vs biotech going-concern language) and occasional partial
answers when the user question packs multiple sub-questions. Additional diverse teacher
data covering jurisdiction-specific regimes, plus preference/DPO on hallucinated citations,
would further reduce template rigidity and citation invention risk.
