# Golden Evaluation Set - Baselines

- Generated: 2026-09-11T13:40:26+00:00
- Split: single deterministic train/test split (fixed random_state), stratified by intent label where possible (seed 42, test size 0.2)
- Total labeled records: 216
- Train examples: 172 / Test examples: 44

## Comparison

| Baseline | Intent acc | Intent macro F1 | Intent weighted F1 | Escalation acc | Escalation F1 (ESCALATE_TO_HUMAN) |
| --- | --- | --- | --- | --- | --- |
| Majority Intent Baseline + Majority Escalation Baseline | 0.4545 | 0.0694 | 0.2841 | 0.7955 | 0.0000 |
| TF-IDF + Logistic Regression Baseline | 0.4545 | 0.0694 | 0.2841 | 0.7955 | 0.0000 |

Details per baseline (metrics.json, confusion matrices, per-intent precision/recall/F1 and report) live in the directories above.
