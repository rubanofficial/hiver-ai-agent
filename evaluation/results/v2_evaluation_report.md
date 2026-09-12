# Golden Evaluation Set - Evaluation Report

- Generated: 2026-09-12T04:12:41+00:00
- Baseline: `v2_production_agent`
- Golden Set: `evaluation/golden_set.json`
- Predictions: `evaluation/predictions_v2.json`
- Records with ground truth: 216
- **Examples evaluated: 216**
- Human labels overlaid from: `evaluation/golden_set.labels.json`

## Summary

| Metric | Value |
| --- | --- |
| Examples evaluated | 216 |
| Intent overall accuracy | 0.4028 |
| Intent macro F1 | 0.2880 |
| Intent weighted F1 | 0.3979 |
| Escalation accuracy | 0.6343 |
| Escalation precision (ESCALATE_TO_HUMAN positive) | 0.2368 |
| Escalation recall (ESCALATE_TO_HUMAN positive) | 0.4615 |
| Escalation F1 (ESCALATE_TO_HUMAN positive) | 0.3130 |

## Intent metrics per class

| Intent | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Technical Troubleshooting | 0.5851 | 0.5612 | 0.5729 | 98 |
| Product / Feature How-To | 0.3448 | 0.3030 | 0.3226 | 33 |
| Account & Login | 0.3846 | 0.2381 | 0.2941 | 21 |
| Billing & Payments | 0.0000 | 0.0000 | 0.0000 | 17 |
| Order / Delivery | 0.4444 | 0.8000 | 0.5714 | 5 |
| Warranty / Repair | 0.3333 | 0.7500 | 0.4615 | 4 |
| Network / Connectivity | 0.2000 | 0.2500 | 0.2222 | 4 |
| Microsoft Store | 0.0000 | 0.0000 | 0.0000 | 2 |
| Complaint / Feedback | 0.2000 | 0.2857 | 0.2353 | 28 |
| Cancellation / Subscription | 0.1667 | 0.2500 | 0.2000 | 4 |

## Escalation metrics per class

| Escalation | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| AUTO_HANDLE | 0.8500 | 0.6723 | 0.7508 | 177 |
| ESCALATE_TO_HUMAN | 0.2368 | 0.4615 | 0.3130 | 39 |
| macro | - | - | 0.5319 | - |

## Intent confusion matrix (rows = actual, columns = predicted)

```
actual \ predicted   Technical Troubleshooting    Product / Feature How-To             Account & Login          Billing & Payments            Order / Delivery           Warranty / Repair      Network / Connectivity             Microsoft Store        Complaint / Feedback Cancellation / Subscription
   Technical Troubleshooting                          55                           8                           4                           1                           2                           1                           2                           6                          18                           1
    Product / Feature How-To                          15                          10                           1                           0                           1                           2                           0                           1                           2                           1
             Account & Login                           6                           2                           5                           0                           2                           0                           0                           0                           5                           1
          Billing & Payments                           6                           2                           2                           0                           0                           1                           0                           1                           3                           2
            Order / Delivery                           0                           0                           0                           0                           4                           1                           0                           0                           0                           0
           Warranty / Repair                           0                           0                           0                           0                           0                           3                           0                           0                           1                           0
      Network / Connectivity                           3                           0                           0                           0                           0                           0                           1                           0                           0                           0
             Microsoft Store                           0                           1                           0                           0                           0                           0                           0                           0                           1                           0
        Complaint / Feedback                           9                           6                           1                           0                           0                           1                           2                           1                           8                           0
 Cancellation / Subscription                           0                           0                           0                           1                           0                           0                           0                           0                           2                           1
```

## Escalation confusion matrix (rows = actual, columns = predicted)

```
                 AUTO_HANDLE                         119                          58
           ESCALATE_TO_HUMAN                          21                          18
```

## Methodology

- Ground truth is the human-assigned `intent_label` and `escalation_label` in the Golden Set (no keyword heuristics).
- Macro/weighted F1 are averaged over the intent classes observed in the evaluated subset (scikit-learn defaults).
- Per-intent metrics and the confusion matrix cover the full 10-intent taxonomy; `zero_division=0`.
- Escalation precision/recall/F1 use `ESCALATE_TO_HUMAN` as the positive class.
- This evaluator never calls an AI model and never fabricates labels or performance values: metrics only exist if human labels and agent predictions were both supplied.
