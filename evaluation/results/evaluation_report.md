# Golden Evaluation Set - Evaluation Report

- Generated: 2026-09-11T13:31:02+00:00
- Baseline: `evaluation/evaluate.py`
- Golden Set: `evaluation\golden_set.labeled.json`
- Predictions: `evaluation\predictions.json`
- Records with ground truth: 216
- **Examples evaluated: 215**

- Warning: no prediction for 1 record(s): GOLDEN-0157

## Summary

| Metric | Value |
| --- | --- |
| Examples evaluated | 215 |
| Intent overall accuracy | 0.4140 |
| Intent macro F1 | 0.3132 |
| Intent weighted F1 | 0.4044 |
| Escalation accuracy | 0.6047 |
| Escalation precision (ESCALATE_TO_HUMAN positive) | 0.2195 |
| Escalation recall (ESCALATE_TO_HUMAN positive) | 0.4615 |
| Escalation F1 (ESCALATE_TO_HUMAN positive) | 0.2975 |

## Intent metrics per class

| Intent | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Technical Troubleshooting | 0.5773 | 0.5773 | 0.5773 | 97 |
| Product / Feature How-To | 0.3226 | 0.3030 | 0.3125 | 33 |
| Account & Login | 0.4545 | 0.2381 | 0.3125 | 21 |
| Billing & Payments | 0.0000 | 0.0000 | 0.0000 | 17 |
| Order / Delivery | 0.5000 | 0.8000 | 0.6154 | 5 |
| Warranty / Repair | 0.3333 | 0.7500 | 0.4615 | 4 |
| Network / Connectivity | 0.2500 | 0.2500 | 0.2500 | 4 |
| Microsoft Store | 0.0000 | 0.0000 | 0.0000 | 2 |
| Complaint / Feedback | 0.2051 | 0.2857 | 0.2388 | 28 |
| Cancellation / Subscription | 0.2857 | 0.5000 | 0.3636 | 4 |

## Escalation metrics per class

| Escalation | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| AUTO_HANDLE | 0.8421 | 0.6364 | 0.7249 | 176 |
| ESCALATE_TO_HUMAN | 0.2195 | 0.4615 | 0.2975 | 39 |
| macro | - | - | 0.5112 | - |

## Intent confusion matrix (rows = actual, columns = predicted)

```
actual \ predicted   Technical Troubleshooting    Product / Feature How-To             Account & Login          Billing & Payments            Order / Delivery           Warranty / Repair      Network / Connectivity             Microsoft Store        Complaint / Feedback Cancellation / Subscription
   Technical Troubleshooting                          56                          10                           3                           1                           2                           1                           2                           5                          16                           1
    Product / Feature How-To                          15                          10                           1                           0                           0                           2                           0                           1                           3                           1
             Account & Login                           6                           2                           5                           0                           2                           0                           0                           0                           5                           1
          Billing & Payments                           6                           2                           2                           0                           0                           1                           0                           1                           3                           2
            Order / Delivery                           0                           0                           0                           0                           4                           1                           0                           0                           0                           0
           Warranty / Repair                           0                           0                           0                           0                           0                           3                           0                           0                           1                           0
      Network / Connectivity                           3                           0                           0                           0                           0                           0                           1                           0                           0                           0
             Microsoft Store                           0                           1                           0                           0                           0                           0                           0                           0                           1                           0
        Complaint / Feedback                          11                           6                           0                           0                           0                           1                           1                           1                           8                           0
 Cancellation / Subscription                           0                           0                           0                           0                           0                           0                           0                           0                           2                           2
```

## Escalation confusion matrix (rows = actual, columns = predicted)

```
                 AUTO_HANDLE                         112                          64
           ESCALATE_TO_HUMAN                          21                          18
```

## Methodology

- Ground truth is the human-assigned `intent_label` and `escalation_label` in the Golden Set (no keyword heuristics).
- Macro/weighted F1 are averaged over the intent classes observed in the evaluated subset (scikit-learn defaults).
- Per-intent metrics and the confusion matrix cover the full 10-intent taxonomy; `zero_division=0`.
- Escalation precision/recall/F1 use `ESCALATE_TO_HUMAN` as the positive class.
- This evaluator never calls an AI model and never fabricates labels or performance values: metrics only exist if human labels and agent predictions were both supplied.
