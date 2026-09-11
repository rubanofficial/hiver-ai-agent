# Golden Evaluation Set - Evaluation Report

- Generated: 2026-09-11T13:40:26+00:00
- Baseline: `TF-IDF + Logistic Regression Baseline`
- Golden Set: `evaluation/golden_set.labeled.json`
- Predictions: `D:\WEB\project\hiver-ai-agent\evaluation\baseline_results\tfidf_logistic\predictions.json`
- Records with ground truth: 216
- **Examples evaluated: 44**

## Summary

| Metric | Value |
| --- | --- |
| Examples evaluated | 44 |
| Intent overall accuracy | 0.4545 |
| Intent macro F1 | 0.0694 |
| Intent weighted F1 | 0.2841 |
| Escalation accuracy | 0.7955 |
| Escalation precision (ESCALATE_TO_HUMAN positive) | 0.0000 |
| Escalation recall (ESCALATE_TO_HUMAN positive) | 0.0000 |
| Escalation F1 (ESCALATE_TO_HUMAN positive) | 0.0000 |

## Intent metrics per class

| Intent | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Technical Troubleshooting | 0.4545 | 1.0000 | 0.6250 | 20 |
| Product / Feature How-To | 0.0000 | 0.0000 | 0.0000 | 7 |
| Account & Login | 0.0000 | 0.0000 | 0.0000 | 4 |
| Billing & Payments | 0.0000 | 0.0000 | 0.0000 | 3 |
| Order / Delivery | 0.0000 | 0.0000 | 0.0000 | 1 |
| Warranty / Repair | 0.0000 | 0.0000 | 0.0000 | 1 |
| Network / Connectivity | 0.0000 | 0.0000 | 0.0000 | 1 |
| Microsoft Store | 0.0000 | 0.0000 | 0.0000 | 0 |
| Complaint / Feedback | 0.0000 | 0.0000 | 0.0000 | 6 |
| Cancellation / Subscription | 0.0000 | 0.0000 | 0.0000 | 1 |

## Escalation metrics per class

| Escalation | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| AUTO_HANDLE | 0.7955 | 1.0000 | 0.8861 | 35 |
| ESCALATE_TO_HUMAN | 0.0000 | 0.0000 | 0.0000 | 9 |
| macro | - | - | 0.4430 | - |

## Intent confusion matrix (rows = actual, columns = predicted)

```
actual \ predicted   Technical Troubleshooting    Product / Feature How-To             Account & Login          Billing & Payments            Order / Delivery           Warranty / Repair      Network / Connectivity             Microsoft Store        Complaint / Feedback Cancellation / Subscription
   Technical Troubleshooting                          20                           0                           0                           0                           0                           0                           0                           0                           0                           0
    Product / Feature How-To                           7                           0                           0                           0                           0                           0                           0                           0                           0                           0
             Account & Login                           4                           0                           0                           0                           0                           0                           0                           0                           0                           0
          Billing & Payments                           3                           0                           0                           0                           0                           0                           0                           0                           0                           0
            Order / Delivery                           1                           0                           0                           0                           0                           0                           0                           0                           0                           0
           Warranty / Repair                           1                           0                           0                           0                           0                           0                           0                           0                           0                           0
      Network / Connectivity                           1                           0                           0                           0                           0                           0                           0                           0                           0                           0
             Microsoft Store                           0                           0                           0                           0                           0                           0                           0                           0                           0                           0
        Complaint / Feedback                           6                           0                           0                           0                           0                           0                           0                           0                           0                           0
 Cancellation / Subscription                           1                           0                           0                           0                           0                           0                           0                           0                           0                           0
```

## Escalation confusion matrix (rows = actual, columns = predicted)

```
                 AUTO_HANDLE                          35                           0
           ESCALATE_TO_HUMAN                           9                           0
```

## Methodology

- Ground truth is the human-assigned `intent_label` and `escalation_label` in the Golden Set (no keyword heuristics).
- Macro/weighted F1 are averaged over the intent classes observed in the evaluated subset (scikit-learn defaults).
- Per-intent metrics and the confusion matrix cover the full 10-intent taxonomy; `zero_division=0`.
- Escalation precision/recall/F1 use `ESCALATE_TO_HUMAN` as the positive class.
- This evaluator never calls an AI model and never fabricates labels or performance values: metrics only exist if human labels and agent predictions were both supplied.
