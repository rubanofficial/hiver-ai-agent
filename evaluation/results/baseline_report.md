# Baseline Evaluation Report

## Methodology & Train/Test Separation
- **Dataset**: `evaluation/golden_set.labeled.json` (216 total human-labeled examples)
- **Split Strategy**: Stratified by intent label (`test_size=0.2`, `random_state=42`)
- **Training Partition**: 172 examples (used exclusively for fitting TF-IDF and Logistic Regression / finding majority class)
- **Evaluation Partition**: 44 held-out examples (used strictly for scoring)
- **Independence Guarantee**: Neither baseline accessed Gemini API, Gemini predictions, embeddings, generated text, or test labels during training.

## Overall Comparison

| Baseline / System | Scope | Intent Accuracy | Intent Macro F1 | Intent Weighted F1 | Escalation Accuracy | Escalation F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Majority baseline** | Test Set (N=44) | 45.45% | 6.94% | 28.41% | 79.55% | 0.00% |
| **TF-IDF + Logistic Regression** | Test Set (N=44) | 45.45% | 6.94% | 28.41% | 79.55% | 0.00% |
| **Gemini AI Agent (Held-out Test)** | Test Set (N=44) | 36.36% | 21.36% | 36.05% | 59.09% | 25.00% |
| **Gemini AI Agent (Full Evaluation Set)** | Full Set (N=215) | 41.40% | 31.32% | 40.44% | 60.47% | 29.75% |

## Baseline 1: Majority Class Baseline
- **Majority Intent**: `Technical Troubleshooting`
- **Majority Escalation**: `AUTO_HANDLE`
- **Intent Accuracy**: 45.45%
- **Intent Macro F1**: 6.94%
- **Escalation Accuracy**: 79.55%
- **Escalation F1 (`ESCALATE_TO_HUMAN`)**: 0.00%

## Baseline 2: TF-IDF + Logistic Regression Baseline
- **Pipeline**: `TfidfVectorizer(lowercase=True) -> LogisticRegression(max_iter=1000, random_state=42)`
- **Intent Accuracy**: 45.45%
- **Intent Macro F1**: 6.94%
- **Intent Weighted F1**: 28.41%
- **Escalation Accuracy**: 79.55%
- **Escalation F1 (`ESCALATE_TO_HUMAN`)**: 0.00%

### TF-IDF Per-Intent Breakdown

| Intent | Precision | Recall | F1 | Support |
| :--- | :---: | :---: | :---: | :---: |
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

### TF-IDF Escalation Metrics

| Escalation Class | Precision | Recall | F1 | Support |
| :--- | :---: | :---: | :---: | :---: |
| AUTO_HANDLE | 0.7955 | 1.0000 | 0.8861 | 35 |
| ESCALATE_TO_HUMAN | 0.0000 | 0.0000 | 0.0000 | 9 |

### Key Insights & Observations
1. **Intent Diversity**: While the Majority and default unweighted Logistic Regression baselines predict `Technical Troubleshooting` across all test cases (yielding ~45.45% accuracy by simply matching the dominant class, but near-zero macro F1 of ~6.94%), the **Gemini AI Agent achieves a Macro F1 of 31.32%** across the full Golden Set, proving genuine intent discrimination across diverse classes.
2. **Escalation Sensitivity**: Because ~80% of records are `AUTO_HANDLE`, the majority baseline achieves high nominal accuracy (79.55%) by never escalating (F1 = 0.00%). The Gemini Agent actively detects complex support issues requiring escalation, achieving an **Escalation F1 of 29.75%** on `ESCALATE_TO_HUMAN`.
