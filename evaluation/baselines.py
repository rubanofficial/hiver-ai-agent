"""
Simple, deterministic baselines for the Golden Evaluation Set.

Two families of baselines live here, both kept completely separate from the
production Gemini-powered agent:

* **Majority baseline** -- predicts the most frequent (majority) class for
  intent and for escalation.  The majority class is always computed from the
  labeled training data; it is never hardcoded.
* **TF-IDF + Logistic Regression baseline** -- a plain scikit-learn text
  classifier (``TfidfVectorizer`` + ``LogisticRegression``) trained only on
  the *training* portion of the labeled Golden Set and evaluated only on the
  *test* portion.  Features are ``customer_message`` + ``conversation_context``.

Nothing in this module calls Gemini or any other LLM, generates labels, or
fabricates results.  Train/test separation is enforced: a model is never
trained on an example and then evaluated on that same example.

The metrics themselves are NOT re-implemented here: every evaluation reuses
``evaluation/evaluate.py`` (``extract_ground_truth``, ``validate_predictions``,
``assemble_metrics``, ``compute_*_metrics``, result writers, report writer).
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from evaluation.evaluate import (
    EvaluationError,
    extract_ground_truth,
    overlay_labels,
    payload_records,
    validate_predictions,
    assemble_metrics,
    write_results,
    write_report,
    load_taxonomy,
)

RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2

LABEL_FIELDS = ("intent_label", "escalation_label")

MAJORITY_INTENT_NAME = "Majority Intent Baseline"
MAJORITY_ESCALATION_NAME = "Majority Escalation Baseline"
TFIDF_NAME = "TF-IDF + Logistic Regression Baseline"


class BaselineError(Exception):
    """Raised when a baseline cannot be fit or evaluated (blocking condition)."""


# ---------------------------------------------------------------------------
# Data loading (reuses the evaluator's label validation)
# ---------------------------------------------------------------------------

def load_labeled_records(
    golden_set_path: str,
    labels_path: Optional[str] = None,
    intents_yaml: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load the Golden Set records with their human labels.

    Raises ``EvaluationError`` (with a clear message) if the Golden Set is not
    fully human-labeled -- the baselines never generate labels themselves.
    """
    taxonomy = load_taxonomy(intents_yaml)
    golden = Path(golden_set_path)
    if not golden.exists():
        raise BaselineError(f"Golden Set not found: {golden}")
    with open(golden, "r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    records = copy.deepcopy(payload_records(payload))
    if labels_path:
        overlay_labels(records, labels_path)
    truth = extract_ground_truth(records, taxonomy)
    by_id = {t["golden_id"]: t for t in truth}
    for rec in records:
        t = by_id[rec["golden_id"]]
        rec["intent_label"] = t["intent_label"]
        rec["escalation_label"] = t["escalation_label"]
    return records


def record_text(record: Dict[str, Any]) -> str:
    """Features for the ML baseline: customer_message + conversation_context."""
    message = str(record.get("customer_message") or "")
    context = str(record.get("conversation_context") or "")
    return f"{message}\n{context}"


# ---------------------------------------------------------------------------
# Reproducible train/test split
# ---------------------------------------------------------------------------

def stratified_split(
    records: List[Dict[str, Any]],
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split labeled records into (train, test) with a fixed random seed.

    Stratification by intent label is used whenever every intent class has at
    least two members (sklearn's requirement); otherwise the split falls back
    to a plain, still-reproducible ``random_state`` split.  The split is
    deterministic: the same inputs and seed always produce the same slices.
    """
    if len(records) < 2:
        raise BaselineError(
            f"Cannot split: need at least 2 labeled records, got {len(records)}."
        )
    ids = [str(r["golden_id"]) for r in records]
    labels = [r["intent_label"] for r in records]
    by_id = {str(r["golden_id"]): r for r in records}

    counts = Counter(labels)
    if not counts:
        raise BaselineError("Cannot split: labeled records have no intent labels.")

    stratify_arg: Optional[List[str]] = labels
    if len(counts) < 2 or min(counts.values()) < 2:
        stratify_arg = None  # stratification impossible -> fall back

    try:
        train_ids, test_ids = train_test_split(
            ids,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arg,
        )
    except ValueError:
        if stratify_arg is not None:
            # Unexpected stratification failure (e.g. test_size edge case):
            # retry as a plain reproducible split.
            train_ids, test_ids = train_test_split(
                ids, test_size=test_size, random_state=random_state
            )
        else:
            raise BaselineError("Cannot split the labeled records into train/test.")

    return [by_id[i] for i in train_ids], [by_id[i] for i in test_ids]


# ---------------------------------------------------------------------------
# Majority baseline
# ---------------------------------------------------------------------------

def majority_classes(
    records: List[Dict[str, Any]],
) -> Tuple[str, str]:
    """Most frequent intent and escalation label among ``records`` (no hardcoding)."""
    if not records:
        raise BaselineError("Cannot compute the majority class from zero records.")
    intent_counts = Counter(r["intent_label"] for r in records)
    esc_counts = Counter(r["escalation_label"] for r in records)
    if not intent_counts or not esc_counts:
        raise BaselineError("Cannot compute the majority class: labels are missing.")
    return intent_counts.most_common(1)[0][0], esc_counts.most_common(1)[0][0]


def make_prediction(
    golden_id: str,
    intent: str,
    escalation: str,
    confidence: float = 1.0,
) -> Dict[str, Any]:
    """Build a prediction dict in the evaluator's prediction-file format."""
    return {
        "golden_id": golden_id,
        "predicted_intent": intent,
        "predicted_intent_confidence": confidence,
        "predicted_escalation": escalation,
        "predicted_reply": "",
        "retrieved_evidence": [],
    }


def predict_majority(
    records: List[Dict[str, Any]],
    majority_intent: str,
    majority_escalation: str,
) -> List[Dict[str, Any]]:
    """Predict the majority intent + majority escalation for every record."""
    return [
        make_prediction(r["golden_id"], majority_intent, majority_escalation)
        for r in records
    ]


# ---------------------------------------------------------------------------
# TF-IDF + Logistic Regression baseline
# ---------------------------------------------------------------------------

class TfidfLogisticClassifier:
    """Plain TF-IDF + Logistic Regression baseline classifier.

    ``label_field`` selects which human label to learn:
    ``"intent_label"`` (multi-class) or ``"escalation_label"`` (binary).
    """

    def __init__(self, label_field: str, random_state: int = RANDOM_STATE,
                 max_iter: int = 1000):
        if label_field not in LABEL_FIELDS:
            raise ValueError(
                f"label_field must be one of {LABEL_FIELDS}; got {label_field!r}"
            )
        self.label_field = label_field
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(lowercase=True)
        self.estimator = LogisticRegression(
            max_iter=max_iter, random_state=random_state
        )
        self._fitted = False

    def fit(self, train_records: List[Dict[str, Any]]) -> "TfidfLogisticClassifier":
        texts = [record_text(r) for r in train_records]
        labels = [r[self.label_field] for r in train_records]
        distinct = set(labels)
        if not distinct:
            raise BaselineError(
                f"Cannot train: no {self.label_field} values in the training data."
            )
        if len(distinct) < 2:
            raise BaselineError(
                f"Cannot train a logistic regression classifier when the training "
                f"data has a single {self.label_field} value: {sorted(distinct)}"
            )
        X = self.vectorizer.fit_transform(texts)
        self.estimator.fit(X, labels)
        self._fitted = True
        return self

    def predict(self, records: List[Dict[str, Any]]) -> List[str]:
        if not self._fitted:
            raise BaselineError("Classifier must be fitted before predict().")
        X = self.vectorizer.transform([record_text(r) for r in records])
        return list(self.estimator.predict(X))


def predict_with_classifiers(
    test_records: List[Dict[str, Any]],
    intent_clf: TfidfLogisticClassifier,
    escalation_clf: TfidfLogisticClassifier,
) -> List[Dict[str, Any]]:
    """Predict intent (10 classes) + escalation (binary) for the test records."""
    intents = intent_clf.predict(test_records)
    escalations = escalation_clf.predict(test_records)
    return [
        make_prediction(r["golden_id"], intent, esc, confidence=1.0)
        for r, intent, esc in zip(test_records, intents, escalations)
    ]


# ---------------------------------------------------------------------------
# Evaluation of baseline predictions (reuses the evaluator)
# ---------------------------------------------------------------------------

def evaluate_predictions(
    predictions: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    taxonomy: List[str],
    *,
    golden_set_path: str,
    labels_path: Optional[str],
    golden_total: int,
    predictions_path: str,
    results_dir: str,
    report_path: str,
    source_label: str,
) -> Tuple[Dict[str, Any], List[Path]]:
    """Score baseline predictions against the TEST ground truth only.

    Everything metric-related is delegated to ``evaluation/evaluate.py`` so the
    baseline numbers are fully comparable with the agent's numbers.
    """
    truth = extract_ground_truth(test_records, taxonomy)
    pairs, warnings = validate_predictions(truth, predictions, taxonomy)
    metrics = assemble_metrics(
        pairs=pairs,
        warnings=warnings,
        taxonomy=taxonomy,
        golden_set_path=str(golden_set_path),
        predictions_path=str(predictions_path),
        golden_total=golden_total,
        labels_path=labels_path,
        source_label=source_label,
    )
    written = write_results(metrics, str(results_dir))
    written.append(write_report(metrics, str(report_path)))
    return metrics, written