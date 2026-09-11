"""
Golden Evaluation Set evaluator.

Measures the support agent's performance against the 200 human-labeled Golden
Evaluation Set (``evaluation/golden_set.json``).

Ground truth comes ONLY from human labels (``intent_label`` + ``escalation_label``
filled in by a person via the labeling workflow).  Agent predictions are
supplied SEPARATELY in their own JSON file, so this module never calls the
agent, never calls Gemini, never labels anything, and never invents numbers.

Prediction file format (a JSON list of objects, one per record)::

    [
      {
        "golden_id": "GOLDEN-0001",
        "predicted_intent": "Order / Delivery",        # one of the 10 intents
        "predicted_intent_confidence": 0.93,            # optional
        "predicted_escalation": "AUTO_HANDLE",          # AUTO_HANDLE | ESCALATE_TO_HUMAN
        "predicted_reply": "We can update the address ...",   # optional
        "retrieved_evidence": [ ... ]                  # optional
      },
      ...
    ]

Predictions are matched to Golden Set records by ``golden_id``.  By default
the evaluator looks for ``evaluation/predictions.json``.  If that file does
not exist (or the Golden Set has no human labels yet), the evaluator fails
with a clear message instead of fabricating numbers.

Usage::

    python -m evaluation.evaluate --predictions evaluation/predictions.json
    python -m evaluation.evaluate
    python -m evaluation.evaluate --labels-file evaluation/golden_set.labels.json

Machine-readable results are written to ``evaluation/results/``
(``metrics.json``, ``confusion_matrix.csv``, ``per_intent_metrics.csv``,
``escalation_confusion_matrix.csv``) and a human-readable report to
``evaluation/evaluation_report.md``.

Metric conventions
------------------
* Intent metrics use scikit-learn.  Aggregate macro/weighted F1 are averaged
  over the intent classes observed in the evaluated example subset (sklearn's
  default), so perfect predictions give F1 = 1.0.  Per-intent metrics and the
  confusion matrix always cover the full 10-intent taxonomy.
* Escalation metrics: accuracy plus per-class precision/recall/F1, and binary
  metrics with ``ESCALATE_TO_HUMAN`` as the positive class (the action that
  really matters for support quality).
* ``zero_division=0`` is used throughout, so classes with no examples yield
  precision/recall/F1 of 0.0 rather than raising.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN_SET_PATH = PROJECT_ROOT / "evaluation" / "golden_set.json"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "evaluation" / "predictions.json"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "evaluation" / "evaluation_report.md"
DEFAULT_INTENTS_YAML = PROJECT_ROOT / "config" / "intents.yaml"

SCHEMA_VERSION = "1.0.0"
EVALUATOR_NAME = "evaluation/evaluate.py"

# Valid escalation values; must match the runtime agent's enum exactly.
ESCALATION_VALUES = ["AUTO_HANDLE", "ESCALATE_TO_HUMAN"]
ESCALATION_POSITIVE_CLASS = "ESCALATE_TO_HUMAN"

PREDICTION_REQUIRED_FIELDS = [
    "golden_id",
    "predicted_intent",
    "predicted_escalation",
]


class EvaluationError(Exception):
    """Raised when the evaluation cannot be performed (blocking condition)."""


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_taxonomy(intents_yaml: Optional[str] = None) -> List[str]:
    """Read the intent names (read-only) from ``config/intents.yaml``."""
    path = Path(intents_yaml) if intents_yaml else DEFAULT_INTENTS_YAML
    if not path.exists():
        raise EvaluationError(f"Intent taxonomy file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    names = [
        entry["name"]
        for entry in (cfg.get("intents") or [])
        if isinstance(entry, dict) and entry.get("name")
    ]
    if len(names) != len(set(names)):
        raise EvaluationError("Intent taxonomy contains duplicate intent names.")
    if not names:
        raise EvaluationError("Intent taxonomy is empty; nothing to evaluate against.")
    return names


def payload_records(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = payload.get("records")
    if not isinstance(records, list):
        raise EvaluationError("Golden Set payload has no 'records' list.")
    return records


def overlay_labels(records: List[Dict[str, Any]], labels_path: str) -> None:
    """Fill label fields in-memory from a labels store (``golden_set.labels.json``).

    This is a read-only overlay on working copies: no file is modified and the
    existing human labels in the store are never changed.
    """
    path = Path(labels_path)
    if not path.exists():
        raise EvaluationError(f"Labels file not found: {path}")
    with open(path, "r", encoding="utf-8-sig") as f:
        store = json.load(f)
    table = store.get("labels")
    if not isinstance(table, dict):
        raise EvaluationError(f"Invalid labels file: missing 'labels' map in {path}")
    for rec in records:
        entry = table.get(rec["golden_id"])
        if isinstance(entry, dict):
            rec["intent_label"] = (entry.get("intent_label") or "").strip()
            rec["escalation_label"] = (entry.get("escalation_label") or "").strip()
            rec["notes"] = entry.get("notes", "")


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def extract_ground_truth(
    records: List[Dict[str, Any]],
    taxonomy: List[str],
) -> List[Dict[str, str]]:
    """Validate that EVERY record has the required human labels.

    Returns the ground-truth rows (golden_id, intent_label, escalation_label).
    Raises ``EvaluationError`` listing every record with a missing or invalid
    label, because evaluation must never run on incomplete ground truth.
    """
    issues: List[str] = []
    truth: List[Dict[str, str]] = []
    for rec in records:
        gid = rec.get("golden_id")
        intent = (rec.get("intent_label") or "").strip()
        escalation = (rec.get("escalation_label") or "").strip()
        problems: List[str] = []
        if not intent:
            problems.append("missing intent_label")
        elif intent not in taxonomy:
            problems.append(f"invalid intent_label '{intent}'")
        if not escalation:
            problems.append("missing escalation_label")
        elif escalation not in ESCALATION_VALUES:
            problems.append(f"invalid escalation_label '{escalation}'")
        if problems:
            issues.append(f"{gid}: " + "; ".join(problems))
        else:
            truth.append({
                "golden_id": gid,
                "intent_label": intent,
                "escalation_label": escalation,
            })
    if issues:
        details = "\n  - ".join(issues)
        raise EvaluationError(
            "Cannot run evaluation: the Golden Set is missing valid human labels.\n"
            f"  - {details}"
        )
    return truth


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def load_predictions(path: str) -> List[Dict[str, Any]]:
    """Load the agent's predictions file (a JSON list of objects)."""
    p = Path(path)
    if not p.exists():
        raise EvaluationError(
            "Cannot run evaluation: no predictions file at "
            f"{p}.\n"
            "Supply the agent's predictions via --predictions "
            "(e.g. evaluation/predictions.json)."
        )
    with open(p, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise EvaluationError("Predictions file must contain a JSON list.")
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise EvaluationError(f"Prediction #{i} is not a JSON object.")
        for field in PREDICTION_REQUIRED_FIELDS:
            if item.get(field) is None:
                raise EvaluationError(
                    f"Prediction #{i} is missing required field '{field}'."
                )
    return data


def validate_predictions(
    truth: List[Dict[str, str]],
    predictions: List[Dict[str, Any]],
    taxonomy: List[str],
) -> Tuple[List[Tuple[Dict[str, str], Dict[str, Any]]], Dict[str, Any]]:
    """Match predictions to ground truth and validate them.

    Matching is by ``golden_id``.  Returns ``(pairs, report)`` where ``pairs``
    is the list of ``(truth_row, prediction)`` tuples in Golden Set order.

    Validation semantics:
      - duplicates, predictions without a golden_id, invalid intent names, and
        invalid escalation labels are ERRORS -> ``EvaluationError`` is raised;
      - missing predictions (golden records with no prediction) and unexpected
        golden_ids (predictions that match no Golden record) are WARNINGS: the
        evaluation proceeds on the validated subset and the issues are included
        in the results/report.
    """
    truth_by_id = {t["golden_id"]: t for t in truth}
    seen: set[str] = set()
    matched: Dict[str, Tuple[Dict[str, str], Dict[str, Any]]] = {}

    errors: List[str] = []
    warnings = {
        "duplicates": [],
        "missing_golden_id": [],
        "invalid_intents": [],
        "invalid_escalations": [],
        "missing_predictions": [],
        "unexpected_golden_ids": [],
    }

    for idx, pred in enumerate(predictions, 1):
        gid = pred.get("golden_id")
        if gid is None:
            warnings["missing_golden_id"].append(idx)
            errors.append(f"prediction #{idx} has no golden_id")
            continue
        if gid not in truth_by_id:
            warnings["unexpected_golden_ids"].append(gid)
            continue
        if gid in seen:
            warnings["duplicates"].append(gid)
            errors.append(f"duplicate golden_id in predictions: {gid}")
            continue
        intent = pred.get("predicted_intent")
        escalation = pred.get("predicted_escalation")
        if intent not in taxonomy:
            errors.append(
                f"invalid predicted_intent '{intent}' for {gid} "
                f"(must be one of the {len(taxonomy)} taxonomy intents)"
            )
            continue
        if escalation not in ESCALATION_VALUES:
            errors.append(
                f"invalid predicted_escalation '{escalation}' for {gid} "
                f"(must be one of {ESCALATION_VALUES})"
            )
            continue
        seen.add(gid)
        matched[gid] = (truth_by_id[gid], pred)

    if errors:
        raise EvaluationError(
            "Predictions are invalid; evaluation cannot run.\n  - " + "\n  - ".join(errors)
        )

    for row in truth:
        if row["golden_id"] not in seen:
            warnings["missing_predictions"].append(row["golden_id"])

    # Pairs in Golden Set order for deterministic output.
    pairs = [matched[t["golden_id"]] for t in truth if t["golden_id"] in matched]
    return pairs, warnings


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _per_class_dicts(
    precision: Any,
    recall: Any,
    f1: Any,
    support: Any,
    labels: List[str],
) -> Dict[str, Dict[str, float]]:
    return {
        name: {
            "precision": round(float(precision[i]), 6),
            "recall": round(float(recall[i]), 6),
            "f1": round(float(f1[i]), 6),
            "support": int(support[i]),
        }
        for i, name in enumerate(labels)
    }


def compute_intent_metrics(
    pairs: List[Tuple[Dict[str, str], Dict[str, Any]]],
    taxonomy: List[str],
) -> Dict[str, Any]:
    """Intent classification metrics over the matched predictions."""
    y_true = [t["intent_label"] for t, _ in pairs]
    y_pred = [p["predicted_intent"] for _, p in pairs]

    accuracy = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=taxonomy)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=taxonomy, zero_division=0
    )

    return {
        "overall_accuracy": round(float(accuracy), 6),
        "macro_f1": round(float(macro), 6),
        "macro_precision": round(float(macro_precision), 6),
        "macro_recall": round(float(macro_recall), 6),
        "weighted_f1": round(float(weighted), 6),
        "weighted_precision": round(float(weighted_precision), 6),
        "weighted_recall": round(float(weighted_recall), 6),
        "per_intent": _per_class_dicts(prec, rec, f1, support, taxonomy),
        "confusion_matrix": {
            "labels": taxonomy,
            "matrix": [[int(v) for v in row] for row in cm.tolist()],
        },
    }


def compute_escalation_metrics(
    pairs: List[Tuple[Dict[str, str], Dict[str, Any]]],
) -> Dict[str, Any]:
    """Escalation decision metrics with ESCALATE_TO_HUMAN as positive class."""
    y_true = [t["escalation_label"] for t, _ in pairs]
    y_pred = [p["predicted_escalation"] for _, p in pairs]

    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=ESCALATION_VALUES)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=ESCALATION_VALUES, zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    positive = ESCALATION_POSITIVE_CLASS
    binary_precision = precision_score(
        y_true, y_pred, pos_label=positive, zero_division=0
    )
    binary_recall = recall_score(
        y_true, y_pred, pos_label=positive, zero_division=0
    )
    binary_f1 = f1_score(y_true, y_pred, pos_label=positive, zero_division=0)

    return {
        "accuracy": round(float(accuracy), 6),
        "macro_f1": round(float(macro_f1), 6),
        "positive_class": positive,
        "precision": round(float(binary_precision), 6),
        "recall": round(float(binary_recall), 6),
        "f1": round(float(binary_f1), 6),
        "per_class": _per_class_dicts(prec, rec, f1, support, ESCALATION_VALUES),
        "confusion_matrix": {
            "labels": ESCALATION_VALUES,
            "matrix": [[int(v) for v in row] for row in cm.tolist()],
        },
    }


def assemble_metrics(
    pairs: List[Tuple[Dict[str, str], Dict[str, Any]]],
    warnings: Dict[str, Any],
    taxonomy: List[str],
    golden_set_path: str,
    predictions_path: str,
    golden_total: int,
    labels_path: Optional[str],
) -> Dict[str, Any]:
    """Assemble the full machine-readable metrics document."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": EVALUATOR_NAME,
        "evaluated_at": _now_iso(),
        "golden_set": str(golden_set_path),
        "labels_file": labels_path,
        "predictions_file": str(predictions_path),
        "intent_taxonomy": taxonomy,
        "golden_records_total": golden_total,
        "examples_evaluated": len(pairs),
        "warnings": {
            "missing_predictions": warnings["missing_predictions"],
            "unexpected_golden_ids": warnings["unexpected_golden_ids"],
        },
        "intent_metrics": compute_intent_metrics(pairs, taxonomy),
        "escalation_metrics": compute_escalation_metrics(pairs),
        "notes": (
            "Ground truth comes only from human labels. No AI model was called "
            "by this evaluator and no performance values were fabricated."
        ),
    }


# ---------------------------------------------------------------------------
# Result writers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def _write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def write_confusion_matrix_csv(cm_data: Dict[str, Any], path: Path) -> Path:
    labels = cm_data["labels"]
    matrix = cm_data["matrix"]
    header = ["actual \\ predicted"] + labels
    rows = [
        [actual] + [str(int(cell)) for cell in row]
        for actual, row in zip(labels, matrix)
    ]
    return _write_csv(path, header, rows)


def write_per_intent_csv(per_intent: Dict[str, Dict[str, float]], path: Path) -> Path:
    header = ["intent", "precision", "recall", "f1", "support"]
    rows = [
        [name, m["precision"], m["recall"], m["f1"], m["support"]]
        for name, m in per_intent.items()
    ]
    return _write_csv(path, header, rows)


def write_results(metrics: Dict[str, Any], results_dir: str) -> List[Path]:
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    paths.append(_write_json(out_dir / "metrics.json", metrics))
    paths.append(write_confusion_matrix_csv(
        metrics["intent_metrics"]["confusion_matrix"],
        out_dir / "confusion_matrix.csv",
    ))
    paths.append(write_per_intent_csv(
        metrics["intent_metrics"]["per_intent"],
        out_dir / "per_intent_metrics.csv",
    ))
    paths.append(write_confusion_matrix_csv(
        metrics["escalation_metrics"]["confusion_matrix"],
        out_dir / "escalation_confusion_matrix.csv",
    ))
    return paths


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------

def _fmt(value: Any) -> str:
    return f"{float(value):.4f}"


def build_report(metrics: Dict[str, Any]) -> str:
    im = metrics["intent_metrics"]
    em = metrics["escalation_metrics"]
    lines: List[str] = []

    lines.append("# Golden Evaluation Set - Evaluation Report")
    lines.append("")
    lines.append(
        f"- Generated: {metrics['evaluated_at']}"
    )
    lines.append(f"- Golden Set: `{metrics['golden_set']}`")
    lines.append(f"- Predictions: `{metrics['predictions_file']}`")
    lines.append(f"- Records with ground truth: {metrics['golden_records_total']}")
    lines.append(f"- **Examples evaluated: {metrics['examples_evaluated']}**")
    if metrics.get("labels_file"):
        lines.append(f"- Human labels overlaid from: `{metrics['labels_file']}`")
    lines.append("")

    w = metrics["warnings"]
    if w["missing_predictions"]:
        lines.append(
            f"- Warning: no prediction for {len(w['missing_predictions'])} "
            "record(s): " + ", ".join(w["missing_predictions"][:20])
        )
    if w["unexpected_golden_ids"]:
        lines.append(
            f"- Warning: {len(w['unexpected_golden_ids'])} unexpected "
            "golden_id(s) ignored: " + ", ".join(w["unexpected_golden_ids"][:20])
        )
    if w["missing_predictions"] or w["unexpected_golden_ids"]:
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Examples evaluated | {metrics['examples_evaluated']} |")
    lines.append(f"| Intent overall accuracy | {_fmt(im['overall_accuracy'])} |")
    lines.append(f"| Intent macro F1 | {_fmt(im['macro_f1'])} |")
    lines.append(f"| Intent weighted F1 | {_fmt(im['weighted_f1'])} |")
    lines.append(f"| Escalation accuracy | {_fmt(em['accuracy'])} |")
    esc_key = em["positive_class"]
    lines.append(
        f"| Escalation precision ({esc_key} positive) | {_fmt(em['precision'])} |"
    )
    lines.append(
        f"| Escalation recall ({esc_key} positive) | {_fmt(em['recall'])} |"
    )
    lines.append(f"| Escalation F1 ({esc_key} positive) | {_fmt(em['f1'])} |")
    lines.append("")

    lines.append("## Intent metrics per class")
    lines.append("")
    lines.append("| Intent | Precision | Recall | F1 | Support |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name, m in im["per_intent"].items():
        lines.append(
            f"| {name} | {_fmt(m['precision'])} | {_fmt(m['recall'])} | "
            f"{_fmt(m['f1'])} | {m['support']} |"
        )
    lines.append("")

    lines.append("## Escalation metrics per class")
    lines.append("")
    lines.append("| Escalation | Precision | Recall | F1 | Support |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name, m in em["per_class"].items():
        lines.append(
            f"| {name} | {_fmt(m['precision'])} | {_fmt(m['recall'])} | "
            f"{_fmt(m['f1'])} | {m['support']} |"
        )
    lines.append(f"| macro | - | - | {_fmt(em['macro_f1'])} | - |")
    lines.append("")

    lines.append("## Intent confusion matrix (rows = actual, columns = predicted)")
    lines.append("")
    lines.append("```")
    cm = im["confusion_matrix"]
    labels = cm["labels"]
    header = "".join(f"{lbl:>28.28}" for lbl in labels)
    lines.append("actual \\ predicted" + "".join(f"{lbl:>28.28}" for lbl in labels))
    for actual, row in zip(labels, cm["matrix"]):
        cells = "".join(f"{int(c):>28}" for c in row)
        lines.append(f"{actual:>28.28}" + cells)
    lines.append("```")
    lines.append("")

    lines.append("## Escalation confusion matrix (rows = actual, columns = predicted)")
    lines.append("")
    lines.append("```")
    ecm = em["confusion_matrix"]
    elabels = ecm["labels"]
    for actual, row in zip(elabels, ecm["matrix"]):
        cells = "".join(f"{int(c):>28}" for c in row)
        lines.append(f"{actual:>28.28}" + cells)
    lines.append("```")
    lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "- Ground truth is the human-assigned `intent_label` and "
        "`escalation_label` in the Golden Set (no keyword heuristics)."
    )
    lines.append(
        "- Macro/weighted F1 are averaged over the intent classes observed in "
        "the evaluated subset (scikit-learn defaults)."
    )
    lines.append(
        "- Per-intent metrics and the confusion matrix cover the full "
        "10-intent taxonomy; `zero_division=0`."
    )
    lines.append(
        "- Escalation precision/recall/F1 use `ESCALATE_TO_HUMAN` as the "
        "positive class."
    )
    lines.append(
        "- This evaluator never calls an AI model and never fabricates labels "
        "or performance values: metrics only exist if human labels and agent "
        "predictions were both supplied."
    )
    lines.append("")
    return "\n".join(lines)


def write_report(metrics: Dict[str, Any], report_path: str) -> Path:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_report(metrics))
    return path


# ---------------------------------------------------------------------------
# Top-level flow
# ---------------------------------------------------------------------------

def run_evaluation(
    golden_set_path: str = str(DEFAULT_GOLDEN_SET_PATH),
    predictions_path: str = str(DEFAULT_PREDICTIONS_PATH),
    labels_path: Optional[str] = None,
    results_dir: str = str(DEFAULT_RESULTS_DIR),
    report_path: str = str(DEFAULT_REPORT_PATH),
    intents_yaml: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[Path]]:
    """Load ground truth + predictions, validate, compute, and write results.

    Raises ``EvaluationError`` when the evaluation cannot be performed
    (missing human labels, missing/invalid predictions).  On success writes
    ``metrics.json``, ``confusion_matrix.csv``, ``per_intent_metrics.csv``,
    ``escalation_confusion_matrix.csv`` and the markdown report, returning
    the metrics dict and the list of written paths.
    """
    taxonomy = load_taxonomy(intents_yaml)

    golden = Path(golden_set_path)
    if not golden.exists():
        raise EvaluationError(f"Golden Set not found: {golden}")
    with open(golden, "r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    records = copy.deepcopy(payload_records(payload))

    if labels_path:
        overlay_labels(records, labels_path)

    truth = extract_ground_truth(records, taxonomy)

    predictions = load_predictions(predictions_path)
    pairs, warnings = validate_predictions(truth, predictions, taxonomy)

    metrics = assemble_metrics(
        pairs=pairs,
        warnings=warnings,
        taxonomy=taxonomy,
        golden_set_path=str(golden),
        predictions_path=str(Path(predictions_path)),
        golden_total=len(payload_records(payload)),
        labels_path=labels_path,
    )

    written = write_results(metrics, results_dir)
    written.append(write_report(metrics, report_path))
    return metrics, written


def print_summary(metrics: Dict[str, Any]) -> None:
    im = metrics["intent_metrics"]
    em = metrics["escalation_metrics"]
    print(f"[evaluate] examples evaluated : {metrics['examples_evaluated']}")
    print(f"[evaluate] intent accuracy    : {im['overall_accuracy']:.4f}")
    print(f"[evaluate] intent macro F1    : {im['macro_f1']:.4f}")
    print(f"[evaluate] intent weighted F1 : {im['weighted_f1']:.4f}")
    print(f"[evaluate] escalation accuracy: {em['accuracy']:.4f}")
    print(
        f"[evaluate] escalation F1 ({em['positive_class']} positive): "
        f"{em['f1']:.4f}"
    )


def main(argv: Optional[List[str]] = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Evaluate the support agent against the human-labeled "
                    "Golden Evaluation Set."
    )
    parser.add_argument("--golden-set", default=str(DEFAULT_GOLDEN_SET_PATH))
    parser.add_argument(
        "--predictions", default=str(DEFAULT_PREDICTIONS_PATH),
        help="Path to the agent's predictions JSON (default: "
             "evaluation/predictions.json)",
    )
    parser.add_argument(
        "--labels-file", default=None,
        help="Optional labels store (evaluation/golden_set.labels.json); "
             "human labels are overlaid onto the Golden Set in memory.",
    )
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--intents-yaml", default=str(DEFAULT_INTENTS_YAML))
    args = parser.parse_args(argv)

    try:
        metrics, written = run_evaluation(
            golden_set_path=args.golden_set,
            predictions_path=args.predictions,
            labels_path=args.labels_file,
            results_dir=args.results_dir,
            report_path=args.report,
            intents_yaml=args.intents_yaml,
        )
    except EvaluationError as exc:
        print(f"[evaluate] Evaluation cannot run.\n{exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"[evaluate] Evaluation cannot run.\n{exc}", file=sys.stderr)
        return 1

    print_summary(metrics)
    print("[evaluate] results written:")
    for path in written:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())