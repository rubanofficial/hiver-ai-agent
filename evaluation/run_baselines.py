"""
Run the evaluation baselines against the human-labeled Golden Evaluation Set.

Wraps ``evaluation/baselines.py`` and scores every baseline with the shared
metric implementation from ``evaluation/evaluate.py``, writing per-baseline
artifacts to ``evaluation/baseline_results/``.

Baselines produced
------------------
* ``majority/``
    Majority Intent Baseline + Majority Escalation Baseline
* ``tfidf_logistic/``
    TF-IDF + Logistic Regression Baseline (intent + escalation)

Usage::

    python -m evaluation.run_baselines
    python -m evaluation.run_baselines --labels-file evaluation/golden_set.labels.json
    python -m evaluation.run_baselines --test-size 0.2 --seed 42

Requirements
------------
* The Golden Set must be FULLY human-labeled.  This script never generates
  labels and never runs a fake evaluation: if any record is unlabeled it
  fails with a clear message telling you to finish human labeling first.
* No Gemini / LLM is involved anywhere in this pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evaluation.baselines import (
    DEFAULT_TEST_SIZE,
    MAJORITY_ESCALATION_NAME,
    MAJORITY_INTENT_NAME,
    TFIDF_NAME,
    BaselineError,
    TfidfLogisticClassifier,
    evaluate_predictions,
    load_labeled_records,
    majority_classes,
    predict_majority,
    predict_with_classifiers,
    stratified_split,
)
from evaluation.evaluate import (
    DEFAULT_GOLDEN_SET_PATH,
    DEFAULT_INTENTS_YAML,
    EVALUATOR_NAME,
    EvaluationError,
    load_taxonomy,
)

DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent / "baseline_results"
LABELING_INSTRUCTION = (
    "Human labeling must be completed first: every Golden Set record needs a "
    "human intent_label and escalation_label. Use src.labels.help / the "
    "labeling workflow (evaluation/labeling_guide.md) before running baselines."
)

MAJORITY_DIR = "majority"
TFIDF_DIR = "tfidf_logistic"

MAJORITY_SOURCE_LABEL = f"{MAJORITY_INTENT_NAME} + {MAJORITY_ESCALATION_NAME}"
TFIDF_SOURCE_LABEL = TFIDF_NAME


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def evaluate_baseline(
    name: str,
    source_label: str,
    directory: str,
    predictions: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    taxonomy: List[str],
    *,
    golden_set_path: str,
    labels_path: Optional[str],
    golden_total: int,
    results_root: Path,
) -> Tuple[Dict[str, Any], List[Path]]:
    """Score one baseline and write its artifacts under ``results_root``."""
    out_dir = results_root / directory
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.json"
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    metrics, written = evaluate_predictions(
        predictions=predictions,
        test_records=test_records,
        taxonomy=taxonomy,
        golden_set_path=golden_set_path,
        labels_path=labels_path,
        golden_total=golden_total,
        predictions_path=str(predictions_path),
        results_dir=str(out_dir),
        report_path=str(out_dir / "evaluation_report.md"),
        source_label=source_label,
    )
    metrics["baseline_name"] = name
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return metrics, written


def run_baselines(
    golden_set_path: str = str(DEFAULT_GOLDEN_SET_PATH),
    labels_path: Optional[str] = None,
    intents_yaml: Optional[str] = None,
    results_root: str = str(DEFAULT_RESULTS_ROOT),
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = 42,
) -> Tuple[Dict[str, Any], List[Path]]:
    """Fit both baselines on the training split and evaluate on the test split.

    Returns ``(summary, written_paths)``.  Raises ``EvaluationError`` /
    ``BaselineError`` (e.g. incomplete human labeling) before anything is run.
    """
    records = load_labeled_records(golden_set_path, labels_path, intents_yaml)
    taxonomy = load_taxonomy(intents_yaml)
    train_records, test_records = stratified_split(
        records, test_size=test_size, random_state=random_state
    )
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)

    # --- Baseline 1: majority (intent + escalation) -------------------------
    majority_intent, majority_escalation = majority_classes(train_records)
    majority_preds = predict_majority(
        test_records, majority_intent, majority_escalation
    )
    majority_metrics, majority_written = evaluate_baseline(
        name=MAJORITY_SOURCE_LABEL,
        source_label=MAJORITY_SOURCE_LABEL,
        directory=MAJORITY_DIR,
        predictions=majority_preds,
        test_records=test_records,
        taxonomy=taxonomy,
        golden_set_path=golden_set_path,
        labels_path=labels_path,
        golden_total=len(records),
        results_root=root,
    )

    # --- Baseline 2: TF-IDF + Logistic Regression (intent + escalation) -----
    intent_clf = TfidfLogisticClassifier("intent_label", random_state=random_state)
    escalation_clf = TfidfLogisticClassifier(
        "escalation_label", random_state=random_state
    )
    intent_clf.fit(train_records)
    escalation_clf.fit(train_records)
    tfidf_preds = predict_with_classifiers(test_records, intent_clf, escalation_clf)
    tfidf_metrics, tfidf_written = evaluate_baseline(
        name=TFIDF_SOURCE_LABEL,
        source_label=TFIDF_SOURCE_LABEL,
        directory=TFIDF_DIR,
        predictions=tfidf_preds,
        test_records=test_records,
        taxonomy=taxonomy,
        golden_set_path=golden_set_path,
        labels_path=labels_path,
        golden_total=len(records),
        results_root=root,
    )

    baselines = [
        {
            "name": majority_metrics["baseline_name"],
            "directory": MAJORITY_DIR,
            "metrics_file": str(root / MAJORITY_DIR / "metrics.json"),
            "report_file": str(root / MAJORITY_DIR / "evaluation_report.md"),
            "intent_accuracy": majority_metrics["intent_metrics"]["overall_accuracy"],
            "intent_macro_f1": majority_metrics["intent_metrics"]["macro_f1"],
            "intent_weighted_f1": majority_metrics["intent_metrics"]["weighted_f1"],
            "escalation_accuracy": (
                majority_metrics["escalation_metrics"]["accuracy"]
            ),
            "escalation_f1": majority_metrics["escalation_metrics"]["f1"],
        },
        {
            "name": tfidf_metrics["baseline_name"],
            "directory": TFIDF_DIR,
            "metrics_file": str(root / TFIDF_DIR / "metrics.json"),
            "report_file": str(root / TFIDF_DIR / "evaluation_report.md"),
            "intent_accuracy": tfidf_metrics["intent_metrics"]["overall_accuracy"],
            "intent_macro_f1": tfidf_metrics["intent_metrics"]["macro_f1"],
            "intent_weighted_f1": tfidf_metrics["intent_metrics"]["weighted_f1"],
            "escalation_accuracy": tfidf_metrics["escalation_metrics"]["accuracy"],
            "escalation_f1": tfidf_metrics["escalation_metrics"]["f1"],
        },
    ]

    summary = {
        "schema_version": "1.0.0",
        "generator": "evaluation/run_baselines.py",
        "generated_at": _now_iso(),
        "evaluator": EVALUATOR_NAME,
        "golden_set": str(golden_set_path),
        "labels_file": labels_path,
        "split_method": (
            "single deterministic train/test split (fixed random_state), "
            "stratified by intent label where possible"
        ),
        "random_state": random_state,
        "test_size": test_size,
        "golden_records_total": len(records),
        "train_examples": len(train_records),
        "test_examples": len(test_records),
        "baselines": baselines,
    }
    with open(root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    (root / "summary.md").write_text(build_summary_markdown(summary), encoding="utf-8")

    written: List[Path] = list(root.glob("**/*"))
    return summary, written


def build_summary_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# Golden Evaluation Set - Baselines",
        "",
        f"- Generated: {summary['generated_at']}",
        f"- Split: {summary['split_method']} (seed {summary['random_state']}, "
        f"test size {summary['test_size']})",
        f"- Total labeled records: {summary['golden_records_total']}",
        f"- Train examples: {summary['train_examples']} / "
        f"Test examples: {summary['test_examples']}",
        "",
        "## Comparison",
        "",
        "| Baseline | Intent acc | Intent macro F1 | Intent weighted F1 | "
        "Escalation acc | Escalation F1 (ESCALATE_TO_HUMAN) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for b in summary["baselines"]:
        lines.append(
            f"| {b['name']} | {b['intent_accuracy']:.4f} | "
            f"{b['intent_macro_f1']:.4f} | {b['intent_weighted_f1']:.4f} | "
            f"{b['escalation_accuracy']:.4f} | {b['escalation_f1']:.4f} |"
        )
    lines.extend([
        "",
        "Details per baseline (metrics.json, confusion matrices, per-intent "
        "precision/recall/F1 and report) live in the directories above.",
        "",
    ])
    return "\n".join(lines)


def print_summary(summary: Dict[str, Any]) -> None:
    print(f"[baselines] split: seed={summary['random_state']} "
          f"test_size={summary['test_size']} "
          f"train={summary['train_examples']} test={summary['test_examples']}")
    for b in summary["baselines"]:
        print(f"[baselines] {b['name']}")
        print(f"  intent  acc={b['intent_accuracy']:.4f} "
              f"macro_f1={b['intent_macro_f1']:.4f} "
              f"weighted_f1={b['intent_weighted_f1']:.4f}")
        print(f"  escal.  acc={b['escalation_accuracy']:.4f} "
              f"f1={b['escalation_f1']:.4f}")


def main(argv: Optional[List[str]] = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Run the majority and TF-IDF+LogisticRegression baselines "
                    "against the human-labeled Golden Evaluation Set."
    )
    parser.add_argument("--golden-set", default=str(DEFAULT_GOLDEN_SET_PATH))
    parser.add_argument(
        "--labels-file", default=None,
        help="Optional labels store (evaluation/golden_set.labels.json); "
             "human labels are overlaid onto the Golden Set in memory.",
    )
    parser.add_argument("--intents-yaml", default=str(DEFAULT_INTENTS_YAML))
    parser.add_argument(
        "--results-root", default=str(DEFAULT_RESULTS_ROOT),
        help="Directory for baseline artifacts "
             "(default: evaluation/baseline_results).",
    )
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    try:
        summary, written = run_baselines(
            golden_set_path=args.golden_set,
            labels_path=args.labels_file,
            intents_yaml=args.intents_yaml,
            results_root=args.results_root,
            test_size=args.test_size,
            random_state=args.seed,
        )
    except EvaluationError as exc:
        print("[baselines] Cannot run baselines.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print(LABELING_INSTRUCTION, file=sys.stderr)
        return 1
    except BaselineError as exc:
        print("[baselines] Cannot run baselines.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"[baselines] Cannot run baselines.\n{exc}", file=sys.stderr)
        return 1

    print_summary(summary)
    print(f"[baselines] results written under {Path(args.results_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())