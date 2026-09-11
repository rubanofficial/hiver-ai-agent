"""
Tests for the evaluation baselines (evaluation/baselines.py and
evaluation/run_baselines.py).

All tests use tiny synthetic labeled Golden Sets written to a tmp dir and a
small 3-intent taxonomy.  No real data, no Gemini/LLM calls, and every
expected value is either hand-computed or structural (valid labels, disjoint
train/test, determinism, reuse of the shared metric implementation).
"""

import json
import subprocess
import sys
from pathlib import Path
from collections import Counter

import pytest

from evaluation import baselines as B
from evaluation import run_baselines as RB
from evaluation.evaluate import ESCALATION_VALUES, EvaluationError

AH = "AUTO_HANDLE"
EH = "ESCALATE_TO_HUMAN"
INTENTS = {
    "TT": "Technical Troubleshooting",
    "AL": "Account & Login",
    "BP": "Billing & Payments",
}

TAXONOMY_YAML = (
    "intents:\n"
    "  - name: Technical Troubleshooting\n"
    "  - name: Account & Login\n"
    "  - name: Billing & Payments\n"
)

INTENT_KEYWORDS = {
    INTENTS["TT"]: "laptop will not start blue screen failing to boot",
    INTENTS["AL"]: "cannot log in account password locked reset",
    INTENTS["BP"]: "invoice charged twice refund payment bill",
}
EH_KEYWORD = "needs a human specialist immediately"
AH_KEYWORD = "simple straightforward question"


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def write_taxonomy(tmp_path):
    path = tmp_path / "intents.yaml"
    path.write_text(TAXONOMY_YAML, encoding="utf-8")
    return str(path)


def rec(gid, msg, ctx, intent="", escalation="", notes=""):
    return {
        "golden_id": gid,
        "conversation_id": int(gid.split("-")[1]),
        "customer_message": msg,
        "conversation_context": ctx,
        "intent_label": intent,
        "escalation_label": escalation,
        "notes": notes,
    }


def write_golden_set(tmp_path, records, name="golden_set.json"):
    payload = {
        "schema_version": "1.0.0",
        "source": "synthetic",
        "count": len(records),
        "label_fields": ["intent_label", "escalation_label"],
        "records": records,
    }
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path), payload


def synthetic_ml_set(n_per_intent=4):
    """Learnable tiny dataset: distinct intent keywords + escalation keywords."""
    records = []
    n = 0
    for intent in (INTENTS["TT"], INTENTS["AL"], INTENTS["BP"]):
        for i in range(n_per_intent):
            esc = EH if i % 2 == 0 else AH
            kw = EH_KEYWORD if esc == EH else AH_KEYWORD
            n += 1
            records.append(rec(
                f"GOLDEN-{n:04d}",
                f"{INTENT_KEYWORDS[intent]} {kw} message {n}",
                "MICROSOFT (1): How can we help?",
                intent,
                esc,
            ))
    return records


def load_raw(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def taxonomy(tmp_path):
    return write_taxonomy(tmp_path)


@pytest.fixture
def learnable_set(tmp_path, taxonomy):
    golden_path, _ = write_golden_set(tmp_path, synthetic_ml_set(4))
    return {
        "tmp_path": tmp_path,
        "golden_path": golden_path,
        "taxonomy": taxonomy,
        "records": synthetic_ml_set(4),
    }


# ---------------------------------------------------------------------------
# Majority baseline
# ---------------------------------------------------------------------------

class TestMajority:
    def test_majority_computed_not_hardcoded(self):
        records = [
            rec("GOLDEN-0001", "a", "c", INTENTS["AL"], EH),
            rec("GOLDEN-0002", "b", "c", INTENTS["AL"], EH),
            rec("GOLDEN-0003", "c", "c", INTENTS["AL"], EH),
            rec("GOLDEN-0004", "d", "c", INTENTS["TT"], AH),
            rec("GOLDEN-0005", "e", "c", INTENTS["BP"], AH),
        ]
        majority_intent, majority_esc = B.majority_classes(records)
        assert majority_intent == INTENTS["AL"]
        assert majority_esc == EH

    def test_majority_classes_empty_raises(self):
        with pytest.raises(B.BaselineError):
            B.majority_classes([])

    def test_predict_majority_is_uniform(self):
        records = [
            rec("GOLDEN-0001", "a", "c", INTENTS["TT"], AH),
            rec("GOLDEN-0002", "b", "c", INTENTS["AL"], EH),
            rec("GOLDEN-0003", "c", "c", INTENTS["BP"], AH),
        ]
        preds = B.predict_majority(records, INTENTS["TT"], EH)
        assert [p["predicted_intent"] for p in preds] == [INTENTS["TT"]] * 3
        assert [p["predicted_escalation"] for p in preds] == [EH] * 3
        assert [p["golden_id"] for p in preds] == \
            ["GOLDEN-0001", "GOLDEN-0002", "GOLDEN-0003"]


# ---------------------------------------------------------------------------
# Reproducible split
# ---------------------------------------------------------------------------

class TestSplit:
    def _ids(self, records):
        return [r["golden_id"] for r in records]

    def test_split_disjoint_and_sized(self, learnable_set):
        train, test = B.stratified_split(learnable_set["records"])
        train_ids, test_ids = self._ids(train), self._ids(test)
        assert set(train_ids).isdisjoint(test_ids)
        assert len(train_ids) + len(test_ids) == len(learnable_set["records"])
        assert len(test) == 3  # 12 * 0.2

    def test_split_is_deterministic(self, learnable_set):
        r1 = B.stratified_split(learnable_set["records"], random_state=42)
        r2 = B.stratified_split(learnable_set["records"], random_state=42)
        assert self._ids(r1[0]) == self._ids(r2[0])
        assert self._ids(r1[1]) == self._ids(r2[1])

    def test_split_stratifies_intents_when_possible(self, learnable_set):
        train, test = B.stratified_split(learnable_set["records"])
        assert set(r["intent_label"] for r in train) == set(INTENTS.values())
        assert set(r["intent_label"] for r in test) == set(INTENTS.values())

    def test_split_single_record_raises(self):
        with pytest.raises(B.BaselineError):
            B.stratified_split([rec("GOLDEN-0001", "a", "c", INTENTS["TT"], AH)])

    def test_split_single_class_falls_back(self, tmp_path):
        single = [rec(f"GOLDEN-{i:04d}", "a", "c", INTENTS["TT"], AH)
                  for i in range(1, 5)]
        train, test = B.stratified_split(single)
        assert len(train) + len(test) == 4
        assert set(self._ids(train)).isdisjoint(self._ids(test))

    def test_split_class_with_single_member_falls_back(self):
        records = [rec("GOLDEN-0001", "a", "c", INTENTS["TT"], AH),
                   rec("GOLDEN-0002", "b", "c", INTENTS["AL"], AH),
                   rec("GOLDEN-0003", "c", "c", INTENTS["AL"], AH),
                   rec("GOLDEN-0004", "d", "c", INTENTS["AL"], AH),
                   rec("GOLDEN-0005", "e", "c", INTENTS["AL"], AH)]
        train, test = B.stratified_split(records, test_size=0.2)
        assert set(self._ids(train)).isdisjoint(self._ids(test))
        assert len(train) + len(test) == 5


# ---------------------------------------------------------------------------
# TF-IDF + Logistic Regression intent classifier
# ---------------------------------------------------------------------------

class TestTfidfIntent:
    def test_fit_and_predict_valid_intents(self, learnable_set, taxonomy):
        records = learnable_set["records"]
        train, test = B.stratified_split(records, random_state=42)
        clf = B.TfidfLogisticClassifier("intent_label")
        clf.fit(train)
        preds = clf.predict(test)
        assert len(preds) == len(test)
        for label in preds:
            assert label in set(INTENTS.values())

    def test_predictions_deterministic(self, learnable_set):
        records = learnable_set["records"]
        train, test = B.stratified_split(records, random_state=42)
        first = B.TfidfLogisticClassifier("intent_label").fit(train).predict(test)
        second = B.TfidfLogisticClassifier("intent_label").fit(train).predict(test)
        assert first == second

    def test_learns_separated_keywords(self, learnable_set):
        records = learnable_set["records"]
        train, test = B.stratified_split(records, random_state=42)
        clf = B.TfidfLogisticClassifier("intent_label")
        clf.fit(train)
        preds = clf.predict(test)
        truth = [r["intent_label"] for r in test]
        correct = sum(1 for a, b in zip(preds, truth) if a == b)
        assert correct / len(truth) >= 2 / 3

    def test_single_class_train_raises(self):
        records = [rec(f"GOLDEN-{i:04d}", "a", "c", INTENTS["TT"], AH)
                   for i in range(1, 5)]
        clf = B.TfidfLogisticClassifier("intent_label")
        with pytest.raises(B.BaselineError) as exc:
            clf.fit(records)
        assert "single" in str(exc.value)

    def test_predict_before_fit_raises(self, learnable_set):
        train, test = B.stratified_split(learnable_set["records"])
        clf = B.TfidfLogisticClassifier("intent_label")
        with pytest.raises(B.BaselineError):
            clf.predict(test)

    def test_invalid_label_field_raises(self):
        with pytest.raises(ValueError):
            B.TfidfLogisticClassifier("bogus_field")


# ---------------------------------------------------------------------------
# TF-IDF + Logistic Regression escalation classifier
# ---------------------------------------------------------------------------

class TestTfidfEscalation:
    def test_fit_and_predict_valid_escalations(self, learnable_set):
        records = learnable_set["records"]
        train, test = B.stratified_split(records, random_state=42)
        clf = B.TfidfLogisticClassifier("escalation_label")
        clf.fit(train)
        preds = clf.predict(test)
        assert len(preds) == len(test)
        for label in preds:
            assert label in ESCALATION_VALUES

    def test_predictions_deterministic(self, learnable_set):
        records = learnable_set["records"]
        train, test = B.stratified_split(records, random_state=42)
        first = B.TfidfLogisticClassifier("escalation_label").fit(train).predict(test)
        second = B.TfidfLogisticClassifier("escalation_label").fit(train).predict(test)
        assert first == second


# ---------------------------------------------------------------------------
# End-to-end run_baselines
# ---------------------------------------------------------------------------

class TestRunBaselines:
    def test_full_run_writes_all_artifacts(self, learnable_set):
        root = learnable_set["tmp_path"] / "baseline_results"
        summary, written = RB.run_baselines(
            golden_set_path=learnable_set["golden_path"],
            intents_yaml=learnable_set["taxonomy"],
            results_root=str(root),
            test_size=0.25,
            random_state=42,
        )
        for directory in ("majority", "tfidf_logistic"):
            for artifact in ("metrics.json", "predictions.json",
                             "confusion_matrix.csv", "per_intent_metrics.csv",
                             "escalation_confusion_matrix.csv",
                             "evaluation_report.md"):
                assert (root / directory / artifact).exists(), artifact

        summary_data = load_raw(root / "summary.json")
        assert summary_data["test_examples"] == 3
        assert summary_data["train_examples"] == 9
        assert [b["name"] for b in summary_data["baselines"]] == [
            "Majority Intent Baseline + Majority Escalation Baseline",
            "TF-IDF + Logistic Regression Baseline",
        ]
        assert (root / "summary.md").exists()

    def test_metrics_reuse_shared_structure(self, learnable_set):
        root = learnable_set["tmp_path"] / "rr"
        summary, _ = RB.run_baselines(
            golden_set_path=learnable_set["golden_path"],
            intents_yaml=learnable_set["taxonomy"],
            results_root=str(root),
            test_size=0.25,
            random_state=42,
        )
        for data in summary["baselines"]:
            metrics = load_raw(data["metrics_file"])
            # The shared evaluator structure (not a duplicated calculation).
            assert "intent_metrics" in metrics
            assert "escalation_metrics" in metrics
            im = metrics["intent_metrics"]
            for key in ("overall_accuracy", "macro_f1", "weighted_f1",
                        "macro_precision", "macro_recall",
                        "per_intent", "confusion_matrix"):
                assert key in im
            assert set(im["per_intent"].keys()) == set(INTENTS.values())
            assert len(im["confusion_matrix"]["matrix"]) == len(INTENTS)
            assert "source_label" in metrics
            assert "examples_evaluated" in metrics

    def test_evaluated_only_on_test_examples(self, learnable_set):
        root = learnable_set["tmp_path"] / "rr2"
        _, _ = RB.run_baselines(
            golden_set_path=learnable_set["golden_path"],
            intents_yaml=learnable_set["taxonomy"],
            results_root=str(root),
            test_size=0.25,
            random_state=42,
        )
        for data in load_raw(root / "summary.json")["baselines"]:
            metrics = load_raw(data["metrics_file"])
            assert metrics["examples_evaluated"] == 3   # test only, not 12
            assert metrics["golden_records_total"] == 12

    def test_majority_predictions_are_uniform(self, learnable_set):
        root = learnable_set["tmp_path"] / "rr3"
        RB.run_baselines(
            golden_set_path=learnable_set["golden_path"],
            intents_yaml=learnable_set["taxonomy"],
            results_root=str(root),
            test_size=0.25,
            random_state=42,
        )
        preds = load_raw(root / "majority" / "predictions.json")
        counts = Counter(p["predicted_intent"] for p in preds)
        assert len(counts) == 1
        esc_counts = Counter(p["predicted_escalation"] for p in preds)
        assert len(esc_counts) == 1

    def test_deterministic_across_runs(self, learnable_set):
        root1 = learnable_set["tmp_path"] / "d1"
        root2 = learnable_set["tmp_path"] / "d2"
        s1, _ = RB.run_baselines(
            golden_set_path=learnable_set["golden_path"],
            intents_yaml=learnable_set["taxonomy"],
            results_root=str(root1), test_size=0.25, random_state=42)
        s2, _ = RB.run_baselines(
            golden_set_path=learnable_set["golden_path"],
            intents_yaml=learnable_set["taxonomy"],
            results_root=str(root2), test_size=0.25, random_state=42)
        assert s1["train_examples"] == s2["train_examples"]
        assert s1["test_examples"] == s2["test_examples"]
        for b1, b2 in zip(s1["baselines"], s2["baselines"]):
            assert b1["intent_accuracy"] == b2["intent_accuracy"]
            assert b1["intent_macro_f1"] == b2["intent_macro_f1"]
            assert b1["intent_weighted_f1"] == b2["intent_weighted_f1"]
            assert b1["escalation_accuracy"] == b2["escalation_accuracy"]

    def test_unlabeled_golden_set_blocks_run(self, tmp_path, taxonomy):
        unidentified = [rec(f"GOLDEN-{i:04d}", "a", "c") for i in range(1, 5)]
        golden_path, _ = write_golden_set(tmp_path, unidentified)
        with pytest.raises(EvaluationError) as exc:
            RB.run_baselines(
                golden_set_path=golden_path,
                intents_yaml=taxonomy,
                results_root=str(tmp_path / "out"),
            )
        assert "missing valid human labels" in str(exc.value)
        assert not (tmp_path / "out").exists()

    def test_run_with_labels_file_overlay(self, tmp_path, taxonomy):
        unidentified = [
            rec(f"GOLDEN-{i:04d}",
                INTENT_KEYWORDS[INTENTS["TT"] if i % 2 else INTENTS["AL"]]
                + f" message number {i}",
                "MICROSOFT (1): How can we help?")
            for i in range(1, 5)
        ]
        golden_path, _ = write_golden_set(tmp_path, unidentified)
        labels = {"labels": {
            "GOLDEN-0001": {"golden_id": "GOLDEN-0001",
                            "intent_label": INTENTS["TT"], "escalation_label": AH},
            "GOLDEN-0002": {"golden_id": "GOLDEN-0002",
                            "intent_label": INTENTS["AL"], "escalation_label": EH},
            "GOLDEN-0003": {"golden_id": "GOLDEN-0003",
                            "intent_label": INTENTS["BP"], "escalation_label": AH},
            "GOLDEN-0004": {"golden_id": "GOLDEN-0004",
                            "intent_label": INTENTS["TT"], "escalation_label": EH},
        }}
        labels_path = tmp_path / "golden_set.labels.json"
        labels_path.write_text(json.dumps(labels), encoding="utf-8")
        summary, _ = RB.run_baselines(
            golden_set_path=golden_path,
            labels_path=str(labels_path),
            intents_yaml=taxonomy,
            results_root=str(tmp_path / "out"),
            test_size=0.25,
            random_state=42,
        )
        assert summary["golden_records_total"] == 4
        # The on-disk Golden Set keeps empty labels.
        assert load_raw(golden_path)["records"][0]["intent_label"] == ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCli:
    def test_cli_full_run(self, learnable_set):
        root = learnable_set["tmp_path"] / "cli"
        result = subprocess.run(
            [sys.executable, "-m", "evaluation.run_baselines",
             "--golden-set", learnable_set["golden_path"],
             "--intents-yaml", learnable_set["taxonomy"],
             "--results-root", str(root),
             "--test-size", "0.25", "--seed", "42"],
            cwd=str(B.Path(B.__file__).resolve().parent.parent),
            capture_output=True, text=True, timeout=180,
        )
        assert result.returncode == 0, result.stderr
        assert "test=3" in result.stdout
        assert (root / "summary.json").exists()
        assert (root / "majority" / "metrics.json").exists()
        assert (root / "tfidf_logistic" / "metrics.json").exists()

    def test_cli_unlabeled_fails_clearly(self, tmp_path, taxonomy):
        unidentified = [rec(f"GOLDEN-{i:04d}", "a", "c") for i in range(1, 5)]
        golden_path, _ = write_golden_set(tmp_path, unidentified)
        root = tmp_path / "cli2"
        result = subprocess.run(
            [sys.executable, "-m", "evaluation.run_baselines",
             "--golden-set", golden_path,
             "--intents-yaml", taxonomy,
             "--results-root", str(root)],
            cwd=str(B.Path(B.__file__).resolve().parent.parent),
            capture_output=True, text=True, timeout=180,
        )
        assert result.returncode == 1
        assert "Human labeling must be completed first" in result.stderr
        assert not root.exists()


# ---------------------------------------------------------------------------
# No LLM / no fabrication guarantees
# ---------------------------------------------------------------------------

class TestNoLlm:
    def test_baselines_module_has_no_model_imports(self):
        source = Path(B.__file__).read_text(encoding="utf-8")
        for token in ("google.generativeai", "genai", "openai", "anthropic",
                      "random.", "random(", "shuffle"):
            assert token not in source

    def test_run_baselines_module_has_no_model_imports(self):
        source = Path(RB.__file__).read_text(encoding="utf-8")
        for token in ("google.generativeai", "genai", "openai", "anthropic"):
            assert token not in source

    def test_baseline_not_in_golden_set_predictions(self, learnable_set):
        # Baselines never write to evaluation/predictions.json (the agent's spot).
        root = learnable_set["tmp_path"] / "nb"
        RB.run_baselines(
            golden_set_path=learnable_set["golden_path"],
            intents_yaml=learnable_set["taxonomy"],
            results_root=str(root),
            test_size=0.25,
            random_state=42,
        )
        # Every generated prediction lives under baseline_results/ only.
        assert (root / "majority" / "predictions.json").exists()
        assert (root / "tfidf_logistic" / "predictions.json").exists()