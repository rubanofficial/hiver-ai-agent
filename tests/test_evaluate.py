"""
Tests for the Golden Evaluation Set evaluator (evaluation/evaluate.py).

All tests use a tiny synthetic Golden Set, a tiny 3-intent taxonomy and
synthetic predictions written to a tmp dir.  No real data files and no AI
calls are involved.  Every metric value asserted here is hand-computed so
the tests are independent of any other implementation.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evaluation import evaluate as E


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

INTENTS = {
    "TT": "Technical Troubleshooting",
    "AL": "Account & Login",
    "BP": "Billing & Payments",
}
AH = "AUTO_HANDLE"
EH = "ESCALATE_TO_HUMAN"

TAXONOMY_YAML = (
    "intents:\n"
    "  - name: Technical Troubleshooting\n"
    "  - name: Account & Login\n"
    "  - name: Billing & Payments\n"
)


def write_taxonomy(tmp_path):
    path = tmp_path / "intents.yaml"
    path.write_text(TAXONOMY_YAML, encoding="utf-8")
    return str(path)


def _record(golden_id, conversation_id, intent="", escalation="", notes=""):
    return {
        "golden_id": golden_id,
        "conversation_id": conversation_id,
        "customer_message": "My laptop will not start after the update.",
        "intent_label": intent,
        "escalation_label": escalation,
        "notes": notes,
    }


def _labeled_records(n=3):
    return [_record(f"GOLDEN-{i:04d}", 100 + i) for i in range(1, n + 1)]


def write_golden_set(tmp_path, records):
    payload = {
        "schema_version": "1.0.0",
        "source": "synthetic",
        "count": len(records),
        "label_fields": ["intent_label", "escalation_label"],
        "records": records,
    }
    golden_path = tmp_path / "golden_set.json"
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    golden_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(golden_path), payload


def load_raw(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prediction(golden_id, intent, escalation, confidence=0.9):
    return {
        "golden_id": golden_id,
        "predicted_intent": intent,
        "predicted_intent_confidence": confidence,
        "predicted_escalation": escalation,
        "predicted_reply": "We can help with that.",
        "retrieved_evidence": ["doc-1"],
    }


def write_predictions(tmp_path, preds, name="predictions.json"):
    path = tmp_path / name
    path.write_text(json.dumps(preds, indent=2), encoding="utf-8")
    return str(path)


def multi_label(records, assignments):
    """Assign human labels onto the given records in place."""
    for rec, (intent, escalation) in zip(records, assignments):
        rec["intent_label"] = intent
        rec["escalation_label"] = escalation
    return records


@pytest.fixture
def taxonomy(tmp_path):
    return write_taxonomy(tmp_path)


@pytest.fixture
def populated(tmp_path):
    """Golden Set of 3 labeled records + 3-intent taxonomy + predictions dir.""" 
    records = multi_label(_labeled_records(3), [
        (INTENTS["TT"], AH), (INTENTS["AL"], EH), (INTENTS["BP"], AH),
    ])
    golden_path, _ = write_golden_set(tmp_path, records)
    taxonomy_path = write_taxonomy(tmp_path)
    return {
        "tmp_path": tmp_path,
        "golden_path": golden_path,
        "records": records,
        "taxonomy": taxonomy_path,
    }


# ---------------------------------------------------------------------------
# Loading / ground truth
# ---------------------------------------------------------------------------

class TestLoading:
    def test_load_taxonomy_reads_names(self, taxonomy):
        names = E.load_taxonomy(taxonomy)
        assert names == ["Technical Troubleshooting", "Account & Login",
                         "Billing & Payments"]

    def test_missing_taxonomy_raises(self, tmp_path):
        with pytest.raises(E.EvaluationError):
            E.load_taxonomy(str(tmp_path / "nope.yaml"))

    def test_payload_records_validates(self):
        with pytest.raises(E.EvaluationError):
            E.payload_records({"count": 0})

    def test_extract_ground_truth_accepts_all_labeled(self, populated, taxonomy):
        truth = E.extract_ground_truth(populated["records"], E.load_taxonomy(taxonomy))
        assert len(truth) == 3
        assert truth[0]["intent_label"] == INTENTS["TT"]
        assert truth[0]["escalation_label"] == AH

    def test_extract_ground_truth_rejects_unlabeled(self, tmp_path, taxonomy):
        golden_path, _ = write_golden_set(tmp_path, _labeled_records(2))
        payload = load_raw(golden_path)
        with pytest.raises(E.EvaluationError) as exc:
            E.extract_ground_truth(payload["records"], E.load_taxonomy(taxonomy))
        assert "missing intent_label" in str(exc.value)
        assert "missing escalation_label" in str(exc.value)
        assert "GOLDEN-0001" in str(exc.value)

    def test_extract_ground_truth_rejects_invalid_labels(self, tmp_path, taxonomy):
        rec = _record("GOLDEN-0001", 101, "Made Up Intent", "MAYBE")
        golden_path, _ = write_golden_set(tmp_path, [rec])
        payload = load_raw(golden_path)
        with pytest.raises(E.EvaluationError) as exc:
            E.extract_ground_truth(payload["records"], E.load_taxonomy(taxonomy))
        assert "invalid intent_label" in str(exc.value)
        assert "invalid escalation_label" in str(exc.value)

    def test_labels_file_overlay_fills_in_memory_only(self, populated):
        store = {"labels": {
            "GOLDEN-0001": {"golden_id": "GOLDEN-0001",
                            "intent_label": INTENTS["BP"], "escalation_label": AH},
        }}
        labels_path = populated["tmp_path"] / "labels.json"
        labels_path.write_text(json.dumps(store), encoding="utf-8")
        recs = [dict(r) for r in populated["records"]]
        recs[0]["intent_label"] = INTENTS["TT"]   # file value
        E.overlay_labels(recs, str(labels_path))
        assert recs[0]["intent_label"] == INTENTS["BP"]   # overlay wins in memory
        assert recs[1]["intent_label"] == INTENTS["AL"]   # untouched
        # The on-disk Golden Set is never modified by the overlay.
        assert load_raw(populated["golden_path"])["records"][0]["intent_label"] == \
            INTENTS["TT"]


# ---------------------------------------------------------------------------
# Predictions validation
# ---------------------------------------------------------------------------

class TestPredictionValidation:
    def _truth(self, populated):
        return E.extract_ground_truth(populated["records"],
                                      E.load_taxonomy(populated["taxonomy"]))

    def test_all_valid_pairs_in_golden_order(self, populated):
        taxonomy = E.load_taxonomy(populated["taxonomy"])
        truth = self._truth(populated)
        preds = [
            prediction("GOLDEN-0001", INTENTS["TT"], AH),
            prediction("GOLDEN-0002", INTENTS["AL"], EH),
            prediction("GOLDEN-0003", INTENTS["BP"], AH),
        ]
        pairs, warnings = E.validate_predictions(truth, preds, taxonomy)
        assert [p[0]["golden_id"] for p in pairs] == \
            ["GOLDEN-0001", "GOLDEN-0002", "GOLDEN-0003"]
        assert warnings["missing_predictions"] == []
        assert warnings["unexpected_golden_ids"] == []

    def test_missing_prediction_evaluates_subset(self, populated):
        taxonomy = E.load_taxonomy(populated["taxonomy"])
        truth = self._truth(populated)
        preds = [prediction("GOLDEN-0001", INTENTS["TT"], AH)]
        pairs, warnings = E.validate_predictions(truth, preds, taxonomy)
        assert len(pairs) == 1
        assert warnings["missing_predictions"] == ["GOLDEN-0002", "GOLDEN-0003"]
        assert warnings["unexpected_golden_ids"] == []

    def test_unexpected_golden_id_warned_and_ignored(self, populated):
        taxonomy = E.load_taxonomy(populated["taxonomy"])
        truth = self._truth(populated)
        preds = [
            prediction("GOLDEN-0001", INTENTS["TT"], AH),
            prediction("GOLDEN-9999", INTENTS["TT"], AH),
        ]
        pairs, warnings = E.validate_predictions(truth, preds, taxonomy)
        assert len(pairs) == 1
        assert warnings["unexpected_golden_ids"] == ["GOLDEN-9999"]

    def test_duplicate_golden_id_raises(self, populated):
        taxonomy = E.load_taxonomy(populated["taxonomy"])
        truth = self._truth(populated)
        preds = [
            prediction("GOLDEN-0001", INTENTS["TT"], AH),
            prediction("GOLDEN-0001", INTENTS["AL"], EH),
        ]
        with pytest.raises(E.EvaluationError) as exc:
            E.validate_predictions(truth, preds, taxonomy)
        assert "duplicate golden_id" in str(exc.value)

    def test_invalid_intent_raises(self, populated):
        taxonomy = E.load_taxonomy(populated["taxonomy"])
        truth = self._truth(populated)
        preds = [prediction("GOLDEN-0001", "Not An Intent", AH)]
        with pytest.raises(E.EvaluationError) as exc:
            E.validate_predictions(truth, preds, taxonomy)
        assert "invalid predicted_intent" in str(exc.value)

    def test_invalid_escalation_raises(self, populated):
        taxonomy = E.load_taxonomy(populated["taxonomy"])
        truth = self._truth(populated)
        preds = [prediction("GOLDEN-0001", INTENTS["TT"], "MAYBE")]
        with pytest.raises(E.EvaluationError) as exc:
            E.validate_predictions(truth, preds, taxonomy)
        assert "invalid predicted_escalation" in str(exc.value)

    def test_missing_golden_id_raises(self, populated):
        taxonomy = E.load_taxonomy(populated["taxonomy"])
        truth = self._truth(populated)
        preds = [{"predicted_intent": INTENTS["TT"], "predicted_escalation": AH}]
        with pytest.raises(E.EvaluationError) as exc:
            E.validate_predictions(truth, preds, taxonomy)
        assert "no golden_id" in str(exc.value)

    def test_missing_required_field_raises(self, populated):
        taxonomy = E.load_taxonomy(populated["taxonomy"])
        truth = self._truth(populated)
        bad = dict(prediction("GOLDEN-0001", INTENTS["TT"], AH))
        del bad["predicted_escalation"]
        with pytest.raises(E.EvaluationError) as exc:
            E.validate_predictions(truth, [bad], taxonomy)
        assert "missing required field" in str(exc.value)


# ---------------------------------------------------------------------------
# Metrics (hand-computed values)
# ---------------------------------------------------------------------------

class TestIntentMetrics:
    def test_perfect_predictions(self, populated):
        # GOLDEN-0001..0003 labeled: TT/AH, AL/EH, BP/AH
        taxonomy = E.load_taxonomy(populated["taxonomy"])
        truth = E.extract_ground_truth(populated["records"], taxonomy)
        preds = [
            prediction("GOLDEN-0001", INTENTS["TT"], AH),
            prediction("GOLDEN-0002", INTENTS["AL"], EH),
            prediction("GOLDEN-0003", INTENTS["BP"], AH),
        ]
        pairs, _ = E.validate_predictions(truth, preds, taxonomy)
        m = E.compute_intent_metrics(pairs, taxonomy)
        assert m["overall_accuracy"] == 1.0
        assert m["macro_f1"] == 1.0
        assert m["weighted_f1"] == 1.0
        assert m["per_intent"][INTENTS["TT"]]["precision"] == 1.0
        assert m["confusion_matrix"]["matrix"][0][0] == 1
        # Off-diagonal all zero.
        flat = [c for row in m["confusion_matrix"]["matrix"] for c in row]
        assert sum(flat) == 3
        assert m["confusion_matrix"]["matrix"][0][0] + \
            m["confusion_matrix"]["matrix"][1][1] + \
            m["confusion_matrix"]["matrix"][2][2] == 3

    def test_known_hand_computed_case(self, populated):
        """4 examples, 3 classes; all values verified by hand."""
        # synthetic golden set of 4 records
        recs = multi_label(_labeled_records(4), [
            (INTENTS["TT"], AH), (INTENTS["TT"], EH),
            (INTENTS["AL"], AH), (INTENTS["AL"], EH),
        ])
        golden_path, _ = write_golden_set(populated["tmp_path"] / "set2", recs)
        _ = golden_path
        taxonomy = E.load_taxonomy(populated["taxonomy"])
        truth = E.extract_ground_truth(recs, taxonomy)
        preds = [
            prediction("GOLDEN-0001", INTENTS["TT"], AH),
            prediction("GOLDEN-0002", INTENTS["AL"], EH),
            prediction("GOLDEN-0003", INTENTS["AL"], EH),
            prediction("GOLDEN-0004", INTENTS["BP"], EH),
        ]
        pairs, _ = E.validate_predictions(truth, preds, taxonomy)
        m = E.compute_intent_metrics(pairs, taxonomy)
        # accuracy 2/4, macro f1 (2/3+1/2+0)/3, weighted f1 7/12
        assert m["overall_accuracy"] == 0.5
        assert m["macro_f1"] == pytest.approx(0.388889, abs=1e-5)
        assert m["weighted_f1"] == pytest.approx(0.583333, abs=1e-5)
        assert m["macro_precision"] == pytest.approx(0.5, abs=1e-5)
        assert m["macro_recall"] == pytest.approx(0.333333, abs=1e-5)
        per = m["per_intent"]
        assert per[INTENTS["TT"]]["precision"] == 1.0
        assert per[INTENTS["TT"]]["recall"] == pytest.approx(0.5, abs=1e-5)
        assert per[INTENTS["TT"]]["f1"] == pytest.approx(0.666667, abs=1e-5)
        assert per[INTENTS["TT"]]["support"] == 2
        assert per[INTENTS["AL"]]["precision"] == 0.5
        assert per[INTENTS["AL"]]["recall"] == 0.5
        assert per[INTENTS["BP"]]["precision"] == 0.0
        assert per[INTENTS["BP"]]["support"] == 0
        cm = m["confusion_matrix"]
        assert cm["matrix"][0] == [1, 1, 0]
        assert cm["matrix"][1] == [0, 1, 1]
        assert cm["matrix"][2] == [0, 0, 0]

    def test_zero_division_empty_predicted_class(self, populated):
        taxonomy = E.load_taxonomy(populated["taxonomy"])
        truth = E.extract_ground_truth(populated["records"], taxonomy)
        # Predict TT for everything: AL and BP have no predicted instances,
        # so their precision must fall back to 0.0 (zero_division) instead of
        # raising, and recall is 0.0 because nothing was caught.
        preds = [
            prediction("GOLDEN-0001", INTENTS["TT"], AH),
            prediction("GOLDEN-0002", INTENTS["TT"], AH),
            prediction("GOLDEN-0003", INTENTS["TT"], AH),
        ]
        pairs, _ = E.validate_predictions(truth, preds, taxonomy)
        m = E.compute_intent_metrics(pairs, taxonomy)
        # TT: predicted 3 times, 1 of which is correct -> precision 1/3,
        # recall 1/1.  AL/BP: never predicted -> precision falls back to 0.0
        # (zero_division) instead of raising; recall is 0.0.
        assert m["per_intent"][INTENTS["TT"]]["precision"] == \
            pytest.approx(1 / 3, abs=1e-5)
        assert m["per_intent"][INTENTS["TT"]]["recall"] == 1.0
        assert m["per_intent"][INTENTS["AL"]]["precision"] == 0.0
        assert m["per_intent"][INTENTS["AL"]]["recall"] == 0.0
        assert m["per_intent"][INTENTS["BP"]]["recall"] == 0.0
        assert m["per_intent"][INTENTS["AL"]]["support"] == 1


class TestEscalationMetrics:
    def test_perfect_predictions(self, populated):
        taxonomy = E.load_taxonomy(populated["taxonomy"])
        truth = E.extract_ground_truth(populated["records"], taxonomy)
        preds = [
            prediction("GOLDEN-0001", INTENTS["TT"], AH),
            prediction("GOLDEN-0002", INTENTS["AL"], EH),
            prediction("GOLDEN-0003", INTENTS["BP"], AH),
        ]
        pairs, _ = E.validate_predictions(truth, preds, taxonomy)
        m = E.compute_escalation_metrics(pairs)
        assert m["accuracy"] == 1.0
        assert m["positive_class"] == EH
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["macro_f1"] == 1.0

    def test_hand_computed_case(self, populated):
        recs = multi_label(_labeled_records(4), [
            (INTENTS["TT"], AH), (INTENTS["TT"], EH),
            (INTENTS["AL"], AH), (INTENTS["AL"], EH),
        ])
        taxonomy = E.load_taxonomy(populated["taxonomy"])
        truth = E.extract_ground_truth(recs, taxonomy)
        preds = [
            prediction("GOLDEN-0001", INTENTS["TT"], AH),
            prediction("GOLDEN-0002", INTENTS["TT"], EH),
            prediction("GOLDEN-0003", INTENTS["AL"], EH),
            prediction("GOLDEN-0004", INTENTS["AL"], EH),
        ]
        pairs, _ = E.validate_predictions(truth, preds, taxonomy)
        m = E.compute_escalation_metrics(pairs)
        # cm rows=[AH,EH]: [AH:[1,1], EH:[0,2]] -> accuracy 3/4, EH
        # positive: p=2/3, r=1.0, f1=0.8
        assert m["accuracy"] == 0.75
        assert m["precision"] == pytest.approx(2 / 3, abs=1e-5)
        assert m["recall"] == 1.0
        assert m["f1"] == 0.8
        assert m["per_class"][AH]["recall"] == 0.5
        assert m["per_class"][EH]["precision"] == pytest.approx(2 / 3, abs=1e-5)
        assert m["confusion_matrix"]["matrix"] == [[1, 1], [0, 2]]


# ---------------------------------------------------------------------------
# End-to-end behaviour
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_missing_predictions_file_blocks_run(self, populated):
        with pytest.raises(E.EvaluationError) as exc:
            E.run_evaluation(
                golden_set_path=populated["golden_path"],
                predictions_path=str(populated["tmp_path"] / "nope.json"),
                intents_yaml=populated["taxonomy"],
            )
        assert "no predictions file" in str(exc.value)

    def test_unlabeled_golden_set_blocks_run(self, tmp_path, populated):
        golden_path, _ = write_golden_set(tmp_path, _labeled_records(2))
        preds = [prediction("GOLDEN-0001", INTENTS["TT"], AH),
                 prediction("GOLDEN-0002", INTENTS["AL"], EH)]
        pred_path = write_predictions(tmp_path, preds)
        with pytest.raises(E.EvaluationError) as exc:
            E.run_evaluation(
                golden_set_path=golden_path,
                predictions_path=pred_path,
                intents_yaml=populated["taxonomy"],
            )
        assert "missing valid human labels" in str(exc.value)

    def test_successful_run_writes_all_artifacts(self, populated):
        pred_path = write_predictions(populated["tmp_path"], [
            prediction("GOLDEN-0001", INTENTS["TT"], AH),
            prediction("GOLDEN-0002", INTENTS["AL"], EH),
            prediction("GOLDEN-0003", INTENTS["BP"], AH),
        ])
        out = populated["tmp_path"] / "out"
        metrics, written = E.run_evaluation(
            golden_set_path=populated["golden_path"],
            predictions_path=pred_path,
            results_dir=str(out),
            report_path=str(out / "evaluation_report.md"),
            intents_yaml=populated["taxonomy"],
        )
        assert metrics["golden_records_total"] == 3
        assert metrics["examples_evaluated"] == 3
        assert metrics["intent_metrics"]["overall_accuracy"] == 1.0
        names = [str(p) for p in written]
        expected = {
            out / "metrics.json", out / "confusion_matrix.csv",
            out / "per_intent_metrics.csv",
            out / "escalation_confusion_matrix.csv",
            out / "evaluation_report.md",
        }
        assert expected <= set(names)
        report = (out / "evaluation_report.md").read_text(encoding="utf-8")
        assert "Examples evaluated: 3" in report
        assert "1.0000" in report

    def test_run_with_labels_file_overlay(self, populated):
        # A Golden Set with EMPTY labels plus a separate labels store: the
        # store must supply the ground truth used for evaluation.
        empty_golden, _ = write_golden_set(
            populated["tmp_path"] / "empty", _labeled_records(3))
        store = {"labels": {
            "GOLDEN-0001": {"golden_id": "GOLDEN-0001",
                            "intent_label": INTENTS["TT"], "escalation_label": AH},
            "GOLDEN-0002": {"golden_id": "GOLDEN-0002",
                            "intent_label": INTENTS["AL"], "escalation_label": EH},
            "GOLDEN-0003": {"golden_id": "GOLDEN-0003",
                            "intent_label": INTENTS["BP"], "escalation_label": AH},
        }}
        labels_path = populated["tmp_path"] / "golden_set.labels.json"
        labels_path.write_text(json.dumps(store), encoding="utf-8")
        pred_path = write_predictions(populated["tmp_path"], [
            prediction("GOLDEN-0001", INTENTS["TT"], AH),
            prediction("GOLDEN-0002", INTENTS["AL"], EH),
            prediction("GOLDEN-0003", INTENTS["BP"], AH),
        ])
        metrics, _ = E.run_evaluation(
            golden_set_path=empty_golden,
            predictions_path=pred_path,
            labels_path=str(labels_path),
            results_dir=str(populated["tmp_path"] / "out2"),
            report_path=str(populated["tmp_path"] / "out2" / "r.md"),
            intents_yaml=populated["taxonomy"],
        )
        assert metrics["examples_evaluated"] == 3
        assert metrics["intent_metrics"]["overall_accuracy"] == 1.0

    def test_missing_predictions_counted_in_examples(self, populated):
        pred_path = write_predictions(populated["tmp_path"], [
            prediction("GOLDEN-0001", INTENTS["TT"], AH),
        ])
        metrics, _ = E.run_evaluation(
            golden_set_path=populated["golden_path"],
            predictions_path=pred_path,
            results_dir=str(populated["tmp_path"] / "out3"),
            report_path=str(populated["tmp_path"] / "out3" / "r.md"),
            intents_yaml=populated["taxonomy"],
        )
        assert metrics["examples_evaluated"] == 1
        assert metrics["warnings"]["missing_predictions"] == \
            ["GOLDEN-0002", "GOLDEN-0003"]

    def test_run_does_not_modify_golden_set_or_predictions(self, populated):
        pred_path = write_predictions(populated["tmp_path"], [
            prediction("GOLDEN-0001", INTENTS["TT"], AH),
            prediction("GOLDEN-0002", INTENTS["AL"], EH),
            prediction("GOLDEN-0003", INTENTS["BP"], AH),
        ])
        before_golden = Path(populated["golden_path"]).read_bytes()
        before_pred = Path(pred_path).read_bytes()
        E.run_evaluation(
            golden_set_path=populated["golden_path"],
            predictions_path=pred_path,
            results_dir=str(populated["tmp_path"] / "out4"),
            report_path=str(populated["tmp_path"] / "out4" / "r.md"),
            intents_yaml=populated["taxonomy"],
        )
        assert Path(populated["golden_path"]).read_bytes() == before_golden
        assert Path(pred_path).read_bytes() == before_pred

    def test_per_intent_csv_has_one_row_per_taxonomy_intent(self, populated):
        pred_path = write_predictions(populated["tmp_path"], [
            prediction("GOLDEN-0001", INTENTS["TT"], AH),
            prediction("GOLDEN-0002", INTENTS["AL"], EH),
            prediction("GOLDEN-0003", INTENTS["BP"], AH),
        ])
        out = populated["tmp_path"] / "out5"
        E.run_evaluation(
            golden_set_path=populated["golden_path"],
            predictions_path=pred_path,
            results_dir=str(out),
            report_path=str(out / "r.md"),
            intents_yaml=populated["taxonomy"],
        )
        lines = (out / "per_intent_metrics.csv").read_text(
            encoding="utf-8").strip().splitlines()
        assert len(lines) == 4  # header + 3 taxonomy intents
        cm = (out / "confusion_matrix.csv").read_text(
            encoding="utf-8").strip().splitlines()
        assert len(cm) == 4  # header + 3 rows
        esc = (out / "escalation_confusion_matrix.csv").read_text(
            encoding="utf-8").strip().splitlines()
        assert len(esc) == 3  # header + 2 escalation rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCli:
    def test_cli_runs_end_to_end(self, populated):
        pred_path = write_predictions(populated["tmp_path"], [
            prediction("GOLDEN-0001", INTENTS["TT"], AH),
            prediction("GOLDEN-0002", INTENTS["AL"], EH),
        ])
        out = populated["tmp_path"] / "cli"
        result = subprocess.run(
            [sys.executable, "-m", "evaluation.evaluate",
             "--golden-set", populated["golden_path"],
             "--predictions", pred_path,
             "--intents-yaml", populated["taxonomy"],
             "--results-dir", str(out),
             "--report", str(out / "evaluation_report.md")],
            cwd=str(E.PROJECT_ROOT),
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "examples evaluated : 2" in result.stdout
        assert (out / "metrics.json").exists()
        assert "Examples evaluated: 2" in \
            (out / "evaluation_report.md").read_text(encoding="utf-8")

    def test_cli_without_predictions_fails_cleanly(self, populated):
        result = subprocess.run(
            [sys.executable, "-m", "evaluation.evaluate",
             "--golden-set", populated["golden_path"],
             "--intents-yaml", populated["taxonomy"],
             "--predictions", str(populated["tmp_path"] / "nope.json"),
             "--results-dir", str(populated["tmp_path"] / "cdn"),
             "--report", str(populated["tmp_path"] / "cdn" / "r.md")],
            cwd=str(E.PROJECT_ROOT),
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 1
        assert "no predictions file" in result.stderr


# ---------------------------------------------------------------------------
# No fabrication / no AI guarantees
# ---------------------------------------------------------------------------

class TestNoFabrication:
    def test_evaluator_module_has_no_model_imports(self):
        source = Path(E.__file__).read_text(encoding="utf-8")
        for token in ("google.generativeai", "genai", "openai", "anthropic"):
            assert token not in source

    def test_evaluator_module_has_no_random_logic(self):
        source = Path(E.__file__).read_text(encoding="utf-8")
        for token in ("random.", "random(", "shuffle", ".sample("):
            assert token not in source

    def test_nothing_written_when_blocked(self, populated):
        # A blocking condition (here: missing predictions file) must write
        # no result files at all.
        out = populated["tmp_path"] / "nothing"
        with pytest.raises(E.EvaluationError):
            E.run_evaluation(
                golden_set_path=populated["golden_path"],
                predictions_path=str(populated["tmp_path"] / "absent.json"),
                results_dir=str(out),
                report_path=str(out / "r.md"),
                intents_yaml=populated["taxonomy"],
            )
        assert not out.exists()