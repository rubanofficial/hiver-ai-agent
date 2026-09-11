"""
Tests for the LLM-as-a-Judge framework (evaluation/judge.py and
evaluation/run_judge.py).

Gemini is MOCKED everywhere: either a FakeGeminiClient is injected, or a fake
google.generativeai module is substituted (for the real-suite test that
verifies temperature=0 and JSON mime).  No real Gemini API calls ever happen,
and no GEMINI_API_KEY is required.
"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from evaluation import judge as J
from evaluation import run_judge as RJ

AH = "AUTO_HANDLE"
EH = "ESCALATE_TO_HUMAN"

VALID_RESULT = {
    "correctness": 5,
    "helpfulness": 5,
    "relevance": 4,
    "clarity": 4,
    "escalation_appropriateness": 5,
    "overall_reply_score": 5,
    "evidence_support": 4,
    "unsupported_claims": 5,
    "evidence_relevance": 4,
    "grounding_score": 4,
    "reason": "Clear, professional reply that stays grounded.",
}


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def prediction(gid, intent="Technical Troubleshooting", escalation=AH, reply="We can help.",
               evidence=None):
    return {
        "golden_id": gid,
        "predicted_intent": intent,
        "predicted_intent_confidence": 0.9,
        "predicted_escalation": escalation,
        "predicted_reply": reply,
        "retrieved_evidence": evidence if evidence is not None else [
            {"doc_id": "d1", "text": "Try a clean boot after the update."},
        ],
    }


def record(gid, msg="My laptop will not start after the update.",
           ctx="MICROSOFT (1): How can we help?", intent="", escalation=""):
    return {
        "golden_id": gid,
        "conversation_id": int(gid.split("-")[1]),
        "customer_message": msg,
        "conversation_context": ctx,
        "intent_label": intent,
        "escalation_label": escalation,
        "notes": "",
    }


def write_golden_set(tmp_path, records):
    payload = {
        "schema_version": "1.0.0",
        "source": "synthetic",
        "count": len(records),
        "label_fields": ["intent_label", "escalation_label"],
        "records": records,
    }
    path = tmp_path / "golden_set.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def write_predictions(tmp_path, preds):
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(preds, indent=2), encoding="utf-8")
    return str(path)


def write_taxonomy(tmp_path):
    path = tmp_path / "intents.yaml"
    path.write_text(
        "intents:\n"
        "  - name: Technical Troubleshooting\n"
        "  - name: Account & Login\n"
        "  - name: Billing & Payments\n",
        encoding="utf-8",
    )
    return str(path)


class FakeGeminiClient:
    """Injected stand-in for Gemini; records calls, returns canned JSON."""

    def __init__(self, response_texts=None):
        self.calls = 0
        self.prompts = []
        self.model_name = "fake-judge-model"
        self._queue = list(response_texts or [])

    def generate(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        if self._queue:
            return self._queue.pop(0)
        return json.dumps(VALID_RESULT)


def _judged_result(gid, result=None):
    result = json.loads(json.dumps(result or VALID_RESULT))
    enriched = J.enrich_result(result)
    enriched["golden_id"] = gid
    enriched["model"] = "fake-judge-model"
    enriched["input"] = {
        "golden_id": gid,
        "customer_message": "My laptop will not start after the update.",
        "conversation_context": "MICROSOFT (1): How can we help?",
        "predicted_intent": "Technical Troubleshooting",
        "retrieved_evidence": "doc: try a clean boot.",
        "generated_reply": "We can help.",
    }
    return enriched


@pytest.fixture
def judge_examples(tmp_path):
    """Golden Set + predictions for 3 examples + taxonomy."""
    evidence = [{"doc_id": "d1", "text": "Try a clean boot after the update."}]
    preds = [
        prediction("GOLDEN-0001", evidence=evidence),
        prediction("GOLDEN-0002", intent="Account & Login",
                   escalation=EH, reply="Reset your account password."),
        prediction("GOLDEN-0003", intent="Billing & Payments",
                   reply="We can issue a refund request."),
    ]
    examples = [
        J.build_judge_input(record("GOLDEN-0001"), preds[0]),
        J.build_judge_input(record("GOLDEN-0002"), preds[1]),
        J.build_judge_input(record("GOLDEN-0003"), preds[2]),
    ]
    return {
        "tmp_path": tmp_path,
        "examples": examples,
        "records": [record("GOLDEN-0001"), record("GOLDEN-0002"),
                    record("GOLDEN-0003")],
        "preds": preds,
        "golden_path": write_golden_set(tmp_path,
                                        [record("GOLDEN-0001"),
                                         record("GOLDEN-0002"),
                                         record("GOLDEN-0003")]),
        "pred_path": write_predictions(tmp_path, preds),
        "taxonomy": write_taxonomy(tmp_path),
    }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class TestParsing:
    def test_parse_valid_json(self):
        assert J.parse_judge_json('{"a": 1}') == {"a": 1}

    def test_parse_json_inside_code_fence(self):
        text = "```json\n{\"correctness\": 5}\n```"
        assert J.parse_judge_json(text) == {"correctness": 5}

    def test_parse_empty_raises(self):
        with pytest.raises(J.JudgeError):
            J.parse_judge_json("")

    def test_parse_invalid_json_raises(self):
        with pytest.raises(J.JudgeError):
            J.parse_judge_json("{not json")

    def test_parse_non_object_raises(self):
        with pytest.raises(J.JudgeError):
            J.parse_judge_json("[1, 2, 3]")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_payload_passes(self):
        result = J.parse_and_validate(json.dumps(VALID_RESULT))
        assert result["overall_reply_score"] == 5

    def test_missing_field_raises(self):
        bad = dict(VALID_RESULT)
        del bad["reason"]
        with pytest.raises(J.JudgeError) as exc:
            J.parse_and_validate(json.dumps(bad))
        assert "reason" in str(exc.value)

    def test_missing_score_field_raises(self):
        bad = dict(VALID_RESULT)
        del bad["helpfulness"]
        with pytest.raises(J.JudgeError) as exc:
            J.parse_and_validate(json.dumps(bad))
        assert "helpfulness" in str(exc.value)

    @pytest.mark.parametrize("field,value", [
        ("correctness", 0), ("correctness", 6), ("correctness", -1),
        ("correctness", 4.5), ("correctness", "5"), ("correctness", True),
        ("grounding_score", 1.0), ("evidence_support", 7),
    ])
    def test_invalid_score_values_raise(self, field, value):
        bad = dict(VALID_RESULT)
        bad[field] = value
        with pytest.raises(J.JudgeError) as exc:
            J.parse_and_validate(json.dumps(bad))
        assert field in str(exc.value)

    def test_empty_reason_raises(self):
        bad = dict(VALID_RESULT)
        bad["reason"] = "   "
        with pytest.raises(J.JudgeError):
            J.parse_and_validate(json.dumps(bad))

    def test_enrich_derives_fields(self):
        result = J.enrich_result(dict(VALID_RESULT))
        assert result["overall_score"] == 5
        assert result["short_reason"] == VALID_RESULT["reason"]
        assert result["grounding_decision"] == J.SUPPORTED
        assert set(result["reply_quality"]) == set(J.REPLY_SCORE_FIELDS)
        assert set(result["evidence_grounding"]) == set(J.GROUNDING_SCORE_FIELDS)

    def test_grounding_decision_threshold(self):
        assert J.grounding_decision(3) == J.SUPPORTED
        assert J.grounding_decision(5) == J.SUPPORTED
        assert J.grounding_decision(2) == J.UNSUPPORTED
        assert J.grounding_decision(1) == J.UNSUPPORTED


# ---------------------------------------------------------------------------
# Judge input construction / no leakage
# ---------------------------------------------------------------------------

class TestJudgeInput:
    def test_input_contains_only_allowed_fields(self, judge_examples):
        for example in judge_examples["examples"]:
            assert set(example.keys()) == set(J.JUDGE_INPUT_FIELDS)
            assert "intent_label" not in example
            assert "escalation_label" not in example
            assert "notes" not in example

    def test_human_labels_never_reach_prompt(self):
        # Golden record carries (simulated) human labels with secret markers.
        labeled = record("GOLDEN-0001", intent="ZZZ_SECRET_INTENT",
                         escalation="ESCALATE_TO_HUMAN")
        judge_input = J.build_judge_input(
            labeled, prediction("GOLDEN-0001")
        )
        assert "ZZZ_SECRET_INTENT" not in json.dumps(judge_input)
        prompt = J.build_judge_prompt(judge_input)
        assert "ZZZ_SECRET_INTENT" not in prompt
        assert "intent_label" not in prompt
        assert "escalation_label" not in prompt
        # Even the golden_id is not sent to the model (only kept locally).
        assert "GOLDEN-0001" not in prompt
        assert "We can help." in prompt

    def test_missing_reply_raises(self):
        pred = prediction("GOLDEN-0001")
        pred["predicted_reply"] = ""
        with pytest.raises(J.JudgeError):
            J.build_judge_input(record("GOLDEN-0001"), pred)

    def test_render_evidence_handles_types(self):
        assert J.render_evidence([{"a": 1}, "plain"]) == \
            '{"a": 1}\nplain'
        assert J.render_evidence("plain") == "plain"
        assert J.render_evidence(None) == ""

    def test_load_judge_examples_ordered_no_labels(self, judge_examples):
        examples = J.load_judge_examples(
            judge_examples["golden_path"], judge_examples["pred_path"],
            intents_yaml=judge_examples["taxonomy"],
        )
        assert [e["golden_id"] for e in examples] == \
            ["GOLDEN-0001", "GOLDEN-0002", "GOLDEN-0003"]
        for example in examples:
            assert set(example.keys()) == set(J.JUDGE_INPUT_FIELDS)
            assert example["retrieved_evidence"] != ""

    def test_load_judge_examples_ignores_labels_file_contents(self, judge_examples,
                                                              tmp_path):
        # Even if a labels file is passed, judgment input excludes it entirely.
        labels_path = tmp_path / "labels.json"
        labels_path.write_text(json.dumps({"labels": {
            "GOLDEN-0001": {"intent_label": "ZZZ_SECRET",
                            "escalation_label": "ESCALATE_TO_HUMAN"}}}),
            encoding="utf-8")
        examples = J.load_judge_examples(
            judge_examples["golden_path"], judge_examples["pred_path"],
            labels_path=str(labels_path),
            intents_yaml=judge_examples["taxonomy"],
        )
        assert "ZZZ_SECRET" not in json.dumps(examples)

    def test_missing_prediction_raises(self, tmp_path, judge_examples):
        golden_path = write_golden_set(tmp_path, [record("GOLDEN-0099")])
        with pytest.raises(J.JudgeError) as exc:
            J.load_judge_examples(golden_path, judge_examples["pred_path"],
                                  intents_yaml=judge_examples["taxonomy"])
        assert "no agent prediction" in str(exc.value)

    def test_unknown_prediction_id_raises(self, judge_examples):
        preds = [
            prediction("GOLDEN-0001"),
            prediction("GOLDEN-0002", intent="Account & Login",
                       escalation=EH, reply="Reset your password."),
            prediction("GOLDEN-0003", intent="Billing & Payments",
                       reply="We can issue a refund."),
            prediction("GOLDEN-9999"),
        ]
        pred_path = write_predictions(judge_examples["tmp_path"], preds)
        with pytest.raises(J.JudgeError) as exc:
            J.load_judge_examples(
                judge_examples["golden_path"], pred_path,
                intents_yaml=judge_examples["taxonomy"])
        assert "GOLDEN-9999" in str(exc.value)


# ---------------------------------------------------------------------------
# Gemini client construction (fully mocked)
# ---------------------------------------------------------------------------

class FakeGenai:
    class types:
        class GenerationConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

    def __init__(self):
        self.configured_key = None
        self.models = []

    def configure(self, api_key=None):
        self.configured_key = api_key

    def GenerativeModel(self, name):
        model = FakeModel(name)
        self.models.append(model)
        return model


class FakeModel:
    def __init__(self, name):
        self.name = name
        self.last_prompt = None
        self.last_config = None

    def generate_content(self, prompt, generation_config=None):
        self.last_prompt = prompt
        self.last_config = generation_config
        return FakeResponse(json.dumps(VALID_RESULT))


class FakeResponse:
    def __init__(self, text):
        self.text = text


class TestGeminiClient:
    def test_client_uses_json_mime_and_zero_temperature(self):
        fake_genai = FakeGenai()
        client = J.GeminiJudgeClient(
            model_name="test-model", api_key="k", genai_module=fake_genai
        )
        raw = client.generate("hello")
        assert fake_genai.configured_key == "k"
        model = fake_genai.models[-1]
        assert model.name == "test-model"
        assert model.last_prompt == "hello"
        config = model.last_config
        assert config.kwargs["temperature"] == 0.0
        assert config.kwargs["response_mime_type"] == "application/json"
        assert raw == json.dumps(VALID_RESULT)

    def test_client_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        fake = FakeGenai()
        with pytest.raises(J.JudgeError):
            J.GeminiJudgeClient(model_name="m", api_key=None,
                                genai_module=fake)

    def test_client_model_defaults_to_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL", "env-model")
        fake = FakeGenai()
        client = J.GeminiJudgeClient(api_key="k", genai_module=fake)
        assert client.model_name == "env-model"

    def test_client_never_writes_api_key(self, judge_examples):
        fake_genai = FakeGenai()
        client = J.GeminiJudgeClient(model_name="m", api_key="secret-key",
                                     genai_module=fake_genai)
        results, _ = J.run_judgments(client, judge_examples["examples"][:1],
                                     cache_dir=str(judge_examples["tmp_path"] / "c"))
        document = RJ.build_results_document(results, "m", {"fresh": 1, "cached": 0})
        text = json.dumps(document)
        assert "secret-key" not in text
        assert "api_key" not in text


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestCaching:
    def test_fingerprint_stable_and_sensitive(self, judge_examples):
        example = judge_examples["examples"][0]
        assert J.input_fingerprint(example) == J.input_fingerprint(example)
        changed = dict(example, generated_reply="different reply")
        assert J.input_fingerprint(changed) != J.input_fingerprint(example)

    def test_cache_round_trip(self, judge_examples, tmp_path):
        example = judge_examples["examples"][0]
        J.write_cache_entry(str(tmp_path), "GOLDEN-0001", example,
                            "raw-json", _judged_result("GOLDEN-0001"))
        cache = J.load_cache(str(tmp_path))
        assert J.is_cached(cache, example)
        other = dict(example, generated_reply="changed")
        assert not J.is_cached(cache, other)

    def test_run_judgments_caching_prevents_recalls(self, judge_examples, tmp_path):
        client = FakeGeminiClient()
        cache_dir = str(tmp_path / "cache")
        first, stats1 = J.run_judgments(
            client, judge_examples["examples"], cache_dir=cache_dir)
        assert stats1 == {"examples": 3, "fresh": 3, "cached": 0}
        calls_after_first = client.calls

        second, stats2 = J.run_judgments(
            client, judge_examples["examples"], cache_dir=cache_dir)
        assert stats2 == {"examples": 3, "fresh": 0, "cached": 3}
        assert client.calls == calls_after_first  # no new Gemini calls
        assert [r["golden_id"] for r in first] == ["GOLDEN-0001", "GOLDEN-0002",
                                                    "GOLDEN-0003"]
        assert second[0]["overall_reply_score"] == first[0]["overall_reply_score"]

    def test_cache_invalidation_when_input_changes(self, judge_examples, tmp_path):
        cache_dir = str(tmp_path / "cache")
        client = FakeGeminiClient()
        _, stats1 = J.run_judgments(client, judge_examples["examples"],
                                    cache_dir=cache_dir)
        assert stats1["fresh"] == 3
        # Change the reply for example 1 -> fingerprint mismatch -> recalc.
        changed = [dict(judge_examples["examples"][0], generated_reply="NEW reply")]
        _, stats2 = J.run_judgments(client, changed, cache_dir=cache_dir)
        assert stats2 == {"examples": 1, "fresh": 1, "cached": 0}


# ---------------------------------------------------------------------------
# Human review + agreement
# ---------------------------------------------------------------------------

class TestHumanReview:
    def test_write_human_review_csv(self, judge_examples, tmp_path):
        client = FakeGeminiClient()
        results, _ = J.run_judgments(
            client, judge_examples["examples"], cache_dir=None)
        path = J.write_human_review_csv(str(tmp_path / "review.csv"), results)
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        expect_empty = ["human_grounding", "human_notes"]
        for row in rows:
            for col in expect_empty:
                assert row[col] == ""   # never auto-created
            assert row["llm_grounding_decision"] in J.GROUNDING_LABELS
            assert row["golden_id"] in {"GOLDEN-0001", "GOLDEN-0002", "GOLDEN-0003"}
            assert row["reason"] != ""

    def test_read_human_grounding_only_labeled(self, tmp_path):
        path = tmp_path / "review.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=J.HUMAN_REVIEW_COLUMNS)
            writer.writeheader()
            writer.writerow({col: "" for col in J.HUMAN_REVIEW_COLUMNS})
            row = {col: "" for col in J.HUMAN_REVIEW_COLUMNS}
            row["golden_id"] = "GOLDEN-0001"
            row["human_grounding"] = "SUPPORTED"
            writer.writerow(row)
        labeled = J.read_human_grounding_csv(str(path))
        assert labeled == [("GOLDEN-0001", "SUPPORTED")]

    def test_read_human_grounding_invalid_label_raises(self, tmp_path):
        path = tmp_path / "review.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=J.HUMAN_REVIEW_COLUMNS)
            writer.writeheader()
            row = {col: "" for col in J.HUMAN_REVIEW_COLUMNS}
            row["golden_id"] = "GOLDEN-0001"
            row["human_grounding"] = "MAYBE"
            writer.writerow(row)
        with pytest.raises(J.JudgeError) as exc:
            J.read_human_grounding_csv(str(path))
        assert "MAYBE" in str(exc.value)


class TestAgreement:
    def test_kappa_and_raw_agreement_hand_computed(self):
        agreement = J.compute_agreement(
            ["SUPPORTED", "SUPPORTED", "UNSUPPORTED", "UNSUPPORTED"],
            ["SUPPORTED", "UNSUPPORTED", "SUPPORTED", "UNSUPPORTED"],
        )
        assert agreement["n"] == 4
        assert agreement["raw_agreement"] == 0.5
        assert agreement["cohen_kappa"] == 0.0

    def test_perfect_agreement(self):
        agreement = J.compute_agreement(
            ["SUPPORTED", "UNSUPPORTED", "SUPPORTED", "UNSUPPORTED"],
            ["SUPPORTED", "UNSUPPORTED", "SUPPORTED", "UNSUPPORTED"],
        )
        assert agreement["raw_agreement"] == 1.0
        assert agreement["cohen_kappa"] == 1.0

    def test_single_label_annotator_kappa_undefined(self):
        agreement = J.compute_agreement(
            ["SUPPORTED", "SUPPORTED", "SUPPORTED"],
            ["SUPPORTED", "SUPPORTED", "SUPPORTED"],
        )
        assert agreement["raw_agreement"] == 1.0
        assert agreement["cohen_kappa"] is None
        assert agreement["cohen_kappa_note"] is not None

    def test_mismatched_lengths_raise(self):
        with pytest.raises(J.JudgeError):
            J.compute_agreement(["SUPPORTED"], ["SUPPORTED", "UNSUPPORTED"])

    def test_agreement_from_human_review_integration(self, judge_examples, tmp_path):
        results = [_judged_result("GOLDEN-0001"),
                   _judged_result("GOLDEN-0002"),
                   _judged_result("GOLDEN-0003")]
        review = J.write_human_review_csv(str(tmp_path / "review.csv"), results)
        mapping = {"GOLDEN-0001": "SUPPORTED", "GOLDEN-0002": "UNSUPPORTED"}
        with open(review, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if row["golden_id"] in mapping:
                row["human_grounding"] = mapping[row["golden_id"]]
        with open(review, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=J.HUMAN_REVIEW_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        agreement = J.agreement_from_human_review(
            results, str(review))
        assert agreement["human_reviewed_examples"] == 2
        assert agreement["n"] == 2
        assert "raw_agreement" in agreement and "cohen_kappa" in agreement

    def test_agreement_rejects_gid_without_result(self, tmp_path):
        results = [_judged_result("GOLDEN-0001")]
        review = J.write_human_review_csv(str(tmp_path / "review.csv"), results)
        with open(review, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        rows.append({col: "" for col in J.HUMAN_REVIEW_COLUMNS} | {
            "golden_id": "GOLDEN-4242", "human_grounding": "SUPPORTED"})
        with open(review, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=J.HUMAN_REVIEW_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        with pytest.raises(J.JudgeError):
            J.agreement_from_human_review(results, str(review))


# ---------------------------------------------------------------------------
# run_judge / CLI
# ---------------------------------------------------------------------------

class TestRunJudge:
    def test_run_judge_writes_artifacts(self, judge_examples, tmp_path):
        client = FakeGeminiClient()
        outcome = RJ.run_judge(
            predictions_path=judge_examples["pred_path"],
            golden_set_path=judge_examples["golden_path"],
            intents_yaml=judge_examples["taxonomy"],
            model="fake-judge-model",
            limit=3,
            cache_dir=str(tmp_path / "cache"),
            results_dir=str(tmp_path / "results"),
            human_review_path=str(tmp_path / "review.csv"),
            client=client,
        )
        assert outcome["stats"]["fresh"] == 3
        assert (tmp_path / "results" / "results.json").exists()
        assert (tmp_path / "review.csv").exists()
        doc = json.loads((tmp_path / "results" / "results.json").read_text(
            encoding="utf-8"))
        assert doc["count"] == 3
        assert all("correctness" in r for r in doc["results"])
        assert all("intent_label" not in json.dumps(r) for r in doc["results"])

    def test_run_judge_serves_from_cache(self, judge_examples, tmp_path):
        client = FakeGeminiClient()
        common = dict(
            predictions_path=judge_examples["pred_path"],
            golden_set_path=judge_examples["golden_path"],
            intents_yaml=judge_examples["taxonomy"],
            model="fake-judge-model",
            limit=2,
            cache_dir=str(tmp_path / "cache"),
            results_dir=str(tmp_path / "r1"),
            human_review_path=str(tmp_path / "h1.csv"),
            client=client,
        )
        first = RJ.run_judge(**common)
        assert first["stats"]["fresh"] == 2
        second = RJ.run_judge(**{**common, "results_dir": str(tmp_path / "r2"),
                                 "human_review_path": str(tmp_path / "h2.csv")})
        assert second["stats"]["fresh"] == 0
        assert second["stats"]["cached"] == 2

    def test_select_examples_limit_and_ids(self, judge_examples):
        examples = judge_examples["examples"]
        assert [e["golden_id"] for e in
                RJ.select_examples(examples, limit=2, golden_ids=[], all_examples=False)] \
            == ["GOLDEN-0001", "GOLDEN-0002"]
        assert [e["golden_id"] for e in
                RJ.select_examples(examples, limit=0, golden_ids=[], all_examples=False)] \
            == ["GOLDEN-0001", "GOLDEN-0002", "GOLDEN-0003"]
        assert [e["golden_id"] for e in
                RJ.select_examples(examples, limit=5, golden_ids=["GOLDEN-0003"],
                                   all_examples=False)] == ["GOLDEN-0003"]
        with pytest.raises(J.JudgeError):
            RJ.select_examples(examples, limit=5, golden_ids=["NOPE"],
                               all_examples=False)

    def test_cli_dry_run_no_api_key_needed(self, judge_examples, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "evaluation.run_judge",
             "--predictions", judge_examples["pred_path"],
             "--golden-set", judge_examples["golden_path"],
             "--intents-yaml", judge_examples["taxonomy"],
             "--dry-run", "--limit", "2",
             "--cache-dir", str(tmp_path / "cache"),
             "--results-dir", str(tmp_path / "res"),
             "--human-review", str(tmp_path / "review.csv")],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True, text=True, timeout=120,
            env={k: v for k, v in os.environ.items()
                 if k != "GEMINI_API_KEY"},
        )
        assert result.returncode == 0, result.stderr
        assert "NEW CALL" in result.stdout
        assert "planned Gemini calls: 2" in result.stdout

    def test_cli_requires_gemini_api_key(self, judge_examples, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "evaluation.run_judge",
             "--predictions", judge_examples["pred_path"],
             "--golden-set", judge_examples["golden_path"],
             "--intents-yaml", judge_examples["taxonomy"],
             "--limit", "1",
             "--results-dir", str(tmp_path / "res"),
             "--human-review", str(tmp_path / "review.csv")],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True, text=True, timeout=120,
            env={k: v for k, v in os.environ.items()
                 if k != "GEMINI_API_KEY"},
        )
        assert result.returncode == 1
        assert "GEMINI_API_KEY is not set" in result.stderr

    def test_cli_unknown_golden_id_fails(self, judge_examples, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "evaluation.run_judge",
             "--predictions", judge_examples["pred_path"],
             "--golden-set", judge_examples["golden_path"],
             "--intents-yaml", judge_examples["taxonomy"],
             "--golden-ids", "GOLDEN-9999",
             "--results-dir", str(tmp_path / "res"),
             "--human-review", str(tmp_path / "review.csv")],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 1
        assert "Unknown golden_id" in result.stderr

    def test_cli_missing_predictions_file_fails_cleanly(self, judge_examples):
        missing = str(Path(judge_examples["tmp_path"]) / "nope.json")
        result = subprocess.run(
            [sys.executable, "-m", "evaluation.run_judge",
             "--predictions", missing,
             "--golden-set", judge_examples["golden_path"],
             "--intents-yaml", judge_examples["taxonomy"],
             "--dry-run"],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 1
        assert "Cannot run judge." in result.stderr
        assert "no predictions file" in result.stderr
        # No traceback leaks to the user.
        assert "Traceback" not in result.stderr


# ---------------------------------------------------------------------------
# No real Gemini in tests
# ---------------------------------------------------------------------------

class TestNoRealCalls:
    def test_suite_only_uses_injected_client(self, judge_examples, tmp_path):
        client = FakeGeminiClient()
        RJ.run_judge(
            predictions_path=judge_examples["pred_path"],
            golden_set_path=judge_examples["golden_path"],
            intents_yaml=judge_examples["taxonomy"],
            model="fake-judge-model",
            limit=1,
            cache_dir=str(tmp_path / "cache"),
            results_dir=str(tmp_path / "res"),
            human_review_path=str(tmp_path / "h.csv"),
            client=client,
        )
        # We never construct a real GeminiJudgeClient during tests.
        assert client.calls == 1

    def test_no_api_key_literal_in_source(self):
        path = Path(__file__).resolve().parent.parent / "evaluation"
        for module_name in ("judge.py", "run_judge.py"):
            source = (path / module_name).read_text(encoding="utf-8")
            assert "AIza" not in source