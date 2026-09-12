"""
Tests for the Golden Evaluation Set agent-inference runner
(evaluation/run_agent.py).

Everything is fully mocked: a ``RecordingPipeline`` stub stands in for the real
Gemini-backed pipeline, so NO real Gemini calls are ever made and no
GEMINI_API_KEY is required.  Tiny synthetic Golden Sets are written into tmp
dirs.  The human label sentinels in the synthetic records prove that
``intent_label`` / ``escalation_label`` / ``notes`` never reach the agent.
"""

import csv
import json
from pathlib import Path

import pytest

from evaluation import run_agent as RA
from src.agent import (
    AgentResult,
    DraftReply,
    EscalationDecision,
    EscalationResult,
    Evidence,
    IntentClassificationResult,
)

AH = "AUTO_HANDLE"
EH = "ESCALATE_TO_HUMAN"
MODEL = "fake-model"

INTENTS = [
    "Technical Troubleshooting",
    "Product / Feature How-To",
    "Account & Login",
    "Billing & Payments",
]

SENTINEL_INTENT = "HUMAN_INTENT_SENTINEL"
SENTINEL_ESCALATION = "HUMAN_ESCALATION_SENTINEL"
SENTINEL_NOTE = "HUMAN_NOTE_SENTINEL"


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def record(gid, msg="My screen keeps flickering after the update.",
           ctx="MICROSOFT (1): We can help."):
    return {
        "golden_id": gid,
        "conversation_id": int(gid.split("-")[1]),
        "customer_message": msg,
        "conversation_context": ctx,
        "microsoft_responses": ["@user We can help."],
        "source_tweets": [{"tweet_id": 1, "text": msg}],
        "metadata": {},
        "intent_label": SENTINEL_INTENT,
        "escalation_label": SENTINEL_ESCALATION,
        "notes": SENTINEL_NOTE,
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def make_agent_result(msg, intent="Technical Troubleshooting",
                      confidence=0.93, decision=AH, reply=None, evidence=None):
    decision_enum = EscalationDecision(decision)
    return AgentResult(
        customer_message=msg,
        intent=intent,
        intent_result=IntentClassificationResult(intent=intent, confidence=confidence),
        retrieved_evidence=evidence or [
            Evidence(id="E1", text="Historical tweet E1", score=0.81,
                     metadata={"tweet_id": 77}),
        ],
        draft_reply=DraftReply(
            reply_text=reply or "Here is a grounded reply.",
            grounded=True, confidence=0.9,
        ),
        decision=decision_enum,
        escalation_reason=None if decision == AH else "Needs human review.",
        escalation_result=EscalationResult(decision=decision_enum),
    )


class RecordingPipeline:
    """Stub production pipeline that records every customer_message it sees."""

    def __init__(self, make_result=None):
        self.make_result = make_result
        self.calls = []

    def run(self, customer_message, top_k=3):
        self.calls.append(customer_message)
        if self.make_result is None:
            raise AssertionError(
                "RecordingPipeline.run called but no make_result provided."
            )
        outcome = self.make_result(customer_message)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def paths(tmp_path):
    return {
        "cache_dir": str(tmp_path / "agent_cache"),
        "predictions_path": str(tmp_path / "predictions.json"),
        "predictions_csv_path": str(tmp_path / "predictions.csv"),
        "failures_path": str(tmp_path / "failures.json"),
    }


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Schema: one successful prediction
# ---------------------------------------------------------------------------

class TestSinglePrediction:
    def test_writes_valid_prediction_schema(self, tmp_path):
        recs = [record("GOLDEN-0001")]
        p = paths(tmp_path)
        pipe = RecordingPipeline(make_result=make_agent_result)

        summary = RA.run_agent_run(recs, pipe, MODEL, taxonomy=INTENTS, **p)

        assert summary["total"] == 1
        assert summary["processed"] == 1
        assert summary["reused_from_cache"] == 0
        assert summary["failed"] == 0
        assert pipe.calls == [recs[0]["customer_message"]]

        preds = load_json(p["predictions_path"])
        assert len(preds) == 1
        pred = preds[0]
        assert set(pred) == set(RA.PREDICTION_FIELDS)
        assert pred["golden_id"] == "GOLDEN-0001"
        assert pred["predicted_intent"] in INTENTS
        assert isinstance(pred["predicted_intent_confidence"], float)
        assert pred["predicted_escalation"] in (AH, EH)
        assert isinstance(pred["predicted_reply"], str)
        assert isinstance(pred["retrieved_evidence"], list)

        ev = pred["retrieved_evidence"][0]
        for key in ("evidence_id", "source_tweet_id", "text", "score",
                    "source", "metadata"):
            assert key in ev
        assert ev["evidence_id"] == "E1"
        assert ev["source_tweet_id"] == 77
        assert isinstance(ev["score"], float)

        rows = list(csv.DictReader(open(p["predictions_csv_path"], encoding="utf-8")))
        assert len(rows) == 1
        assert rows[0]["golden_id"] == "GOLDEN-0001"
        assert rows[0]["predicted_intent"] == pred["predicted_intent"]
        assert rows[0]["evidence_ids"] == "E1"


# ---------------------------------------------------------------------------
# Multiple predictions, Golden Set order
# ---------------------------------------------------------------------------

class TestMultiplePredictions:
    def test_predictions_kept_in_golden_set_order(self, tmp_path):
        recs = [
            record("GOLDEN-0002", msg="two"),
            record("GOLDEN-0001", msg="one"),
            record("GOLDEN-0003", msg="three"),
        ]
        p = paths(tmp_path)
        pipe = RecordingPipeline(make_result=make_agent_result)

        summary = RA.run_agent_run(recs, pipe, MODEL, taxonomy=INTENTS, **p)

        assert summary["processed"] == 3
        preds = load_json(p["predictions_path"])
        assert [x["golden_id"] for x in preds] == [
            "GOLDEN-0002", "GOLDEN-0001", "GOLDEN-0003"
        ]
        assert pipe.calls == ["two", "one", "three"]


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------

class TestFailureHandling:
    def test_failed_prediction_recorded_and_not_dropped(self, tmp_path):
        recs = [
            record("GOLDEN-0001", msg="fail-this-one"),
            record("GOLDEN-0002", msg="ok-two"),
        ]

        def make(msg):
            if msg == "fail-this-one":
                raise RuntimeError("simulated agent failure")
            return make_agent_result(msg)

        p = paths(tmp_path)
        pipe = RecordingPipeline(make_result=make)
        summary = RA.run_agent_run(recs, pipe, MODEL, taxonomy=INTENTS, **p)

        assert summary["processed"] == 1
        assert summary["failed"] == 1
        assert summary["failure_ids"] == ["GOLDEN-0001"]

        # No fake prediction for the failed example.
        preds = load_json(p["predictions_path"])
        assert [x["golden_id"] for x in preds] == ["GOLDEN-0002"]

        # The failure is recorded against the golden_id.
        failures = load_json(p["failures_path"])
        assert failures["count"] == 1
        assert failures["failures"][0]["golden_id"] == "GOLDEN-0001"
        assert "simulated agent failure" in failures["failures"][0]["error"]

        # The failed example is retried (not silently skipped) on the next run.
        ok_pipe = RecordingPipeline(make_result=make_agent_result)
        resumed = RA.run_agent_run(recs, ok_pipe, MODEL, taxonomy=INTENTS, **p)
        assert resumed["processed"] == 1
        assert resumed["reused_from_cache"] == 1
        assert resumed["failed"] == 0
        assert ok_pipe.calls == ["fail-this-one"]
        assert len(load_json(p["predictions_path"])) == 2


# ---------------------------------------------------------------------------
# Cache / resume
# ---------------------------------------------------------------------------

class TestCacheAndResume:
    def test_second_run_reuses_cache_without_new_calls(self, tmp_path):
        recs = [record("GOLDEN-0001"), record("GOLDEN-0002")]
        p = paths(tmp_path)
        first = RecordingPipeline(make_result=make_agent_result)
        RA.run_agent_run(recs, first, MODEL, taxonomy=INTENTS, **p)
        assert first.calls == [recs[0]["customer_message"], recs[1]["customer_message"]]

        never = RecordingPipeline(
            make_result=lambda m: (_ for _ in ()).throw(
                AssertionError("must not call the pipeline again")
            )
        )
        summary = RA.run_agent_run(recs, never, MODEL, taxonomy=INTENTS, **p)

        assert never.calls == []
        assert summary["processed"] == 0
        assert summary["reused_from_cache"] == 2
        assert summary["failed"] == 0
        assert len(load_json(p["predictions_path"])) == 2

    def test_resume_processes_only_remaining(self, tmp_path):
        recs = [record("GOLDEN-0001", msg="one"), record("GOLDEN-0002", msg="two")]
        p = paths(tmp_path)
        RA.run_agent_run(recs, RecordingPipeline(make_result=make_agent_result),
                         MODEL, taxonomy=INTENTS, **p)

        recs_with_new = recs + [record("GOLDEN-0003", msg="three")]
        pipe = RecordingPipeline(make_result=make_agent_result)
        summary = RA.run_agent_run(recs_with_new, pipe, MODEL, taxonomy=INTENTS, **p)

        assert pipe.calls == ["three"]
        assert summary["processed"] == 1
        assert len(load_json(p["predictions_path"])) == 3

    def test_limit_only_takes_cached_free_examples(self, tmp_path):
        recs = [record("GOLDEN-0001", msg="one"), record("GOLDEN-0002", msg="two"),
                record("GOLDEN-0003", msg="three")]
        p = paths(tmp_path)
        RA.run_agent_run(recs[:2], RecordingPipeline(make_result=make_agent_result),
                         MODEL, taxonomy=INTENTS, **p)

        pipe = RecordingPipeline(make_result=make_agent_result)
        summary = RA.run_agent_run(recs, pipe, MODEL, taxonomy=INTENTS,
                                   limit=10, **p)

        assert pipe.calls == ["three"]
        assert summary["processed"] == 1
        assert summary["reused_from_cache"] == 2

    def test_input_change_invalidates_cached_prediction(self, tmp_path):
        original = [record("GOLDEN-0001", msg="original message")]
        p = paths(tmp_path)
        RA.run_agent_run(original, RecordingPipeline(make_result=make_agent_result),
                         MODEL, taxonomy=INTENTS, **p)

        changed = [record("GOLDEN-0001", msg="changed message now")]
        pipe = RecordingPipeline(make_result=make_agent_result)
        summary = RA.run_agent_run(changed, pipe, MODEL, taxonomy=INTENTS, **p)

        assert pipe.calls == ["changed message now"]
        assert summary["processed"] == 1
        assert summary["reused_from_cache"] == 0

    def test_retriever_change_invalidates_cached_prediction(self, tmp_path):
        p = paths(tmp_path)
        RA.run_agent_run(
            [record("GOLDEN-0001")],
            RecordingPipeline(make_result=make_agent_result),
            MODEL,
            taxonomy=INTENTS,
            embedding_index="evaluation/twcs_evidence_index.json",
            **p,
        )

        pipe = RecordingPipeline(make_result=make_agent_result)
        summary = RA.run_agent_run(
            [record("GOLDEN-0001")],
            pipe,
            MODEL,
            taxonomy=INTENTS,
            **p,
        )

        assert pipe.calls == [record("GOLDEN-0001")["customer_message"]]
        assert summary["processed"] == 1
        assert summary["reused_from_cache"] == 0


# ---------------------------------------------------------------------------
# Invalid AgentResult handling
# ---------------------------------------------------------------------------

class TestInvalidAgentResult:
    def test_none_result_is_a_recorded_failure(self, tmp_path):
        p = paths(tmp_path)
        pipe = RecordingPipeline(make_result=lambda m: None)

        summary = RA.run_agent_run([record("GOLDEN-0001")], pipe, MODEL,
                                   taxonomy=INTENTS, **p)

        assert summary["failed"] == 1
        assert load_json(p["predictions_path"]) == []
        failures = load_json(p["failures_path"])
        assert "AgentResult" in failures["failures"][0]["error"]

    def test_missing_escalation_decision_is_a_recorded_failure(self, tmp_path):
        def make(msg):
            result = make_agent_result(msg)
            result.decision = None
            result.escalation_reason = None
            return result

        p = paths(tmp_path)
        pipe = RecordingPipeline(make_result=make)
        summary = RA.run_agent_run([record("GOLDEN-0001")], pipe, MODEL,
                                   taxonomy=INTENTS, **p)

        assert summary["failed"] == 1
        failures = load_json(p["failures_path"])
        assert "escalation" in failures["failures"][0]["error"].lower()

    def test_decision_outside_allowed_values_is_a_recorded_failure(self, tmp_path):
        def make(msg):
            result = make_agent_result(msg)
            result.decision = "MAYBE"
            return result

        p = paths(tmp_path)
        pipe = RecordingPipeline(make_result=make)
        summary = RA.run_agent_run([record("GOLDEN-0001")], pipe, MODEL,
                                   taxonomy=INTENTS, **p)

        assert summary["failed"] == 1
        assert load_json(p["predictions_path"]) == []
        failures = load_json(p["failures_path"])
        assert "MAYBE" in failures["failures"][0]["error"]


# ---------------------------------------------------------------------------
# Human labels must NEVER reach the agent
# ---------------------------------------------------------------------------

class TestHumanLabelsNeverReachAgent:
    def test_agent_input_contains_no_human_labels(self):
        rec = record("GOLDEN-0001")
        inp = RA.agent_input(rec)

        assert set(inp) == {"customer_message", "conversation_context"}
        blob = json.dumps(inp)
        for sentinel in (SENTINEL_INTENT, SENTINEL_ESCALATION, SENTINEL_NOTE):
            assert sentinel not in blob

    def test_pipeline_receives_only_customer_message(self, tmp_path):
        rec = record("GOLDEN-0001", msg="real customer question")
        p = paths(tmp_path)
        pipe = RecordingPipeline(make_result=make_agent_result)

        RA.run_agent_run([rec], pipe, MODEL, taxonomy=INTENTS, **p)

        assert pipe.calls == ["real customer question"]
        for call in pipe.calls:
            for sentinel in (SENTINEL_INTENT, SENTINEL_ESCALATION, SENTINEL_NOTE):
                assert sentinel not in call

        pred = load_json(p["predictions_path"])[0]
        for key in ("intent_label", "escalation_label", "notes"):
            assert key not in pred


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_plan_marks_cached_and_new_calls(self, tmp_path):
        recs = [record("GOLDEN-0001"), record("GOLDEN-0002")]
        p = paths(tmp_path)
        RA.run_agent_run([recs[0]], RecordingPipeline(make_result=make_agent_result),
                         MODEL, taxonomy=INTENTS, **p)

        cache = RA.load_cache(p["cache_dir"])
        plan = RA.plan_dry_run(recs, cache, MODEL)

        assert ("GOLDEN-0001", "CACHED") in plan
        assert ("GOLDEN-0002", "NEW CALL") in plan

    def test_dry_run_writes_no_artifacts_and_needs_no_api_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        write_golden_set(tmp_path, [record("GOLDEN-0001")])
        p = paths(tmp_path)

        code = RA.main([
            "--golden-set", str(tmp_path / "golden_set.json"),
            "--cache-dir", p["cache_dir"],
            "--predictions", p["predictions_path"],
            "--predictions-csv", p["predictions_csv_path"],
            "--failures", p["failures_path"],
            "--model", MODEL,
            "--dry-run",
        ])

        assert code == 0
        assert not Path(p["predictions_path"]).exists()
        assert not Path(p["predictions_csv_path"]).exists()
        assert not Path(p["failures_path"]).exists()


# ---------------------------------------------------------------------------
# CLI / API key guard
# ---------------------------------------------------------------------------

class TestMissingApiKey:
    def test_real_run_fails_with_clear_message_when_key_missing(self, tmp_path,
                                                                monkeypatch, capsys):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        write_golden_set(tmp_path, [record("GOLDEN-0001")])
        p = paths(tmp_path)

        code = RA.main([
            "--golden-set", str(tmp_path / "golden_set.json"),
            "--cache-dir", p["cache_dir"],
            "--predictions", p["predictions_path"],
            "--predictions-csv", p["predictions_csv_path"],
            "--failures", p["failures_path"],
            "--model", MODEL,
        ])

        assert code == 1
        assert "GEMINI_API_KEY" in capsys.readouterr().err
        assert not Path(p["predictions_path"]).exists()


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------

class TestSelection:
    def test_golden_id_selects_a_single_record(self, tmp_path):
        recs = [record("GOLDEN-0001", msg="m-one"),
                record("GOLDEN-0002", msg="m-two")]
        p = paths(tmp_path)
        pipe = RecordingPipeline(make_result=make_agent_result)

        summary = RA.run_agent_run(recs, pipe, MODEL, taxonomy=INTENTS,
                                   golden_ids=["GOLDEN-0002"], **p)

        assert pipe.calls == ["m-two"]
        assert summary["selected"] == 1
        assert [x["golden_id"] for x in load_json(p["predictions_path"])] == ["GOLDEN-0002"]

    def test_unknown_golden_id_raises(self):
        recs = [record("GOLDEN-0001")]
        cache = {}
        with pytest.raises(RA.AgentRunnerError, match="GOLDEN-9999"):
            RA.select_records(recs, cache, MODEL, golden_ids=["GOLDEN-9999"])


# ---------------------------------------------------------------------------
# Prediction validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_rejects_invalid_intent(self):
        pred = {"golden_id": "GOLDEN-0001",
                "predicted_intent": "Not A Real Intent",
                "predicted_escalation": AH}
        with pytest.raises(RA.AgentRunnerError, match="taxonomy"):
            RA.validate_prediction(pred, INTENTS)

    def test_rejects_invalid_escalation(self):
        pred = {"golden_id": "GOLDEN-0001",
                "predicted_intent": INTENTS[0],
                "predicted_escalation": "MAYBE"}
        with pytest.raises(RA.AgentRunnerError, match="AUTO_HANDLE"):
            RA.validate_prediction(pred, INTENTS)

    def test_rejects_missing_golden_id(self):
        pred = {"predicted_intent": INTENTS[0], "predicted_escalation": AH}
        with pytest.raises(RA.AgentRunnerError, match="golden_id"):
            RA.validate_prediction(pred, INTENTS)


# ---------------------------------------------------------------------------
# export_from_cache
# ---------------------------------------------------------------------------

def _make_prediction(gid, intent="Technical Troubleshooting", escalation=AH):
    return {
        "golden_id": gid,
        "predicted_intent": intent,
        "predicted_intent_confidence": 0.9,
        "predicted_escalation": escalation,
        "predicted_escalation_reason": None,
        "predicted_reply": "A grounded reply.",
        "retrieved_evidence": [],
    }


def _write_cache_entry(cache_dir, gid, prediction, status="ok", error=None,
                       fingerprint="any-fingerprint", model=MODEL):
    """Write a raw cache file the way run_agent.py would."""
    entry = {
        "schema_version": "1.0.0",
        "generator": "evaluation/run_agent.py",
        "golden_id": gid,
        "status": status,
        "model": model,
        "input_fingerprint": fingerprint,
    }
    if status == "ok":
        entry["prediction"] = prediction
    if error:
        entry["error"] = error
    path = Path(cache_dir) / f"{gid}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry, indent=2), encoding="utf-8")


class TestExportFromCache:
    """export_from_cache reads existing cache files and writes predictions
    without calling Gemini or checking input fingerprints."""

    def test_exports_all_ok_entries(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        recs = [record("GOLDEN-0001"), record("GOLDEN-0002")]
        for r in recs:
            _write_cache_entry(cache_dir, r["golden_id"],
                               _make_prediction(r["golden_id"]))

        p = paths(tmp_path)
        summary = RA.export_from_cache(
            recs, cache_dir=cache_dir, taxonomy=INTENTS, **{
                k: v for k, v in p.items() if k != "cache_dir"
            }
        )

        assert summary["cache_ok"] == 2
        assert summary["predictions_exported"] == 2
        assert summary["cache_non_ok"] == 0
        assert summary["unattempted"] == 0

        preds = load_json(p["predictions_path"])
        assert len(preds) == 2
        assert {x["golden_id"] for x in preds} == {"GOLDEN-0001", "GOLDEN-0002"}

    def test_skips_non_ok_entries(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        recs = [record("GOLDEN-0001"), record("GOLDEN-0002")]
        _write_cache_entry(cache_dir, "GOLDEN-0001", _make_prediction("GOLDEN-0001"))
        _write_cache_entry(cache_dir, "GOLDEN-0002", None, status="error",
                           error="Something went wrong")

        p = paths(tmp_path)
        summary = RA.export_from_cache(
            recs, cache_dir=cache_dir, taxonomy=INTENTS, **{
                k: v for k, v in p.items() if k != "cache_dir"
            }
        )

        assert summary["cache_ok"] == 1
        assert summary["cache_non_ok"] == 1
        assert summary["predictions_exported"] == 1

        preds = load_json(p["predictions_path"])
        assert [x["golden_id"] for x in preds] == ["GOLDEN-0001"]

        failures = load_json(p["failures_path"])
        assert failures["count"] == 1
        assert failures["failures"][0]["golden_id"] == "GOLDEN-0002"

    def test_mismatched_fingerprint_does_not_block_export(self, tmp_path):
        """The whole point of export_from_cache: stale fingerprints are fine."""
        cache_dir = str(tmp_path / "cache")
        recs = [record("GOLDEN-0001")]
        # Write a cache entry with an obviously wrong fingerprint.
        _write_cache_entry(cache_dir, "GOLDEN-0001", _make_prediction("GOLDEN-0001"),
                           fingerprint="completely-wrong-fingerprint")

        p = paths(tmp_path)
        summary = RA.export_from_cache(
            recs, cache_dir=cache_dir, taxonomy=INTENTS, **{
                k: v for k, v in p.items() if k != "cache_dir"
            }
        )

        # Must still export the cached prediction.
        assert summary["predictions_exported"] == 1
        preds = load_json(p["predictions_path"])
        assert len(preds) == 1

    def test_unattempted_records_absent_from_predictions(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        # Only cache entry for GOLDEN-0001; GOLDEN-0002 has no cache file.
        recs = [record("GOLDEN-0001"), record("GOLDEN-0002")]
        _write_cache_entry(cache_dir, "GOLDEN-0001", _make_prediction("GOLDEN-0001"))

        p = paths(tmp_path)
        summary = RA.export_from_cache(
            recs, cache_dir=cache_dir, taxonomy=INTENTS, **{
                k: v for k, v in p.items() if k != "cache_dir"
            }
        )

        assert summary["unattempted"] == 1
        assert summary["predictions_exported"] == 1
        preds = load_json(p["predictions_path"])
        assert [x["golden_id"] for x in preds] == ["GOLDEN-0001"]

    def test_export_cache_cli_flag_writes_artifacts_without_api_key(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        cache_dir = str(tmp_path / "cache")
        recs_data = [record("GOLDEN-0001")]
        _write_cache_entry(cache_dir, "GOLDEN-0001", _make_prediction("GOLDEN-0001"))
        gs_path = write_golden_set(tmp_path, recs_data)
        p = paths(tmp_path)

        code = RA.main([
            "--golden-set", gs_path,
            "--cache-dir", cache_dir,
            "--predictions", p["predictions_path"],
            "--predictions-csv", p["predictions_csv_path"],
            "--failures", p["failures_path"],
            "--export-cache",
        ])

        assert code == 0
        preds = load_json(p["predictions_path"])
        assert len(preds) == 1
        assert preds[0]["golden_id"] == "GOLDEN-0001"

    def test_export_preserves_exact_cached_prediction_contents(self, tmp_path):
        cache_dir = str(tmp_path / "cache")
        recs = [record("GOLDEN-0001")]
        expected_pred = _make_prediction("GOLDEN-0001",
                                         intent="Billing & Payments",
                                         escalation=EH)
        expected_pred["predicted_escalation_reason"] = "Needs human review."
        expected_pred["predicted_reply"] = "We'll escalate this."
        expected_pred["retrieved_evidence"] = [
            {"evidence_id": "X1", "source_tweet_id": None, "text": "t",
             "score": 0.5, "source": "twcs_historical", "metadata": {}}
        ]
        _write_cache_entry(cache_dir, "GOLDEN-0001", expected_pred)

        p = paths(tmp_path)
        RA.export_from_cache(
            recs, cache_dir=cache_dir, taxonomy=INTENTS + ["Billing & Payments"],
            **{k: v for k, v in p.items() if k != "cache_dir"}
        )

        preds = load_json(p["predictions_path"])
        assert len(preds) == 1
        assert preds[0] == expected_pred