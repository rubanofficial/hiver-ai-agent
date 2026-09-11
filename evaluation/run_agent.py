"""
Golden Evaluation Set agent-inference runner.

Runs the EXISTING production support agent on the human-labeled Golden
Evaluation Set (``evaluation/golden_set.json``) and writes the agent's
predictions in the format expected by the evaluation framework.

::

    golden_set.json
        -> existing intent classifier
        -> existing evidence retriever
        -> existing reply generator
        -> existing escalation policy
        -> AgentResult
        -> predictions.json / predictions.csv

This runner NEVER calls the evaluator, NEVER computes metrics, and NEVER
generates, guesses or modifies any human label.  The agent receives ONLY the
customer-facing input that the existing pipeline consumes
(``customer_message``); the human ``intent_label`` / ``escalation_label`` /
``notes`` fields are never read into the agent call.  No fake predictions or
performance results are ever created.

Cost / rate-limit safety
------------------------
* ``python -m evaluation.run_agent`` processes the examples that still have
  no valid cached prediction (resume behavior).
* ``python -m evaluation.run_agent --limit 10`` processes at most 10 more.
* ``python -m evaluation.run_agent --golden-id GOLDEN-0001`` tests one example.
* ``python -m evaluation.run_agent --all`` explicitly re-runs EVERY example
  (ignores the cache).
* ``python -m evaluation.run_agent --dry-run`` prints the plan without calling
  Gemini and writes nothing.

Caching / resume
----------------
Every successful prediction is cached per ``golden_id`` under
``evaluation/agent_cache/<golden_id>.json`` together with a SHA-256
fingerprint of the agent input (customer message + conversation context +
model).  A cached prediction is reused only when its fingerprint still matches
the current Golden Set record, so edits to the input never serve stale
predictions.  Failed examples are recorded (never silently dropped) and are
retried on the next run.

Artifacts
---------
* ``evaluation/predictions.json``          - JSON list of valid predictions
  (consumed by ``evaluation.evaluate`` and ``evaluation.run_judge``).
* ``evaluation/predictions.csv``           - flattened copy for review.
* ``evaluation/predictions_failures.json`` - failures recorded against
  golden_id (never fabricated predictions).

Usage::

    export GEMINI_API_KEY=...
    export GEMINI_MODEL=gemini-2.0-flash        # optional

    python -m evaluation.run_agent --dry-run
    python -m evaluation.run_agent --golden-id GOLDEN-0001
    python -m evaluation.run_agent --limit 10
    python -m evaluation.run_agent              # resume on remaining examples
    python -m evaluation.run_agent --all        # re-run every example

This module performs the agent INFERENCE ONLY.  Performance metrics are
computed later by ``evaluation/evaluate.py``, which compares
``predictions.json`` against the human labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.agent import (
    AgentResult,
    EscalationDecision,
    EscalationPolicy,
    Evidence,
    EvidenceRetriever,
    IntentClassifier,
    ReplyGenerator,
    SupportPipeline,
)

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass

from evaluation.evaluate import load_taxonomy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_GOLDEN_SET_PATH = (
    PROJECT_ROOT / "evaluation" / "golden_set.labeled.json"
    if (PROJECT_ROOT / "evaluation" / "golden_set.labeled.json").exists()
    else PROJECT_ROOT / "evaluation" / "golden_set.json"
)
DEFAULT_INTENTS_YAML = PROJECT_ROOT / "config" / "intents.yaml"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "evaluation" / "predictions.json"
DEFAULT_PREDICTIONS_CSV_PATH = PROJECT_ROOT / "evaluation" / "predictions.csv"
DEFAULT_FAILURES_PATH = PROJECT_ROOT / "evaluation" / "predictions_failures.json"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "evaluation" / "agent_cache"
DEFAULT_EMBEDDING_INDEX = PROJECT_ROOT / "evaluation" / "twcs_evidence_index.json"

SCHEMA_VERSION = "1.0.0"
RUNNER_NAME = "evaluation/run_agent.py"

ESCALATION_VALUES = ["AUTO_HANDLE", "ESCALATE_TO_HUMAN"]

# The ONLY fields a golden record is allowed to expose to the agent. Human
# labels (intent_label / escalation_label / notes) are always excluded.
AGENT_INPUT_FIELDS = ("customer_message", "conversation_context")
HUMAN_LABEL_FIELDS = ("intent_label", "escalation_label", "notes")

# Prediction schema as documented by evaluation/evaluate.py plus the reason /
# evidence traceability requested by the runner.
PREDICTION_FIELDS = [
    "golden_id",
    "predicted_intent",
    "predicted_intent_confidence",
    "predicted_escalation",
    "predicted_escalation_reason",
    "predicted_reply",
    "retrieved_evidence",
]

CSV_FIELDS = [
    "golden_id",
    "predicted_intent",
    "predicted_intent_confidence",
    "predicted_escalation",
    "predicted_escalation_reason",
    "predicted_reply",
    "evidence_count",
    "evidence_ids",
]


class AgentRunnerError(Exception):
    """Raised when the agent runner cannot proceed (blocking/correctness)."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: str, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return p


def load_golden_records(golden_set_path: str) -> List[Dict[str, Any]]:
    """Load ``records`` from the Golden Set JSON payload (labels untouched)."""
    p = Path(golden_set_path)
    if not p.exists():
        raise AgentRunnerError(f"Golden Set not found: {p}")
    with open(p, "r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    records = payload.get("records")
    if not isinstance(records, list):
        raise AgentRunnerError("Golden Set payload has no 'records' list.")
    return records


# ---------------------------------------------------------------------------
# Agent input boundary (the ONLY view of a record the agent ever sees)
# ---------------------------------------------------------------------------

def agent_input(record: Dict[str, Any]) -> Dict[str, str]:
    """Return the agent-facing input for a golden record.

    Human labels are NEVER included here.  The current production pipeline
    consumes ``customer_message``; ``conversation_context`` is carried along
    for cache invalidation and for future pipeline stages.
    """
    message = record.get("customer_message")
    if not isinstance(message, str) or not message.strip():
        raise AgentRunnerError(
            f"Golden record {record.get('golden_id', '?')} has no customer_message."
        )
    return {
        "customer_message": message,
        "conversation_context": str(record.get("conversation_context") or ""),
    }


def agent_input_fingerprint(
    record: Dict[str, Any],
    model_name: str,
    embedding_index: Optional[str] = None,
) -> str:
    """SHA-256 fingerprint of the agent input + model + retriever config
    (cache invalidation)."""
    payload = {
        "input": agent_input(record),
        "model": model_name or "",
        "embedding_index": embedding_index or "",
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Pipeline construction (existing production components, unchanged)
# ---------------------------------------------------------------------------

def default_model_name() -> str:
    return os.environ.get("GEMINI_MODEL") or IntentClassifier.DEFAULT_MODEL


def build_pipeline(
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    embedding_index: Optional[str] = None,
) -> SupportPipeline:
    """Build the real production pipeline from its existing components.

    ``embedding_index`` (optional) is the FAISS embedding index the existing
    evidence retriever searches (e.g. ``AGENT_EMBEDDING_INDEX``).  When it is
    not provided the stock ``EvidenceRetriever()`` is used, which returns no
    evidence and therefore causes the existing escalation policy to escalate.
    """
    resolved_model = model_name or default_model_name()
    classifier = IntentClassifier(model_name=resolved_model, api_key=api_key)
    retriever: EvidenceRetriever
    if embedding_index:
        retriever = EvidenceRetriever.from_embedding_json(str(embedding_index))
    else:
        retriever = EvidenceRetriever()
    generator = ReplyGenerator(model_name=resolved_model, api_key=api_key)
    escalation = EscalationPolicy()
    return SupportPipeline(
        classifier=classifier,
        retriever=retriever,
        generator=generator,
        escalation=escalation,
    )


# ---------------------------------------------------------------------------
# AgentResult -> prediction
# ---------------------------------------------------------------------------

def _decision_str(decision: Any) -> Optional[str]:
    if decision is None:
        return None
    if isinstance(decision, EscalationDecision):
        return decision.value
    value = getattr(decision, "value", decision)
    return str(value) if value is not None else None


def evidence_to_dict(evidence: Evidence) -> Dict[str, Any]:
    """Serialize one Evidence record preserving traceability."""
    if not isinstance(evidence, Evidence):
        raise AgentRunnerError(
            f"retrieved_evidence must contain Evidence objects, got "
            f"{type(evidence).__name__}."
        )
    meta = dict(evidence.metadata or {})
    source_tweet_id = meta.get("tweet_id", meta.get("source_tweet_id"))
    return {
        "evidence_id": str(evidence.id),
        "source_tweet_id": source_tweet_id,
        "text": str(evidence.text),
        "score": round(float(evidence.score), 6),
        "source": evidence.source,
        "metadata": meta,
    }


def result_to_prediction(
    record: Dict[str, Any],
    result: AgentResult,
    model_name: str = "",
) -> Dict[str, Any]:
    """Convert a pipeline ``AgentResult`` into a validated prediction dict.

    Raises ``AgentRunnerError`` for any invalid/incomplete ``AgentResult`` so
    that broken agent output is never silently converted into a prediction.
    """
    gid = record.get("golden_id")
    if not gid:
        raise AgentRunnerError("Golden record is missing its golden_id.")

    if not isinstance(result, AgentResult):
        raise AgentRunnerError(
            f"Invalid AgentResult for {gid}: expected an AgentResult object, "
            f"got {type(result).__name__}."
        )

    if not result.intent:
        raise AgentRunnerError(
            f"Invalid AgentResult for {gid}: no predicted intent."
        )

    confidence: Optional[float] = None
    if result.intent_result is not None:
        try:
            confidence = round(float(result.intent_result.confidence), 6)
        except (TypeError, ValueError):
            confidence = None

    decision = _decision_str(result.decision)
    if decision not in ESCALATION_VALUES:
        raise AgentRunnerError(
            f"Invalid AgentResult for {gid}: missing or invalid escalation "
            f"decision {decision!r}."
        )

    reply: Optional[str] = None
    if result.draft_reply is not None:
        reply = getattr(result.draft_reply, "reply_text", None)
        if not isinstance(reply, str):
            raise AgentRunnerError(
                f"Invalid AgentResult for {gid}: draft reply text must be a "
                "string."
            )

    evidence = [
        evidence_to_dict(e) for e in (result.retrieved_evidence or [])
    ]
    return {
        "golden_id": gid,
        "predicted_intent": result.intent,
        "predicted_intent_confidence": confidence,
        "predicted_escalation": decision,
        "predicted_escalation_reason": result.escalation_reason
        if isinstance(result.escalation_reason, str)
        else None,
        "predicted_reply": reply,
        "retrieved_evidence": evidence,
    }


def validate_prediction(
    prediction: Dict[str, Any],
    taxonomy: List[str],
) -> None:
    """Strictly validate a prediction against the evaluator's schema.

    Raises ``AgentRunnerError`` when the prediction cannot be consumed by the
    evaluation framework or the judge.
    """
    gid = prediction.get("golden_id")
    if not gid:
        raise AgentRunnerError("Prediction is missing its golden_id.")

    intent = prediction.get("predicted_intent")
    if intent not in taxonomy:
        raise AgentRunnerError(
            f"predicted_intent {intent!r} (for {gid}) is not in the intent "
            "taxonomy."
        )

    escalation = prediction.get("predicted_escalation")
    if escalation not in ESCALATION_VALUES:
        raise AgentRunnerError(
            f"predicted_escalation {escalation!r} (for {gid}) must be one of "
            f"{ESCALATION_VALUES}."
        )

    reply = prediction.get("predicted_reply")
    if reply is not None and not isinstance(reply, str):
        raise AgentRunnerError(
            f"predicted_reply (for {gid}) must be a string or None."
        )

    evidence = prediction.get("retrieved_evidence")
    if not isinstance(evidence, list):
        raise AgentRunnerError(
            f"retrieved_evidence (for {gid}) must be a list."
        )


def process_record(
    record: Dict[str, Any],
    pipeline: Any,
    model_name: str,
) -> Dict[str, Any]:
    """Run the production pipeline on one golden record.

    Only ``agent_input(record)`` is available to the pipeline.  Any exception
    raised by the agent propagates so callers can record it as a failure.
    """
    inp = agent_input(record)
    result = pipeline.run(customer_message=inp["customer_message"])
    return result_to_prediction(record, result, model_name)


# ---------------------------------------------------------------------------
# Cache / resume
# ---------------------------------------------------------------------------

def _cache_entry_path(cache_dir: str, golden_id: str) -> Path:
    return Path(cache_dir) / f"{golden_id}.json"


def load_cache(cache_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load every per-example cache entry (best-effort)."""
    cache: Dict[str, Dict[str, Any]] = {}
    base = Path(cache_dir)
    if not base.is_dir():
        return cache
    for path in sorted(base.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                entry = json.load(f)
        except Exception:
            continue
        if isinstance(entry, dict) and entry.get("golden_id"):
            cache[entry["golden_id"]] = entry
    return cache


def _save_cache_entry(cache_dir: str, entry: Dict[str, Any]) -> Path:
    path = _cache_entry_path(cache_dir, str(entry["golden_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)
    return path


def save_cache_ok(
    cache_dir: str,
    record: Dict[str, Any],
    prediction: Dict[str, Any],
    model_name: str,
    embedding_index: Optional[str] = None,
) -> Path:
    """Persist a successful prediction with its input fingerprint."""
    entry = {
        "schema_version": SCHEMA_VERSION,
        "generator": RUNNER_NAME,
        "golden_id": record["golden_id"],
        "status": "ok",
        "model": model_name,
        "input_fingerprint": agent_input_fingerprint(
            record, model_name, embedding_index
        ),
        "prediction": prediction,
    }
    return _save_cache_entry(cache_dir, entry)


def save_cache_error(
    cache_dir: str,
    record: Dict[str, Any],
    error: str,
    model_name: str,
    embedding_index: Optional[str] = None,
) -> Path:
    """Persist a failure so it is never silently lost."""
    entry = {
        "schema_version": SCHEMA_VERSION,
        "generator": RUNNER_NAME,
        "golden_id": record["golden_id"],
        "status": "error",
        "model": model_name,
        "input_fingerprint": agent_input_fingerprint(
            record, model_name, embedding_index
        ),
        "error": error,
    }
    return _save_cache_entry(cache_dir, entry)


def valid_cached_prediction(
    cache_entry: Optional[Dict[str, Any]],
    record: Dict[str, Any],
    model_name: str,
    embedding_index: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return a reusable cached prediction, or None when it is stale/invalid.

    A cached prediction is reused only when it exists, was successful, and its
    input fingerprint still matches the current record + model + retriever.
    """
    if not isinstance(cache_entry, dict):
        return None
    if cache_entry.get("status") != "ok":
        return None
    if cache_entry.get("input_fingerprint") != agent_input_fingerprint(
        record, model_name, embedding_index
    ):
        return None
    prediction = cache_entry.get("prediction")
    if not isinstance(prediction, dict):
        return None
    return prediction


# ---------------------------------------------------------------------------
# Selection (cost / rate-limit and resume behavior)
# ---------------------------------------------------------------------------

def select_records(
    records: List[Dict[str, Any]],
    cache: Dict[str, Dict[str, Any]],
    model_name: str,
    limit: Optional[int] = None,
    golden_ids: Optional[List[str]] = None,
    all_examples: bool = False,
    ignore_cache: bool = False,
) -> List[Dict[str, Any]]:
    """Choose the candidate scope for this run (Golden Set order).

    * ``golden_ids``  -> exactly those records (in requested order).
    * ``all_examples`` -> every record (cache ignored when ``ignore_cache``).
    * otherwise        -> every record; the run loop reuses valid cached
      predictions and caps the number of NEW pipeline calls at ``limit``.

    ``lambda`/``cache``/``ignore_cache`` are kept in the signature so callers
    (and the dry-run planner) can mirror the run loop's reuse decisions.
    """
    if golden_ids:
        by_id = {r.get("golden_id"): r for r in records}
        missing = [g for g in golden_ids if g not in by_id]
        if missing:
            raise AgentRunnerError(
                f"Unknown golden_id(s): {missing}. They are not in the Golden Set."
            )
        return [by_id[g] for g in golden_ids]
    return list(records)


def plan_dry_run(
    records: List[Dict[str, Any]],
    cache: Dict[str, Dict[str, Any]],
    model_name: str,
    limit: Optional[int] = None,
    golden_ids: Optional[List[str]] = None,
    all_examples: bool = False,
    ignore_cache: bool = False,
    embedding_index: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Dry-run plan: list of (golden_id, "NEW CALL" | "CACHED").

    Mirrors the run loop exactly, so the plan matches what a real run would
    call Gemini for.  Never calls the pipeline and never writes anything.
    """
    selected = select_records(
        records, cache, model_name,
        limit=limit, golden_ids=golden_ids,
        all_examples=all_examples, ignore_cache=ignore_cache,
    )
    plan: List[Tuple[str, str]] = []
    planned_calls = 0
    for record in selected:
        gid = record.get("golden_id")
        if not ignore_cache and valid_cached_prediction(
            cache.get(gid), record, model_name, embedding_index
        ) is not None:
            plan.append((gid, "CACHED"))
            continue
        if limit is not None and limit > 0 and planned_calls >= limit:
            continue
        planned_calls += 1
        plan.append((gid, "NEW CALL"))
    return plan


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------

def build_predictions_artifact(
    records: List[Dict[str, Any]],
    cache: Dict[str, Dict[str, Any]],
    taxonomy: List[str],
) -> List[Dict[str, Any]]:
    """Rebuild the canonical predictions list from the whole cache.

    Only successful, schema-valid predictions are included (in Golden Set
    order).  Failed entries never become predictions.
    """
    predictions: List[Dict[str, Any]] = []
    for record in records:
        entry = cache.get(record.get("golden_id"))
        if not entry or entry.get("status") != "ok":
            continue
        prediction = entry.get("prediction")
        if not isinstance(prediction, dict):
            continue
        try:
            validate_prediction(prediction, taxonomy)
        except AgentRunnerError:
            continue
        predictions.append(prediction)
    return predictions


def write_predictions_json(path: str, predictions: List[Dict[str, Any]]) -> Path:
    """Write ``evaluation/predictions.json`` (top-level list for the evaluator)."""
    return _write_json(path, predictions)


def prediction_to_csv_row(prediction: Dict[str, Any]) -> Dict[str, Any]:
    evidence = prediction.get("retrieved_evidence") or []
    return {
        "golden_id": prediction["golden_id"],
        "predicted_intent": prediction["predicted_intent"],
        "predicted_intent_confidence": prediction.get(
            "predicted_intent_confidence"
        ),
        "predicted_escalation": prediction["predicted_escalation"],
        "predicted_escalation_reason": prediction.get(
            "predicted_escalation_reason"
        ) or "",
        "predicted_reply": prediction.get("predicted_reply") or "",
        "evidence_count": len(evidence),
        "evidence_ids": "; ".join(
            str(e.get("evidence_id")) for e in evidence
        ),
    }


def write_predictions_csv(path: str, predictions: List[Dict[str, Any]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for prediction in predictions:
            writer.writerow(prediction_to_csv_row(prediction))
    return p


def write_failures(path: str, failures: List[Dict[str, Any]]) -> Path:
    """Write failures against golden_id (never fabricated predictions)."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generator": RUNNER_NAME,
        "generated_at": _now_iso(),
        "count": len(failures),
        "failures": failures,
    }
    return _write_json(path, payload)


# ---------------------------------------------------------------------------
# Core run
# ---------------------------------------------------------------------------

def run_agent_run(
    records: List[Dict[str, Any]],
    pipeline: Any,
    model_name: str,
    cache_dir: str = str(DEFAULT_CACHE_DIR),
    predictions_path: str = str(DEFAULT_PREDICTIONS_PATH),
    predictions_csv_path: str = str(DEFAULT_PREDICTIONS_CSV_PATH),
    failures_path: str = str(DEFAULT_FAILURES_PATH),
    intents_yaml: Optional[str] = None,
    taxonomy: Optional[List[str]] = None,
    limit: Optional[int] = None,
    golden_ids: Optional[List[str]] = None,
    all_examples: bool = False,
    ignore_cache: bool = False,
    embedding_index: Optional[str] = None,
    delay_seconds: float = 0.0,
) -> Dict[str, Any]:
    """Run the agent over the selected example(s) and write the artifacts.

    ``pipeline`` is injectable for tests; it must expose ``run(customer_message)``
    (exactly like ``SupportPipeline.run``).  Returns a summary dict.
    """
    if taxonomy is None:
        taxonomy = load_taxonomy(intents_yaml)

    cache = load_cache(cache_dir)
    selected = select_records(
        records, cache, model_name,
        limit=limit, golden_ids=golden_ids,
        all_examples=all_examples, ignore_cache=ignore_cache,
    )

    processed = 0
    reused = 0
    failures: List[Dict[str, Any]] = []

    for record in selected:
        gid = record.get("golden_id")
        if gid is None:
            failures.append({"golden_id": None, "error": "Record has no golden_id."})
            continue

        # Reuse a matching cached prediction (Gemini is NOT called again).
        if not ignore_cache:
            cached = valid_cached_prediction(
                cache.get(gid), record, model_name, embedding_index
            )
            if cached is not None:
                reused += 1
                continue

        # --limit caps the number of NEW pipeline calls in this run.
        if limit is not None and limit > 0 and processed >= limit:
            continue

        max_retries = 5
        for attempt in range(max_retries):
            try:
                prediction = process_record(record, pipeline, model_name)
                validate_prediction(prediction, taxonomy)
                save_cache_ok(
                    cache_dir, record, prediction, model_name, embedding_index
                )
                cache[gid] = {
                    "status": "ok",
                    "input_fingerprint": agent_input_fingerprint(
                        record, model_name, embedding_index
                    ),
                    "prediction": prediction,
                }
                processed += 1
                print(
                    f"[agent] [{processed + reused}/{len(selected)}] {gid} -> "
                    f"{prediction['predicted_intent']} ({prediction['predicted_escalation']})",
                    flush=True,
                )
                if delay_seconds > 0:
                    time.sleep(delay_seconds)  # Pacing to stay under 15 RPM
                break
            except Exception as exc:
                err_str = f"{type(exc).__name__}: {exc}"
                if ("429" in err_str or "ResourceExhausted" in err_str or "quota" in err_str.lower()) and attempt < max_retries - 1:
                    wait_time = 12 + attempt * 4
                    print(
                        f"[agent] Rate limit on {gid} (attempt {attempt+1}/{max_retries}). "
                        f"Backing off for {wait_time}s...",
                        flush=True,
                    )
                    time.sleep(wait_time)
                    continue
                error = err_str
                save_cache_error(
                    cache_dir, record, error, model_name, embedding_index
                )
                cache[gid] = {"status": "error", "error": error}
                failures.append({"golden_id": gid, "error": error})
                print(f"[agent] ERROR on {gid}: {error}", flush=True)
                break

    predictions = build_predictions_artifact(records, cache, taxonomy)

    written = [
        write_predictions_json(predictions_path, predictions),
        write_predictions_csv(predictions_csv_path, predictions),
        write_failures(failures_path, failures),
    ]

    return {
        "total": len(records),
        "selected": len(selected),
        "processed": processed,
        "reused_from_cache": reused,
        "failed": len(failures),
        "failure_ids": [f["golden_id"] for f in failures],
        "predictions_in_artifact": len(predictions),
        "written": [str(p) for p in written],
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print(f"[agent] golden records            : {summary['total']}")
    print(f"[agent] selected for this run     : {summary['selected']}")
    print(f"[agent] processed via pipeline    : {summary['processed']}")
    print(f"[agent] reused from cache         : {summary['reused_from_cache']}")
    print(f"[agent] failed                    : {summary['failed']}")
    if summary["failure_ids"]:
        print(f"[agent] failure IDs               : {', '.join(map(str, summary['failure_ids']))}")
    print(f"[agent] predictions in artifact   : {summary['predictions_in_artifact']}")
    print("[agent] written:")
    for path in summary["written"]:
        print(f"  - {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the existing AI support agent on the Golden Evaluation "
                    "Set and produce predictions.json (inference only)."
    )
    parser.add_argument("--golden-set", default=str(DEFAULT_GOLDEN_SET_PATH))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS_PATH))
    parser.add_argument("--predictions-csv", default=str(DEFAULT_PREDICTIONS_CSV_PATH))
    parser.add_argument("--failures", default=str(DEFAULT_FAILURES_PATH))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--intents-yaml", default=str(DEFAULT_INTENTS_YAML))
    parser.add_argument(
        "--embedding-index", default=None,
        help="Path to a saved FAISS embedding index for evidence retrieval "
             "(also read from AGENT_EMBEDDING_INDEX).",
    )
    parser.add_argument(
        "--model", default=None,
        help="Gemini model (default: GEMINI_MODEL or "
             f"{IntentClassifier.DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max NEW examples to process in this run (resume-friendly).",
    )
    parser.add_argument(
        "--golden-id", nargs="*", default=None, dest="golden_id",
        help="Run only this golden_id (repeatable / space separated).",
    )
    parser.add_argument(
        "--golden-ids", nargs="*", default=None, dest="golden_ids",
        help="Run only these golden_ids (space separated).",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Re-run EVERY example, ignoring the cache.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print which examples would call Gemini; write nothing.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    args = parse_args(argv)

    golden_ids: List[str] = list(args.golden_id or []) + list(args.golden_ids or [])

    try:
        records = load_golden_records(args.golden_set)
    except AgentRunnerError as exc:
        print(f"[agent] Cannot run.\n{exc}", file=sys.stderr)
        return 1

    model_name = args.model or default_model_name()

    embedding_index = (
        args.embedding_index
        or os.environ.get("AGENT_EMBEDDING_INDEX")
        or (str(DEFAULT_EMBEDDING_INDEX) if DEFAULT_EMBEDDING_INDEX.exists() else None)
    )
    if embedding_index and not os.path.exists(embedding_index):
        print(
            f"[agent] Embedding index not found: {embedding_index}",
            file=sys.stderr,
        )
        return 1

    # --- dry run ------------------------------------------------------------
    if args.dry_run:
        cache = load_cache(args.cache_dir)
        try:
            plan = plan_dry_run(
                records, cache, model_name,
                limit=None if args.all else args.limit,
                golden_ids=golden_ids,
                all_examples=args.all, ignore_cache=args.all,
                embedding_index=embedding_index,
            )
        except AgentRunnerError as exc:
            print(f"[agent] Cannot run.\n{exc}", file=sys.stderr)
            return 1
        print("[agent] DRY RUN - no Gemini calls, no artifacts written.")
        planned = 0
        for gid, action in plan:
            if action == "NEW CALL":
                planned += 1
            print(f"  {gid}: {action}")
        print(f"[agent] planned Gemini calls    : {planned}")
        print(
            "[agent] Dry run only. To produce real predictions, drop "
            "--dry-run and set GEMINI_API_KEY."
        )
        return 0

    # --- real run: the API key is mandatory -------------------------------
    if not os.environ.get("GEMINI_API_KEY"):
        print(
            "[agent] GEMINI_API_KEY is not set. Set the GEMINI_API_KEY "
            "environment variable before running the agent (never hardcode "
            "credentials). Example (PowerShell):\n"
            "    $env:GEMINI_API_KEY = \"your-key-here\"\n"
            "The agent will not fabricate predictions or results without a "
            "real model call.",
            file=sys.stderr,
        )
        return 1


    if not embedding_index:
        print(
            "[agent] INFO: no embedding index configured; the existing evidence "
            "retriever returns no results, so the escalation policy will escalate "
            "each example. Set --embedding-index or AGENT_EMBEDDING_INDEX to "
            "enable evidence retrieval."
        )

    try:
        pipeline = build_pipeline(
            model_name=model_name,
            embedding_index=embedding_index,
        )
    except Exception as exc:
        print(f"[agent] Failed to build the agent pipeline.\n{exc}", file=sys.stderr)
        return 1

    try:
        summary = run_agent_run(
            records,
            pipeline,
            model_name,
            cache_dir=args.cache_dir,
            predictions_path=args.predictions,
            predictions_csv_path=args.predictions_csv,
            failures_path=args.failures,
            intents_yaml=args.intents_yaml,
            limit=None if args.all else args.limit,
            golden_ids=golden_ids,
            all_examples=args.all,
            ignore_cache=args.all,
            embedding_index=embedding_index,
            delay_seconds=2.0,
        )
    except AgentRunnerError as exc:
        print(f"[agent] Cannot run.\n{exc}", file=sys.stderr)
        return 1

    print_summary(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())