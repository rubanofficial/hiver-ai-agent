"""
LLM-as-a-Judge evaluation framework for the AI support agent.

Gemini is used ONLY as an evaluator here -- never to generate Golden Set
labels and never as part of the production agent pipeline.  The judge scores
the *quality* of a generated customer-support reply and whether that reply is
*grounded* in the retrieved historical evidence.

Two judge dimensions (returned in one strict JSON object per example):

1. REPLY QUALITY
   correctness, helpfulness, relevance, clarity, escalation_appropriateness
   (each 1-5) plus an overall_reply_score (1-5).

2. EVIDENCE GROUNDING
   evidence_support, unsupported_claims, evidence_relevance (each 1-5) plus a
   grounding_score (1-5).

Every dimension uses a plain integer 1-5 rubric.  The judge receives ONLY:
customer_message, conversation_context, predicted_intent, retrieved_evidence,
generated_reply.  Human Golden Set labels (intent_label / escalation_label)
are NEVER passed to Gemini, so the Golden Evaluation Set cannot leak into the
judgment.

Reproducibility: the model is configurable via the GEMINI_MODEL environment
variable, the API key comes from GEMINI_API_KEY (never from source code),
temperature is fixed at 0, and responses are requested as strict JSON.

Cost/rate-limit safety: the runner (evaluation/run_judge.py) never judges all
examples by default; it uses --limit / --golden-ids, and locally caches every
example so reruns do not re-query Gemini for already-judged records.

The judge is NOT ground truth.  A subset of examples is designed to be
reviewed by a human (binary SUPPORTED / UNSUPPORTED grounding labels) so we
can measure judge-human agreement with raw agreement percentage and Cohen's
kappa.  Human grounding labels are independent annotations, never
auto-generated.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sklearn.metrics import cohen_kappa_score

from evaluation.evaluate import (
    EvaluationError,
    load_predictions,
    load_taxonomy,
    payload_records,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-2.0-flash"
DEFAULT_TEMPERATURE = 0.0
MAX_OUTPUT_TOKENS = 1024

SCORE_MIN = 1
SCORE_MAX = 5

REPLY_SCORE_FIELDS = [
    "correctness",
    "helpfulness",
    "relevance",
    "clarity",
    "escalation_appropriateness",
]
GROUNDING_SCORE_FIELDS = [
    "evidence_support",
    "unsupported_claims",
    "evidence_relevance",
]
OVERALL_REPLY_SCORE = "overall_reply_score"
GROUNDING_SCORE_KEY = "grounding_score"
REASON_KEY = "reason"

ALL_SCORE_FIELDS = (
    REPLY_SCORE_FIELDS
    + [OVERALL_REPLY_SCORE]
    + GROUNDING_SCORE_FIELDS
    + [GROUNDING_SCORE_KEY]
)
REQUIRED_FIELDS = ALL_SCORE_FIELDS + [REASON_KEY]

JUDGE_INPUT_FIELDS = [
    "golden_id",
    "customer_message",
    "conversation_context",
    "predicted_intent",
    "predicted_escalation",
    "predicted_escalation_reason",
    "retrieved_evidence",
    "generated_reply",
]

# Grounding labels used for human review / agreement.
SUPPORTED = "SUPPORTED"
UNSUPPORTED = "UNSUPPORTED"
GROUNDING_LABELS = [SUPPORTED, UNSUPPORTED]
# A grounding_score >= GROUNDING_THRESHOLD maps to SUPPORTED.
GROUNDING_THRESHOLD = 3

SCHEMA_VERSION = "1.0.0"


class JudgeError(Exception):
    """Raised when a judgment cannot be produced (blocking condition)."""


class ConsensusJudgeError(JudgeError):
    """Raised when Gemini's structured output cannot be validated."""


# ---------------------------------------------------------------------------
# Strict JSON parsing / validation
# ---------------------------------------------------------------------------

def strip_code_fence(text: str) -> str:
    """Remove a surrounding ```json ... ``` code fence if present."""
    stripped = text.strip()
    lines = stripped.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def parse_judge_json(raw_text: str) -> Dict[str, Any]:
    if not raw_text or not raw_text.strip():
        raise ConsensusJudgeError("Gemini returned an empty response.")
    cleaned = strip_code_fence(raw_text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ConsensusJudgeError(f"Gemini returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConsensusJudgeError("Gemini response must be a single JSON object.")
    return payload


def validate_scores(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Strictly validate the judge schema and the integer 1-5 scores."""
    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ConsensusJudgeError(
            f"Judge response is missing required field(s): {missing}."
        )
    problems: List[str] = []
    for field in ALL_SCORE_FIELDS:
        value = payload.get(field)
        # bool is a subclass of int in Python; reject it explicitly.
        if isinstance(value, bool) or not isinstance(value, int):
            problems.append(f"{field} is not an integer (got {value!r}).")
        elif not (SCORE_MIN <= value <= SCORE_MAX):
            problems.append(f"{field} is out of range {SCORE_MIN}-{SCORE_MAX} "
                            f"(got {value}).")
    reason = payload.get(REASON_KEY)
    if not isinstance(reason, str) or not reason.strip():
        problems.append(f"{REASON_KEY} must be a non-empty string.")
    if problems:
        raise ConsensusJudgeError("Invalid judge response:\n  - " + "\n  - ".join(problems))
    return dict(payload)


def parse_and_validate(raw_text: str) -> Dict[str, Any]:
    """Parse raw Gemini text into a fully validated judge result."""
    return validate_scores(parse_judge_json(raw_text))


# ---------------------------------------------------------------------------
# Derived values
# ---------------------------------------------------------------------------

def grounding_decision(grounding_score: int) -> str:
    """Map a 1-5 grounding score to the binary SUPPORTED/UNSUPPORTED label."""
    return SUPPORTED if grounding_score >= GROUNDING_THRESHOLD else UNSUPPORTED


def enrich_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Add convenient, clearly-derived fields (aliases + summaries)."""
    enriched = dict(result)
    enriched["overall_score"] = result[OVERALL_REPLY_SCORE]
    enriched["short_reason"] = result[REASON_KEY]
    enriched["grounding_decision"] = grounding_decision(result[GROUNDING_SCORE_KEY])
    enriched["reply_quality"] = {
        field: result[field] for field in REPLY_SCORE_FIELDS
    }
    enriched["evidence_grounding"] = {
        field: result[field] for field in GROUNDING_SCORE_FIELDS
    }
    return enriched


# ---------------------------------------------------------------------------
# Judge input (no human labels, ever)
# ---------------------------------------------------------------------------

def render_evidence(evidence: Any) -> str:
    if evidence is None:
        return ""
    if isinstance(evidence, str):
        return evidence
    if isinstance(evidence, list):
        return "\n".join(render_evidence(item) for item in evidence)
    if isinstance(evidence, dict):
        return json.dumps(evidence, ensure_ascii=False, sort_keys=True)
    return str(evidence)


def build_judge_input(record: Dict[str, Any],
                      prediction: Dict[str, Any]) -> Dict[str, Any]:
    """Build the judge input from a golden record + its agent prediction.

    Only the fields declared in JUDGE_INPUT_FIELDS are included.  Human Golden
    Set labels (intent_label / escalation_label / notes) are deliberately never
    copied into the judge input, so no labeling information can leak.
    """
    gid = prediction.get("golden_id")
    reply = prediction.get("predicted_reply")
    intent = prediction.get("predicted_intent")
    if not gid:
        raise JudgeError("Prediction has no golden_id.")
    if reply is None or not str(reply).strip():
        raise JudgeError(
            f"Prediction for {gid} has no generated reply (predicted_reply)."
        )
    if intent is None or not str(intent).strip():
        raise JudgeError(f"Prediction for {gid} has no predicted_intent.")

    record_dict = record if isinstance(record, dict) else {}
    judge_input = {
        "golden_id": gid,
        "customer_message": str(record_dict.get("customer_message") or ""),
        "conversation_context": str(record_dict.get("conversation_context") or ""),
        "predicted_intent": str(intent),
        "predicted_escalation": str(prediction.get("predicted_escalation") or "AUTO_HANDLE"),
        "predicted_escalation_reason": str(prediction.get("predicted_escalation_reason") or ""),
        "retrieved_evidence": render_evidence(prediction.get("retrieved_evidence")),
        "generated_reply": str(reply),
    }
    if not judge_input["customer_message"].strip():
        raise JudgeError(f"Golden record {gid} has no customer_message.")
    return judge_input


def load_judge_examples(
    golden_set_path: str,
    predictions_path: str,
    labels_path: Optional[str] = None,
    intents_yaml: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Join Golden Set records with agent predictions, in Golden Set order.

    Even when a labels file is provided it is never used here: the Golden Set
    is loaded without human labels so they can never reach the judge.
    """
    taxonomy = load_taxonomy(intents_yaml)
    golden = Path(golden_set_path)
    if not golden.exists():
        raise JudgeError(f"Golden Set not found: {golden}")
    with open(golden, "r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    records = payload_records(payload)

    predictions = load_predictions(predictions_path)
    pred_by_id = {p["golden_id"]: p for p in predictions}

    examples: List[Dict[str, Any]] = []
    missing: List[str] = []
    unknown: List[str] = []
    for rec in records:
        gid = rec.get("golden_id")
        pred = pred_by_id.get(gid)
        if pred is None:
            missing.append(str(gid))
            continue
        intent = pred.get("predicted_intent")
        if intent not in taxonomy:
            raise JudgeError(
                f"Prediction for {gid} has invalid predicted_intent {intent!r} "
                f"(must be one of the {len(taxonomy)} taxonomy intents)."
            )
        examples.append(build_judge_input(rec, pred))

    for gid, pred in pred_by_id.items():
        if gid not in {r["golden_id"] for r in records}:
            unknown.append(str(gid))

    if missing:
        raise JudgeError(
            "Judge cannot run: no agent prediction for "
            f"{len(missing)} Golden Set record(s): {missing[:20]}."
        )
    if unknown:
        raise JudgeError(
            f"Predictions contain {len(unknown)} golden_id(s) not in the "
            f"Golden Set: {unknown[:20]}."
        )
    return examples


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_judge_prompt(judge_input: Dict[str, Any]) -> str:
    """Build the single prompt asking Gemini for the strict judge JSON.

    The prompt contains ONLY the judge input fields -- never human labels.
    """
    return (
        "You are an evaluation judge for a Microsoft support agent. You are "
        "NOT ground truth: you provide a reasoned, structured assessment.\n\n"
        "Judge TWO dimensions of the generated reply and return a SINGLE "
        "valid JSON object (no markdown).\n\n"
        "DIMENSION 1 - REPLY QUALITY (each 1-5, integer where 5 is best):\n"
        "- correctness: does the reply correctly address the customer's problem?\n"
        "- helpfulness: does it provide useful, actionable support?\n"
        "- relevance: does it stay focused on the customer's issue?\n"
        "- clarity: is it understandable and professionally written?\n"
        "- escalation_appropriateness: does the reply avoid making unsafe or\n"
        "  unjustified claims when human handling is needed?\n"
        "- overall_reply_score: overall quality of the reply (1-5).\n\n"
        "DIMENSION 2 - EVIDENCE GROUNDING (each 1-5, integer where 5 is best):\n"
        "- evidence_support: are the important claims in the reply supported by\n"
        "  the retrieved evidence?\n"
        "- unsupported_claims: does the reply introduce facts, policies,\n"
        "  procedures, or guarantees NOT supported by the evidence? Higher is\n"
        "  better (5 = no unsupported claims).\n"
        "- evidence_relevance: is the retrieved evidence actually relevant to\n"
        "  the customer's issue?\n"
        "- grounding_score: overall degree to which the reply is grounded in the\n"
        "  retrieved evidence (1-5).\n\n"
        "RULES:\n"
        "- Every score MUST be an integer from 1 to 5.\n"
        "- Base scores ONLY on the provided input. Never invent facts.\n"
        "- Keep the reason concise.\n"
        "- Respond with exactly this JSON schema:\n"
        "{\n"
        '  "correctness": 1-5,\n'
        '  "helpfulness": 1-5,\n'
        '  "relevance": 1-5,\n'
        '  "clarity": 1-5,\n'
        '  "escalation_appropriateness": 1-5,\n'
        '  "overall_reply_score": 1-5,\n'
        '  "evidence_support": 1-5,\n'
        '  "unsupported_claims": 1-5,\n'
        '  "evidence_relevance": 1-5,\n'
        '  "grounding_score": 1-5,\n'
        '  "reason": "short explanation"\n'
        "}\n\n"
        "INPUT:\n"
        f"CUSTOMER MESSAGE:\n{judge_input['customer_message']}\n\n"
        f"CONVERSATION CONTEXT:\n{judge_input['conversation_context']}\n\n"
        f"PREDICTED INTENT: {judge_input['predicted_intent']}\n\n"
        f"PREDICTED ESCALATION: {judge_input.get('predicted_escalation') or 'AUTO_HANDLE'}\n"
        f"PREDICTED ESCALATION REASON: {judge_input.get('predicted_escalation_reason') or '(none)'}\n\n"
        "RETRIEVED EVIDENCE:\n"
        f"{judge_input['retrieved_evidence'] or '(none provided)'}\n\n"
        f"GENERATED REPLY:\n{judge_input['generated_reply']}"
    )


# ---------------------------------------------------------------------------
# Gemini client (the ONLY place Gemini is touched)
# ---------------------------------------------------------------------------

class GeminiJudgeClient:
    """Configures Google Gemini as (only) a JSON-scoring judge.

    Injection point: ``genai_module`` lets tests substitute a fake module.
    The API key is only ever read from the environment (or passed explicitly
    by the caller) and is never written to any file.
    """

    def __init__(self, model_name: Optional[str] = None,
                 api_key: Optional[str] = None,
                 genai_module: Any = None):
        if genai_module is None:
            import google.generativeai as gm
            genai_module = gm
        self.genai = genai_module
        self.model_name = (
            model_name
            or os.environ.get("GEMINI_MODEL")
            or DEFAULT_MODEL
        )
        self.api_key = (
            api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
        )
        if not self.api_key:
            raise JudgeError(
                "GEMINI_API_KEY is not set. Set the GEMINI_API_KEY environment "
                "variable before running the judge."
            )

    def generate(self, prompt: str) -> str:
        """Call Gemini and return the raw response text with rate-limit retries."""
        self.genai.configure(api_key=self.api_key)
        model = self.genai.GenerativeModel(self.model_name)
        max_retries = 25
        base_delay = 15.0
        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=self.genai.types.GenerationConfig(
                        response_mime_type="application/json",
                        temperature=DEFAULT_TEMPERATURE,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                    ),
                )
                return response.text
            except Exception as exc:
                exc_str = str(exc)
                if ("429" in exc_str or "quota" in exc_str.lower() or "resourceexhausted" in exc_str.lower()) and attempt < max_retries - 1:
                    import re
                    import time
                    delay = base_delay * (1.2 ** min(attempt, 10))
                    match = re.search(r"retry in (\d+(?:\.\d+)?)s", exc_str, re.IGNORECASE)
                    if match:
                        delay = max(delay, float(match.group(1)) + 5.0)
                    time.sleep(delay)
                else:
                    raise
        raise JudgeError("Gemini call failed after max retries.")


# ---------------------------------------------------------------------------
# Judgment execution
# ---------------------------------------------------------------------------

def _judge_payload(client: Any, judge_input: Dict[str, Any],
                   model_name: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
    """Run one judgment; returns ``(result, raw_response_text)``."""
    prompt = build_judge_prompt(judge_input)
    raw_text = client.generate(prompt)
    validated = parse_and_validate(raw_text)
    result = enrich_result(validated)
    result["golden_id"] = judge_input["golden_id"]
    result["model"] = model_name or getattr(client, "model_name", DEFAULT_MODEL)
    result["input"] = {k: judge_input[k] for k in JUDGE_INPUT_FIELDS}
    return result, raw_text


def judge_single(client: Any, judge_input: Dict[str, Any],
                 model_name: Optional[str] = None) -> Dict[str, Any]:
    """Judge one example via the injected ``client`` (mockable in tests)."""
    result, _ = _judge_payload(client, judge_input, model_name=model_name)
    return result


# ---------------------------------------------------------------------------
# Local caching (rate-limit / cost safety)
# ---------------------------------------------------------------------------

def input_fingerprint(judge_input: Dict[str, Any]) -> str:
    canonical = json.dumps(
        {k: judge_input[k] for k in JUDGE_INPUT_FIELDS},
        sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def cache_path_for(cache_dir, golden_id) -> Path:
    return Path(cache_dir) / f"{golden_id}.json"


def load_cache(cache_dir: str) -> Dict[str, Dict[str, Any]]:
    """Load cached judge results; silently ignore unreadable entries."""
    cache: Dict[str, Dict[str, Any]] = {}
    directory = Path(cache_dir)
    if not directory.exists():
        return cache
    for path in directory.glob("*.json"):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(entry, dict):
                continue
            gid = entry.get("golden_id")
            if gid and isinstance(entry.get("result"), dict):
                cache[str(gid)] = entry
        except (json.JSONDecodeError, OSError):
            continue
    return cache


def write_cache_entry(cache_dir: str, golden_id: str,
                      judge_input: Dict[str, Any], raw_response: str,
                      result: Dict[str, Any], model: Optional[str] = None) -> Path:
    path = cache_path_for(cache_dir, golden_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "schema_version": SCHEMA_VERSION,
        "golden_id": golden_id,
        "input_fingerprint": input_fingerprint(judge_input),
        "model": model or DEFAULT_MODEL,
        "judged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": {k: judge_input[k] for k in JUDGE_INPUT_FIELDS},
        "raw_response": raw_response,
        "result": result,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)
    return path


def is_cached(cache: Dict[str, Dict[str, Any]], judge_input: Dict[str, Any]) -> bool:
    entry = cache.get(str(judge_input["golden_id"]))
    if entry is None:
        return False
    return entry.get("input_fingerprint") == input_fingerprint(judge_input)


def run_judgments(
    client: Any,
    examples: List[Dict[str, Any]],
    cache_dir: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Judge examples, skipping any already cached with a matching fingerprint.

    Returns ``(results, stats)`` with stats = {"examples", "fresh", "cached"}.
    Raises ``JudgeError`` on the first unjudged example that fails validation.
    """
    cache = load_cache(cache_dir) if cache_dir else {}
    results: List[Dict[str, Any]] = []
    fresh = 0
    cached_hits = 0
    for judge_input in examples:
        result: Dict[str, Any]
        if cache_dir and is_cached(cache, judge_input):
            result = cache[str(judge_input["golden_id"])]["result"]
            cached_hits += 1
        else:
            result, raw_text = _judge_payload(client, judge_input,
                                              model_name=model_name)
            if cache_dir:
                write_cache_entry(
                    cache_dir,
                    judge_input["golden_id"],
                    judge_input,
                    raw_text,
                    result,
                    model=model_name,
                )
            fresh += 1
        results.append(result)
    stats = {"examples": len(examples), "fresh": fresh, "cached": cached_hits}
    return results, stats


# ---------------------------------------------------------------------------
# Human review + agreement
# ---------------------------------------------------------------------------

HUMAN_REVIEW_COLUMNS = [
    "golden_id",
    "customer_message",
    "conversation_context",
    "predicted_intent",
    "predicted_escalation",
    "predicted_escalation_reason",
    "retrieved_evidence",
    "generated_reply",
    "correctness",
    "helpfulness",
    "relevance",
    "clarity",
    "escalation_appropriateness",
    "overall_reply_score",
    "reason",
    "evidence_support",
    "unsupported_claims",
    "evidence_relevance",
    "grounding_score",
    "llm_grounding_decision",
    "human_grounding",
    "human_notes",
]


def _build_review_row(result: Dict[str, Any]) -> Dict[str, Any]:
    inp = result["input"]
    return {
        "golden_id": result["golden_id"],
        "customer_message": inp["customer_message"],
        "conversation_context": inp["conversation_context"],
        "predicted_intent": inp["predicted_intent"],
        "predicted_escalation": inp.get("predicted_escalation", ""),
        "predicted_escalation_reason": inp.get("predicted_escalation_reason", ""),
        "retrieved_evidence": inp["retrieved_evidence"],
        "generated_reply": inp["generated_reply"],
        "correctness": result["correctness"],
        "helpfulness": result["helpfulness"],
        "relevance": result["relevance"],
        "clarity": result["clarity"],
        "escalation_appropriateness": result["escalation_appropriateness"],
        "overall_reply_score": result[OVERALL_REPLY_SCORE],
        "reason": result[REASON_KEY],
        "evidence_support": result["evidence_support"],
        "unsupported_claims": result["unsupported_claims"],
        "evidence_relevance": result["evidence_relevance"],
        "grounding_score": result[GROUNDING_SCORE_KEY],
        "llm_grounding_decision": result["grounding_decision"],
        # Human annotations are intentionally left empty - never auto-created.
        "human_grounding": "",
        "human_notes": "",
    }


def write_human_review_csv(path: str, results: List[Dict[str, Any]]) -> Path:
    """Write the human-review CSV with empty human_grounding / human_notes."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HUMAN_REVIEW_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow(_build_review_row(result))
    return target


def read_human_grounding_csv(path: str) -> List[Tuple[str, str]]:
    """Read (golden_id, human_grounding) pairs; only returns labeled rows."""
    labeled: List[Tuple[str, str]] = []
    errors: List[str] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gid = (row.get("golden_id") or "").strip()
            label = (row.get("human_grounding") or "").strip().upper()
            if not label:
                continue  # not yet manually reviewed
            if label not in GROUNDING_LABELS:
                errors.append(f"{gid}: invalid human_grounding {label!r}")
                continue
            labeled.append((gid, label))
    if errors:
        raise JudgeError(
            "Invalid human grounding labels:\n  - " + "\n  - ".join(errors)
        )
    return labeled


def compute_agreement(llm_labels: List[str],
                      human_labels: List[str]) -> Dict[str, Any]:
    """Raw agreement % and Cohen's kappa between two binary annotators.

    Cohen's kappa is reported only when BOTH annotators actually use both
    labels (it is undefined when a rater shows no variability).  Human
    grounding labels must be independent annotations made on the human-review
    CSV; they are never generated for you.
    """
    if len(llm_labels) != len(human_labels):
        raise JudgeError(
            "Cannot compute agreement with mismatched label counts "
            f"({len(llm_labels)} vs {len(human_labels)})."
        )
    n = len(llm_labels)
    if n == 0:
        raise JudgeError("Cannot compute agreement without any labeled examples.")
    for label in set(llm_labels) | set(human_labels):
        if label not in GROUNDING_LABELS:
            raise JudgeError(f"Invalid grounding label: {label!r}.")

    matches = sum(1 for a, b in zip(llm_labels, human_labels) if a == b)
    raw_agreement = matches / n

    kappa: Optional[float] = None
    kappa_note = None
    if len(set(llm_labels)) > 1 and len(set(human_labels)) > 1:
        kappa = float(cohen_kappa_score(
            llm_labels, human_labels, labels=GROUNDING_LABELS
        ))
    else:
        kappa_note = (
            "At least one annotator used a single label; Cohen's kappa is "
            "undefined and was not computed."
        )
    if kappa is not None:
        kappa = round(kappa, 6)

    return {
        "n": n,
        "raw_agreement": round(raw_agreement, 6),
        "cohen_kappa": kappa,
        "cohen_kappa_note": kappa_note,
        "labels": GROUNDING_LABELS,
    }


def agreement_from_human_review(
    results: List[Dict[str, Any]], human_csv_path: str
) -> Dict[str, Any]:
    """Compute judge-human agreement against the (human-filled) review CSV.

    The LLM grounding decision is derived deterministically from the judge's
    grounding_score via ``grounding_decision``.
    """
    llm_by_id = {r["golden_id"]: r["grounding_decision"] for r in results}
    human_pairs = read_human_grounding_csv(human_csv_path)
    llm_labels: List[str] = []
    human_labels: List[str] = []
    missing_from_results: List[str] = []
    for gid, label in human_pairs:
        llm_label = llm_by_id.get(gid)
        if llm_label is None:
            missing_from_results.append(gid)
            continue
        llm_labels.append(llm_label)
        human_labels.append(label)
    if missing_from_results:
        raise JudgeError(
            "The following human-reviewed golden_ids have no judge result: "
            f"{missing_from_results[:20]}."
        )
    agreement = compute_agreement(llm_labels, human_labels)
    agreement["human_reviewed_examples"] = len(human_pairs)
    return agreement